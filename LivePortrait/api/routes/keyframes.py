import os
import os.path as osp
import pickle
import uuid
import numpy as np
from flask import Blueprint, jsonify, request, send_file
from src.config.argument_config import ArgumentConfig
from api.pipeline import get_pipeline

keyframes_bp = Blueprint("keyframes", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")
_OUTPUT_DIR = osp.join(_UPLOADS, "outputs")

_EXP_DIM = 63  # 21 keypoints × 3 coords


def _euler_to_R(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """Build (3,3) rotation matrix matching camera.py get_rotation_matrix convention."""
    p, y, r = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T  # transpose matches camera.py permute(0,2,1)


def _interp(keyframes: list, field: str, default, all_frames: np.ndarray, kf_frames: np.ndarray):
    vals = np.array([kf.get(field, default) for kf in keyframes], dtype=float)
    return np.interp(all_frames, kf_frames, vals)


def _build_template(keyframes: list, fps: int) -> dict:
    keyframes = sorted(keyframes, key=lambda k: k["frame"])
    kf_frames = np.array([k["frame"] for k in keyframes], dtype=float)
    n_frames = int(keyframes[-1]["frame"]) + 1
    all_frames = np.arange(n_frames, dtype=float)

    pitch = _interp(keyframes, "pitch", 0.0, all_frames, kf_frames)
    yaw   = _interp(keyframes, "yaw",   0.0, all_frames, kf_frames)
    roll  = _interp(keyframes, "roll",  0.0, all_frames, kf_frames)
    tx    = _interp(keyframes, "tx",    0.0, all_frames, kf_frames)
    ty    = _interp(keyframes, "ty",    0.0, all_frames, kf_frames)
    scale = _interp(keyframes, "scale", 1.0, all_frames, kf_frames)
    lip   = _interp(keyframes, "lip",   0.0, all_frames, kf_frames)

    # exp: (K, 63) → interp each component → (n_frames, 63)
    exp_kf = np.array([kf.get("exp", [0.0] * _EXP_DIM) for kf in keyframes], dtype=float)
    exp = np.stack([np.interp(all_frames, kf_frames, exp_kf[:, i]) for i in range(_EXP_DIM)], axis=1)

    # eyes: (K, 2) → interp → (n_frames, 2)
    eyes_kf = np.array([kf.get("eyes", [0.4, 0.4]) for kf in keyframes], dtype=float)
    eyes = np.stack([np.interp(all_frames, kf_frames, eyes_kf[:, i]) for i in range(2)], axis=1)

    motion = []
    for i in range(n_frames):
        R = _euler_to_R(pitch[i], yaw[i], roll[i])
        motion.append({
            "scale": np.array([[scale[i]]], dtype=np.float32),
            "R":     R[np.newaxis].astype(np.float32),
            "exp":   exp[i].reshape(1, 21, 3).astype(np.float32),
            "t":     np.array([[tx[i], ty[i], 0.0]], dtype=np.float32),
            "kp":    np.zeros((1, 21, 3), dtype=np.float32),
            "x_s":  np.zeros((1, 21, 3), dtype=np.float32),
        })

    return {
        "n_frames": n_frames,
        "output_fps": fps,
        "motion": motion,
        "c_eyes_lst": [eyes[i:i + 1].astype(np.float32) for i in range(n_frames)],
        "c_lip_lst":  [np.array([[lip[i]]], dtype=np.float32) for i in range(n_frames)],
    }


def _validate_keyframes(keyframes) -> str | None:
    if not isinstance(keyframes, list) or len(keyframes) < 2:
        return "At least 2 keyframes required"
    for kf in keyframes:
        if not isinstance(kf.get("frame"), (int, float)):
            return "Each keyframe must have a numeric 'frame'"
        if "exp" in kf and len(kf["exp"]) != _EXP_DIM:
            return f"'exp' must have {_EXP_DIM} values (21 keypoints × 3)"
        if "eyes" in kf and len(kf["eyes"]) != 2:
            return "'eyes' must be [left, right] — 2 values"
    return None


@keyframes_bp.post("/animate/keyframes")
def animate_keyframes():
    """
    Body:
      {
        "fps": 25,
        "source": "<optional path, defaults to my_photo.png>",
        "keyframes": [
          {"frame": 0, "pitch": 0, "yaw": 0, "roll": 0,
           "tx": 0, "ty": 0, "scale": 1.0,
           "exp": [63 floats],
           "eyes": [0.4, 0.4], "lip": 0.0},
          ...
        ]
      }
    All keyframe fields except "frame" are optional.
    Values are linearly interpolated between keyframes.
    """
    body = request.get_json(silent=True) or {}

    err = _validate_keyframes(body.get("keyframes"))
    if err:
        return jsonify({"error": err}), 400

    fps = int(body.get("fps", 25))
    source = body.get("source", _SOURCE)
    if not osp.exists(source):
        return jsonify({"error": f"source not found: {source}"}), 400

    template = _build_template(body["keyframes"], fps)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    tmp_pkl = osp.join(_OUTPUT_DIR, f"kf_{uuid.uuid4().hex}.pkl")
    with open(tmp_pkl, "wb") as f:
        pickle.dump(template, f)

    try:
        args = ArgumentConfig(
            source=osp.abspath(source),
            driving=tmp_pkl,
            output_dir=_OUTPUT_DIR,
        )
        wfp, _ = get_pipeline().execute(args)
    finally:
        if osp.exists(tmp_pkl):
            os.remove(tmp_pkl)

    return send_file(wfp, mimetype="video/mp4", as_attachment=False)
