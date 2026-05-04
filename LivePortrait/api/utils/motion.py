import json
import random
import numpy as np

_EXP_DIM = 63  # 21 keypoints × 3 coords


class Keyframe:
    def __init__(self, frame_idx: int, eye: list[float] = None):
        self.frame = frame_idx
        self.eyes = eye if eye is not None else [0.4, 0.4]

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "tx": 0.0, "ty": 0.0, "scale": 1.0,
            "exp": [0.0] * _EXP_DIM,
            "eyes": self.eyes,
            "lip": 0.0,
        }


def neutral_keyframes(seconds: float, fps: int = 25) -> list["Keyframe"]:
    total_frames = int(seconds * fps)
    return [Keyframe(frame_idx=i) for i in range(total_frames)]


def add_blinks(keyframes: list["Keyframe"], fps: int = 25) -> list["Keyframe"]:
    from api.templates.blinks import blink  # local import — blinks.py imports Keyframe from here

    total_frames = len(keyframes)
    cursor = 0
    first = True

    while True:
        start = cursor + (int(0.5 * fps) if first else random.randint(4 * fps, 6 * fps))
        first = False
        blink_kfs, end_frame = blink(start_frame=start, type="neutral")

        if end_frame >= total_frames:
            break

        kf_frames = np.array([kf.frame for kf in blink_kfs], dtype=float)
        kf_eyes   = np.array([kf.eyes  for kf in blink_kfs], dtype=float)
        for i in range(blink_kfs[0].frame, end_frame + 1):
            left  = float(np.interp(i, kf_frames, kf_eyes[:, 0]))
            right = float(np.interp(i, kf_frames, kf_eyes[:, 1]))
            keyframes[i].eyes = [left, right]

        cursor = end_frame

    return keyframes


def euler_to_R(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    p, y, r = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg), np.deg2rad(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).T


def build_template(keyframes: list, fps: int) -> dict:
    keyframes = sorted(keyframes, key=lambda k: k["frame"])
    kf_frames = np.array([k["frame"] for k in keyframes], dtype=float)
    n_frames = int(keyframes[-1]["frame"]) + 1
    all_frames = np.arange(n_frames, dtype=float)

    def interp(field, default):
        vals = np.array([kf.get(field, default) for kf in keyframes], dtype=float)
        return np.interp(all_frames, kf_frames, vals)

    pitch = interp("pitch", 0.0)
    yaw   = interp("yaw",   0.0)
    roll  = interp("roll",  0.0)
    tx    = interp("tx",    0.0)
    ty    = interp("ty",    0.0)
    scale = interp("scale", 1.0)
    lip   = interp("lip",   0.0)

    exp_kf = np.array([kf.get("exp", [0.0] * _EXP_DIM) for kf in keyframes], dtype=float)
    exp = np.stack([np.interp(all_frames, kf_frames, exp_kf[:, i]) for i in range(_EXP_DIM)], axis=1)

    eyes_kf = np.array([kf.get("eyes", [0.4, 0.4]) for kf in keyframes], dtype=float)
    eyes = np.stack([np.interp(all_frames, kf_frames, eyes_kf[:, i]) for i in range(2)], axis=1)

    motion = []
    for i in range(n_frames):
        R = euler_to_R(pitch[i], yaw[i], roll[i])
        motion.append({
            "scale": np.array([[scale[i]]], dtype=np.float32),
            "R":     R[np.newaxis].astype(np.float32),
            "exp":   exp[i].reshape(1, 21, 3).astype(np.float32),
            "t":     np.array([[tx[i], ty[i], 0.0]], dtype=np.float32),
            "kp":    np.zeros((1, 21, 3), dtype=np.float32),
            "x_s":  np.zeros((1, 21, 3), dtype=np.float32),
        })

    return {
        "n_frames":    n_frames,
        "output_fps":  fps,
        "motion":      motion,
        "c_eyes_lst":  [eyes[i:i + 1].astype(np.float32) for i in range(n_frames)],
        "c_lip_lst":   [np.array([[lip[i]]], dtype=np.float32) for i in range(n_frames)],
    }


def template_to_json(template: dict) -> dict:
    def _serial(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    return json.loads(json.dumps({
        "n_frames":   template["n_frames"],
        "output_fps": template["output_fps"],
        "motion": [
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()}
            for m in template["motion"]
        ],
        "c_eyes_lst": [e.tolist() for e in template["c_eyes_lst"]],
        "c_lip_lst":  [l.tolist() for l in template["c_lip_lst"]],
    }, default=_serial))
