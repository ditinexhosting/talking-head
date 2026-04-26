# coding: utf-8

import sys
import io
import os
import pickle
import threading
import tempfile
from pathlib import Path

from flask import Flask, jsonify, send_file, request
from api.motion import build_motion_template, build_motion_from_keyframes, Keyframe
from api import keyframe_examples

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

_pipeline = None
_pipeline_lock = threading.Lock()
_infer_lock = threading.Lock()

SOURCE_IMAGE = str(ROOT / "assets" / "my_photo.png")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.config.inference_config import InferenceConfig
                from src.config.crop_config import CropConfig
                from src.live_portrait_pipeline import LivePortraitPipeline

                _pipeline = LivePortraitPipeline(
                    inference_cfg=InferenceConfig(),
                    crop_cfg=CropConfig(),
                )
                if os.path.exists(SOURCE_IMAGE):
                    _pipeline.precompute_source(SOURCE_IMAGE)
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _pipeline is not None})


@app.post("/api/warmup")
def warmup():
    get_pipeline()
    return jsonify({"status": "ok", "message": "models loaded"})


@app.post("/api/render")
def render():
    """
    Render a video of assets/my_photo.png driven by a hardcoded synthetic
    motion template (expression + head pose + eye blinks + lip sync).

    Returns video/mp4 streamed directly — nothing is written to disk permanently.

    Optional JSON body:
      {
        "n_frames": 150,   // number of frames (default 150 = 6 s at 25 fps)
        "fps": 25
      }
    """
    body = request.get_json(silent=True) or {}
    n_frames = max(10, min(int(body.get("n_frames", 150)), 600))
    fps = max(10, min(int(body.get("fps", 25)), 60))

    if not os.path.exists(SOURCE_IMAGE):
        return jsonify({"error": f"Source image not found: {SOURCE_IMAGE}"}), 404

    pipeline = get_pipeline()

    with _infer_lock:
        template_dct = build_motion_template(n_frames=n_frames, fps=fps)

        tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        try:
            pickle.dump(template_dct, tmp_pkl)
            tmp_pkl.close()

            with tempfile.TemporaryDirectory() as tmp_dir:
                from src.config.argument_config import ArgumentConfig
                args = ArgumentConfig(
                    source=SOURCE_IMAGE,
                    driving=tmp_pkl.name,
                    output_dir=tmp_dir,
                    flag_save_concat=False,
                )
                wfp, _ = pipeline.execute(args)

                buf = io.BytesIO()
                with open(wfp, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
        finally:
            os.unlink(tmp_pkl.name)

    return send_file(buf, mimetype="video/mp4", as_attachment=False)


@app.post("/api/render_keyframes")
def render_keyframes():
    """
    Render a video from keyframes with smooth interpolation.

    JSON body:
      {
        "keyframes": [
          {
            "frame_idx": 0,
            "pitch_deg": 0, "yaw_deg": 0, "roll_deg": 0,
            "scale": 1.393,
            "tx": 0.0, "ty": 0.1, "tz": 0.0,
            "eye_open_left": 0.30, "eye_open_right": 0.30,
            "eye_gaze_x": 0, "eye_gaze_y": 0,
            "brow_raise_left": 0, "brow_raise_right": 0,
            "mouth_open": 0, "mouth_smile": 0
          },
          {
            "frame_idx": 25,
            "pitch_deg": -5,
            "eye_open_left": 0.10, "eye_open_right": 0.10,
            "brow_raise_left": 0.5, "brow_raise_right": 0.5,
            "mouth_open": 0.3, "mouth_smile": 0.7
          },
          ...
        ],
        "n_frames": 150,
        "fps": 25
      }

    Full keyframe parameters:
      - Head: pitch_deg, yaw_deg, roll_deg, scale, tx, ty, tz
      - Eyes: eye_open_left, eye_open_right, eye_gaze_x, eye_gaze_y
      - Eyebrows: brow_raise_left, brow_raise_right, brow_inner_raise, brow_angle_left, brow_angle_right
      - Mouth: mouth_open, mouth_smile, mouth_pucker, mouth_x, mouth_y
      - Nose: nose_x, nose_y

    Returns video/mp4 streamed directly.
    """
    body = request.get_json(silent=True) or {}
    n_frames = max(10, min(int(body.get("n_frames", 150)), 600))
    fps = max(10, min(int(body.get("fps", 25)), 60))

    keyframes_data = body.get("keyframes", [])
    if not keyframes_data:
        return jsonify({"error": "No keyframes provided"}), 400

    # Convert dicts to Keyframe objects
    keyframes = []
    for kf_dict in keyframes_data:
        keyframes.append(Keyframe(
            frame_idx=int(kf_dict["frame_idx"]),
            # Head pose & scale
            pitch_deg=float(kf_dict.get("pitch_deg", 0.0)),
            yaw_deg=float(kf_dict.get("yaw_deg", 0.0)),
            roll_deg=float(kf_dict.get("roll_deg", 0.0)),
            scale=float(kf_dict.get("scale", 1.393)),
            tx=float(kf_dict.get("tx", 0.0)),
            ty=float(kf_dict.get("ty", 0.10)),
            tz=float(kf_dict.get("tz", 0.0)),
            # Eye control
            eye_open_left=float(kf_dict.get("eye_open_left", 0.30)),
            eye_open_right=float(kf_dict.get("eye_open_right", 0.30)),
            eye_gaze_x=float(kf_dict.get("eye_gaze_x", 0.0)),
            eye_gaze_y=float(kf_dict.get("eye_gaze_y", 0.0)),
            # Eyebrow control
            brow_raise_left=float(kf_dict.get("brow_raise_left", 0.0)),
            brow_raise_right=float(kf_dict.get("brow_raise_right", 0.0)),
            brow_inner_raise=float(kf_dict.get("brow_inner_raise", 0.0)),
            brow_angle_left=float(kf_dict.get("brow_angle_left", 0.0)),
            brow_angle_right=float(kf_dict.get("brow_angle_right", 0.0)),
            # Mouth control
            mouth_open=float(kf_dict.get("mouth_open", 0.0)),
            mouth_smile=float(kf_dict.get("mouth_smile", 0.0)),
            mouth_pucker=float(kf_dict.get("mouth_pucker", 0.0)),
            mouth_x=float(kf_dict.get("mouth_x", 0.0)),
            mouth_y=float(kf_dict.get("mouth_y", 0.0)),
            # Nose control
            nose_x=float(kf_dict.get("nose_x", 0.0)),
            nose_y=float(kf_dict.get("nose_y", 0.0)),
        ))

    if not os.path.exists(SOURCE_IMAGE):
        return jsonify({"error": f"Source image not found: {SOURCE_IMAGE}"}), 404

    pipeline = get_pipeline()

    with _infer_lock:
        template_dct = build_motion_from_keyframes(keyframes, n_frames=n_frames, fps=fps)

        tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        try:
            pickle.dump(template_dct, tmp_pkl)
            tmp_pkl.close()

            with tempfile.TemporaryDirectory() as tmp_dir:
                from src.config.argument_config import ArgumentConfig
                args = ArgumentConfig(
                    source=SOURCE_IMAGE,
                    driving=tmp_pkl.name,
                    output_dir=tmp_dir,
                    flag_save_concat=False,
                )
                wfp, _ = pipeline.execute(args)

                buf = io.BytesIO()
                with open(wfp, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
        finally:
            os.unlink(tmp_pkl.name)

    return send_file(buf, mimetype="video/mp4", as_attachment=False)




def _render_example_video(template_dct):
    """Helper to render an example video from a pre-built motion template dict."""
    if not isinstance(template_dct, dict):
        return jsonify({"error": "Motion template must be a dict"}), 400

    if not os.path.exists(SOURCE_IMAGE):
        return jsonify({"error": f"Source image not found: {SOURCE_IMAGE}"}), 404

    pipeline = get_pipeline()

    with _infer_lock:
        tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        try:
            pickle.dump(template_dct, tmp_pkl)
            tmp_pkl.close()

            with tempfile.TemporaryDirectory() as tmp_dir:
                from src.config.argument_config import ArgumentConfig
                args = ArgumentConfig(
                    source=SOURCE_IMAGE,
                    driving=tmp_pkl.name,
                    output_dir=tmp_dir,
                    flag_save_concat=False,
                )
                wfp, _ = pipeline.execute(args)

                buf = io.BytesIO()
                with open(wfp, "rb") as f:
                    buf.write(f.read())
                buf.seek(0)
        finally:
            os.unlink(tmp_pkl.name)

    return send_file(buf, mimetype="video/mp4", as_attachment=False)


@app.get("/api/render/test")
def render_test_movements():
    """Render test movement example."""
    end_frame, keyframes = keyframe_examples.talking_chin_up_movement()
    motions = build_motion_from_keyframes(keyframes, n_frames=end_frame, fps=25)
    return _render_example_video(motions)
