import json
import os
import os.path as osp
import pickle
import threading
from flask import Blueprint, Response
from api.pipeline import get_pipeline
from api.utils.motion import add_blinks, build_template, neutral_keyframes
from src.config.argument_config import ArgumentConfig

keyframes_bp = Blueprint("keyframes", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")
_OUTPUT_DIR = osp.join(_UPLOADS, "outputs")

_FPS = 25
_PIPELINE_LOCK = threading.Lock()


@keyframes_bp.post("/animate/keyframes")
def animate_keyframes():
    keyframes = neutral_keyframes(seconds=2, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], _FPS)

    # ── write template JSON to file ──────────────────────────────────────────
    json_path = osp.join(_UPLOADS, "blink_only_motion.json")
    with open(json_path, "w") as f:
        json.dump([kf.to_dict() for kf in keyframes], f, indent=2)

    # ── save template to a temp pkl so the pipeline can load it ──────────────
    tmp_pkl = osp.join(_UPLOADS, "_keyframe_blink_only_motion.pkl")
    with open(tmp_pkl, "wb") as f:
        pickle.dump(template, f)

    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        args = ArgumentConfig(
            source=_SOURCE,
            driving=tmp_pkl,
            output_dir=_OUTPUT_DIR,
        )

        pipeline = get_pipeline()
        inf_cfg = pipeline.live_portrait_wrapper.inference_cfg

        # Enable eye retargeting so c_eyes_lst=[0,0] drives the retargeting
        # network to close the eyes; restore original value after the call.
        with _PIPELINE_LOCK:
            prev = inf_cfg.flag_eye_retargeting
            inf_cfg.flag_eye_retargeting = True
            try:
                wfp, _ = pipeline.execute(args)
            finally:
                inf_cfg.flag_eye_retargeting = prev

        with open(wfp, "rb") as f:
            video_bytes = f.read()
    finally:
        if osp.exists(tmp_pkl):
            os.remove(tmp_pkl)

    return Response(video_bytes, mimetype="video/mp4", headers={
        "Content-Disposition": "inline; filename=blink_only_motion.mp4",
        "Content-Length": str(len(video_bytes)),
    })
