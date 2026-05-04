import json
import os
import os.path as osp
import pickle
import shutil
import tempfile
import threading
from flask import Blueprint, Response
from api.pipeline import get_pipeline
from api.utils.motion import add_blinks, build_template, neutral_keyframes
from src.config.argument_config import ArgumentConfig

keyframes_bp = Blueprint("keyframes", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")

_FPS = 25
_PIPELINE_LOCK = threading.Lock()


@keyframes_bp.post("/animate/keyframes")
def animate_keyframes():
    keyframes = neutral_keyframes(seconds=2, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], _FPS)

    json_path = osp.join(_UPLOADS, "blink_only_motion.json")
    with open(json_path, "w") as f:
        json.dump([kf.to_dict() for kf in keyframes], f, indent=2)

    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_pkl = osp.join(tmp_dir, "blink_only_motion.pkl")
        output_dir = osp.join(tmp_dir, "outputs")
        os.makedirs(output_dir)

        with open(tmp_pkl, "wb") as f:
            pickle.dump(template, f)

        args = ArgumentConfig(
            source=_SOURCE,
            driving=tmp_pkl,
            output_dir=output_dir,
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
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(video_bytes, mimetype="video/mp4", headers={
        "Content-Disposition": "inline; filename=blink_only_motion.mp4",
        "Content-Length": str(len(video_bytes)),
    })
