import json
import os
import os.path as osp
import pickle
import shutil
import tempfile
import threading
from flask import Blueprint, Response
from api.pipeline import get_pipeline
from api.utils.motion import add_blinks, add_talks, build_template, neutral_keyframes
from src.config.argument_config import ArgumentConfig

keyframes_bp = Blueprint("keyframes", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")

_FPS = 25
_PIPELINE_LOCK = threading.Lock()

_VISEME_SEQUENCE = [
        {
            "start": 0.0,
            "end": 0.01,
            "duration": 0.01,
            "viseme": "X",
            "description": "Silence / rest"
        },
        {
            "start": 0.01,
            "end": 0.05,
            "duration": 0.04,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.05,
            "end": 0.09,
            "duration": 0.04,
            "viseme": "G",
            "description": "Relaxed open mouth  (a, i)"
        },
        {
            "start": 0.09,
            "end": 0.16,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.16,
            "end": 0.51,
            "duration": 0.35,
            "viseme": "E",
            "description": "Tongue up  (l)"
        },
        {
            "start": 0.51,
            "end": 0.58,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.58,
            "end": 0.65,
            "duration": 0.07,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.65,
            "end": 0.72,
            "duration": 0.07,
            "viseme": "F",
            "description": "Rounded / puckered lips  (w, q)"
        },
        {
            "start": 0.72,
            "end": 0.79,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 0.79,
            "end": 0.93,
            "duration": 0.14,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 0.93,
            "end": 1.01,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 1.01,
            "end": 1.38,
            "duration": 0.37,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 1.38,
            "end": 1.45,
            "duration": 0.07,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 1.45,
            "end": 1.73,
            "duration": 0.28,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 1.73,
            "end": 1.94,
            "duration": 0.21,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 1.94,
            "end": 2.08,
            "duration": 0.14,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 2.08,
            "end": 2.34,
            "duration": 0.26,
            "viseme": "X",
            "description": "Silence / rest"
        },
        {
            "start": 2.34,
            "end": 2.53,
            "duration": 0.19,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 2.53,
            "end": 2.61,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 2.61,
            "end": 2.83,
            "duration": 0.22,
            "viseme": "C",
            "description": "Slightly open, no teeth  (d, g, k, n, r, s, t, y, z)"
        },
        {
            "start": 2.83,
            "end": 3.11,
            "duration": 0.28,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 3.11,
            "end": 3.19,
            "duration": 0.08,
            "viseme": "A",
            "description": "Closed lips  (m, b, p)"
        },
        {
            "start": 3.19,
            "end": 3.61,
            "duration": 0.42,
            "viseme": "B",
            "description": "Upper teeth on lower lip  (f, v)"
        },
        {
            "start": 3.61,
            "end": 3.66,
            "duration": 0.05,
            "viseme": "X",
            "description": "Silence / rest"
        }
    ]


@keyframes_bp.post("/animate/keyframes")
def animate_keyframes():
    keyframes = neutral_keyframes(seconds=1, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, _VISEME_SEQUENCE, fps=_FPS)
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

        with _PIPELINE_LOCK:
            wfp, _ = pipeline.execute(args)

        with open(wfp, "rb") as f:
            video_bytes = f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(video_bytes, mimetype="video/mp4", headers={
        "Content-Disposition": "inline; filename=blink_only_motion.mp4",
        "Content-Length": str(len(video_bytes)),
    })
