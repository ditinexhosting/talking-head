import os.path as osp
import shutil
import tempfile
from flask import Blueprint, Response, jsonify
from src.config.argument_config import ArgumentConfig
from api.pipeline import get_pipeline

animate_bp = Blueprint("animate", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")
_DRIVING = osp.join(_UPLOADS, "female.pkl")


@animate_bp.post("/animate")
def animate():
    if not osp.exists(_SOURCE):
        return jsonify({"error": f"source not found: {_SOURCE}"}), 400
    if not osp.exists(_DRIVING):
        return jsonify({"error": f"driving template not found: {_DRIVING}"}), 400

    tmp_dir = tempfile.mkdtemp()
    try:
        args = ArgumentConfig(
            source=_SOURCE,
            driving=_DRIVING,
            output_dir=tmp_dir,
        )

        pipeline = get_pipeline()
        wfp, _ = pipeline.execute(args)

        with open(wfp, "rb") as f:
            video_bytes = f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return Response(video_bytes, mimetype="video/mp4", headers={
        "Content-Disposition": "inline; filename=animation.mp4",
        "Content-Length": str(len(video_bytes)),
    })
