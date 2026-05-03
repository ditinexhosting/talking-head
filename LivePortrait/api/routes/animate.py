import os.path as osp
from flask import Blueprint, jsonify, send_file
from src.config.argument_config import ArgumentConfig
from api.pipeline import get_pipeline

animate_bp = Blueprint("animate", __name__)

_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "..", "uploads"))
_SOURCE = osp.join(_UPLOADS, "my_photo.png")
_DRIVING = osp.join(_UPLOADS, "female.pkl")
_OUTPUT_DIR = osp.join(_UPLOADS, "outputs")


@animate_bp.post("/animate")
def animate():
    if not osp.exists(_SOURCE):
        return jsonify({"error": f"source not found: {_SOURCE}"}), 400
    if not osp.exists(_DRIVING):
        return jsonify({"error": f"driving template not found: {_DRIVING}"}), 400

    args = ArgumentConfig(
        source=_SOURCE,
        driving=_DRIVING,
        output_dir=_OUTPUT_DIR,
    )

    pipeline = get_pipeline()
    wfp, _ = pipeline.execute(args)

    return send_file(wfp, mimetype="video/mp4", as_attachment=False)
