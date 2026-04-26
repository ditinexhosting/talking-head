# coding: utf-8

import os
import sys
import uuid
import shutil
import base64
import tempfile
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_file, after_this_request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))  # pipeline uses relative imports for resources

from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from api.emotions import EMOTIONS, EmotionPreset

app = Flask(__name__)

# ── singleton pipeline (loaded once, reused across requests) ──────────────────
_pipeline: Optional[LivePortraitPipeline] = None
_init_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize GPU inference to avoid OOM


def get_pipeline() -> LivePortraitPipeline:
    global _pipeline
    if _pipeline is None:
        with _init_lock:
            if _pipeline is None:
                app.logger.info("Loading LivePortrait models (one-time startup)...")
                _pipeline = LivePortraitPipeline(
                    inference_cfg=InferenceConfig(),
                    crop_cfg=CropConfig(),
                )
                app.logger.info("Models ready.")
    return _pipeline


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_upload_to_tmp(file_storage) -> str:
    suffix = Path(file_storage.filename).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_storage.save(tmp.name)
    return tmp.name


def _resolve_source(req) -> tuple[str, Optional[str]]:
    """Return (absolute_source_path, tmp_path_to_delete_or_None)."""
    if "source" in req.files:
        path = _save_upload_to_tmp(req.files["source"])
        return path, path

    data = req.get_json(silent=True) or {}

    if "source_base64" in data:
        raw = base64.b64decode(data["source_base64"])
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(raw)
        tmp.flush()
        return tmp.name, tmp.name

    if "source_path" in data:
        p = data["source_path"]
        p = p if os.path.isabs(p) else str(ROOT / p)
        return p, None

    raise ValueError("Provide source via file upload, source_base64, or source_path")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _pipeline is not None})


@app.get("/api/emotions")
def list_emotions():
    return jsonify({
        "emotions": [
            {"name": e.name, "description": e.description}
            for e in EMOTIONS.values()
        ]
    })


@app.post("/api/warmup")
def warmup():
    """Pre-load the model without running inference."""
    get_pipeline()
    return jsonify({"status": "ok", "model_loaded": True})


@app.post("/api/animate")
def animate():
    # ── resolve source image ──────────────────────────────────────────────────
    try:
        source_path, tmp_source = _resolve_source(request)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not os.path.exists(source_path):
        return jsonify({"error": f"Source image not found: {source_path}"}), 400

    # ── resolve emotion ───────────────────────────────────────────────────────
    form = request.form
    body = request.get_json(silent=True) or {}

    emotion_name = form.get("emotion") or body.get("emotion", "talking")
    if emotion_name not in EMOTIONS:
        if tmp_source:
            os.unlink(tmp_source)
        return jsonify({
            "error": f"Unknown emotion '{emotion_name}'",
            "available": list(EMOTIONS.keys()),
        }), 400

    preset: EmotionPreset = EMOTIONS[emotion_name]
    driving_path = str(ROOT / preset.driving_file)
    if not os.path.exists(driving_path):
        if tmp_source:
            os.unlink(tmp_source)
        return jsonify({"error": f"Driving template missing: {preset.driving_file}"}), 500

    # ── optional per-request overrides ────────────────────────────────────────
    multiplier = float(form.get("multiplier") or body.get("multiplier", preset.driving_multiplier))

    # ── build ArgumentConfig ──────────────────────────────────────────────────
    output_dir = tempfile.mkdtemp(prefix="lp_")
    args = ArgumentConfig(
        source=source_path,
        driving=driving_path,
        output_dir=output_dir,
        animation_region=preset.animation_region,
        driving_option=preset.driving_option,
        driving_multiplier=multiplier,
        flag_stitching=preset.flag_stitching,
        flag_pasteback=True,
        flag_do_crop=True,
        flag_relative_motion=True,
    )

    # ── run inference (serialized) ────────────────────────────────────────────
    pipeline = get_pipeline()
    with _infer_lock:
        try:
            pipeline.execute(args)
        except Exception as exc:
            app.logger.exception("Inference failed")
            shutil.rmtree(output_dir, ignore_errors=True)
            if tmp_source:
                os.unlink(tmp_source)
            return jsonify({"error": str(exc)}), 500

    if tmp_source:
        try:
            os.unlink(tmp_source)
        except OSError:
            pass

    # ── find output video ─────────────────────────────────────────────────────
    output_files = sorted(Path(output_dir).glob("*.mp4"))
    if not output_files:
        shutil.rmtree(output_dir, ignore_errors=True)
        return jsonify({"error": "No output video was generated"}), 500

    output_video = str(output_files[0])

    @after_this_request
    def _cleanup(response):
        shutil.rmtree(output_dir, ignore_errors=True)
        return response

    return send_file(
        output_video,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"{emotion_name}.mp4",
    )
