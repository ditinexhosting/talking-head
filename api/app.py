# coding: utf-8

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
from flask import Flask, jsonify

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))  # pipeline uses relative imports for resources

SOURCE_IMAGE = str(ROOT / "assets/my_photo.png")
TALKING_VIDEO = str(ROOT / "assets/female.mp4")
TALKING_TEMPLATE = str(ROOT / "assets/female.pkl")

from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.helper import is_square_video
from src.utils.io import dump, load_video
from src.utils.video import get_fps

app = Flask(__name__)

_pipeline: Optional[LivePortraitPipeline] = None
_init_lock = threading.Lock()
_template_lock = threading.Lock()


def get_pipeline() -> LivePortraitPipeline:
    """Load and cache LivePortrait models for optional startup preloading."""
    global _pipeline

    if _pipeline is None:
        with _init_lock:
            if _pipeline is None:
                app.logger.info("Loading LivePortrait models...")
                _pipeline = LivePortraitPipeline(
                    inference_cfg=InferenceConfig(),
                    crop_cfg=CropConfig(),
                )
                _pipeline.precompute_source(SOURCE_IMAGE)
                app.logger.info("LivePortrait models ready.")

    return _pipeline


def generate_talking_template(
    video_path: str = TALKING_VIDEO,
    output_path: str = TALKING_TEMPLATE,
) -> str:
    """Generate a LivePortrait .pkl motion template from assets/talking.mp4."""
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Driving video not found: {video}")

    pipeline = get_pipeline()
    inf_cfg = pipeline.live_portrait_wrapper.inference_cfg

    output_fps = int(get_fps(str(video)))
    app.logger.info("Loading driving video from %s at %s FPS", video, output_fps)
    driving_rgb_lst = load_video(str(video))

    if inf_cfg.flag_crop_driving_video or not is_square_video(str(video)):
        ret_d = pipeline.cropper.crop_driving_video(driving_rgb_lst)
        driving_rgb_crop_lst = ret_d["frame_crop_lst"]
        driving_lmk_crop_lst = ret_d["lmk_crop_lst"]
        app.logger.info("Cropped %s driving frames", len(driving_rgb_crop_lst))
    else:
        driving_rgb_crop_lst = driving_rgb_lst
        driving_lmk_crop_lst = pipeline.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)

    driving_rgb_crop_256x256_lst = [
        cv2.resize(frame, (256, 256)) for frame in driving_rgb_crop_lst
    ]
    c_d_eyes_lst, c_d_lip_lst = pipeline.live_portrait_wrapper.calc_ratio(
        driving_lmk_crop_lst
    )
    I_d_lst = pipeline.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
    template = pipeline.make_motion_template(
        I_d_lst,
        c_d_eyes_lst,
        c_d_lip_lst,
        output_fps=output_fps,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(str(output), template)
    app.logger.info("Saved talking motion template to %s", output)
    return str(output)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _pipeline is not None,
    })


@app.post("/api/generate-talking-template")
def generate_talking_template_endpoint():
    """Create assets/talking.pkl from assets/talking.mp4."""
    try:
        with _template_lock:
            template_path = generate_talking_template()
    except Exception as exc:
        app.logger.exception("Failed to generate talking template")
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500

    return jsonify({
        "status": "ok",
        "template": template_path,
    })
