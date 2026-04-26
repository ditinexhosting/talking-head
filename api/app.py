# coding: utf-8

import copy
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))  # pipeline uses relative imports for resources

SOURCE_IMAGE = str(ROOT / "assets/my_photo.png")
TALKING_VIDEO = str(ROOT / "assets/talking.mp4")
TALKING_TEMPLATE = str(ROOT / "assets/talking.pkl")
OUTPUT_DIR = str(ROOT / "output")
DEFAULT_ANIMATION_DURATION_SECONDS = 5.0

DEFAULT_LIPSYNC_TEXT = (
    "Hello, and welcome to our session today, it is a pleasure to meet you. "
    "I am looking forward to learning more about your professional journey and discussing "
    "how your skills align with our company's mission."
)
LIP_EXPRESSION_INDICES = [6, 12, 14, 17, 19, 20]

from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.helper import is_square_video
from src.utils.io import dump, load, load_video
from src.utils.video import get_fps

app = Flask(__name__)

_pipeline: Optional[LivePortraitPipeline] = None
_init_lock = threading.Lock()
_template_lock = threading.Lock()
_inference_lock = threading.Lock()


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


def neutralize_template_lips(template: dict, neutral_exp) -> dict:
    """Freeze lip/mouth expression channels while preserving all other motion.

    The template keeps the original head pose, translation, scale, eye motion,
    and non-lip expression values. Only the mouth/lip expression keypoints are
    replaced with LivePortrait's neutral lip baseline.
    """
    first_exp = template["motion"][0]["exp"]
    neutral_exp = np.asarray(neutral_exp, dtype=first_exp.dtype)
    if neutral_exp.ndim == 2:
        neutral_exp = neutral_exp[None, ...]

    neutral_lips = neutral_exp[:, LIP_EXPRESSION_INDICES, :]
    for motion in template["motion"]:
        motion["exp"] = motion["exp"].copy()
        motion["exp"][:, LIP_EXPRESSION_INDICES, :] = neutral_lips

    n_frames = template["n_frames"]
    neutral_lip_ratio = np.array([0.0], dtype=np.float32)
    template["c_lip_lst"] = [neutral_lip_ratio.copy() for _ in range(n_frames)]
    template["c_d_lip_lst"] = [neutral_lip_ratio.copy() for _ in range(n_frames)]
    return template


def create_duration_template(
    template_path: str,
    output_path: str,
    duration_seconds: float,
) -> str:
    """Create a duration-specific copy of a motion template without mutating it.

    The template duration is controlled by resampling frame-level template lists.
    Shorter durations are compressed/truncated by sampling fewer frames; longer
    durations are stretched by sampling repeated/interpolated frame positions.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")

    template = load(template_path)
    source_frames = int(template["n_frames"])
    if source_frames <= 0:
        raise ValueError(f"Template has no frames: {template_path}")

    output_fps = int(template.get("output_fps", 25))
    target_frames = max(1, int(round(duration_seconds * output_fps)))
    frame_indices = np.rint(
        np.linspace(0, source_frames - 1, target_frames)
    ).astype(int)

    duration_template = copy.deepcopy(template)
    duration_template["n_frames"] = target_frames

    for key in ("motion", "c_eyes_lst", "c_lip_lst", "c_d_eyes_lst", "c_d_lip_lst"):
        value = template.get(key)
        if isinstance(value, list) and len(value) == source_frames:
            duration_template[key] = [copy.deepcopy(value[idx]) for idx in frame_indices]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(str(output), duration_template)
    return str(output)


def generate_talking_template(
    video_path: str = TALKING_VIDEO,
    output_path: str = TALKING_TEMPLATE,
) -> str:
    """Generate a LivePortrait .pkl motion template from assets/female.mp4 with neutral lips."""
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
    template = neutralize_template_lips(template, neutral_exp=inf_cfg.lip_array)

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
    """Create assets/female.pkl from assets/female.mp4."""
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


@app.post("/api/animate")
def animate():
    """Animate SOURCE_IMAGE and return an MP4 with the requested duration.

    JSON body options:
        {"duration_seconds": 5}
        {"duration": 7.5}
    """
    try:
        if not Path(SOURCE_IMAGE).exists():
            raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE}")

        if not Path(TALKING_TEMPLATE).exists():
            app.logger.info("Driving template missing; generating %s", TALKING_TEMPLATE)
            with _template_lock:
                if not Path(TALKING_TEMPLATE).exists():
                    generate_talking_template()

        request_data = request.get_json(silent=True) or {}
        duration_seconds = float(
            request_data.get(
                "duration_seconds",
                request_data.get("duration", DEFAULT_ANIMATION_DURATION_SECONDS),
            )
        )

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration_token = str(duration_seconds).replace(".", "p")
        duration_template = output_dir / f"female_{duration_token}s.pkl"
        create_duration_template(
            template_path=TALKING_TEMPLATE,
            output_path=str(duration_template),
            duration_seconds=duration_seconds,
        )

        pipeline = get_pipeline()
        args = ArgumentConfig(
            source=SOURCE_IMAGE,
            driving=str(duration_template),
            output_dir=OUTPUT_DIR,
            flag_save_concat=False,
        )

        with _inference_lock:
            video_path, _ = pipeline.execute(args)

        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Generated video not found: {video_file}")

        video_bytes = video_file.read_bytes()

    except Exception as exc:
        app.logger.exception("Failed to animate portrait")
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500

    return Response(
        video_bytes,
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename={video_file.name}",
            "Content-Length": str(len(video_bytes)),
            "X-Output-Path": str(video_file.resolve()),
            "X-Duration-Seconds": str(duration_seconds),
        },
    )


def _text_to_wav(text: str, output_wav: str) -> str:
    """Convert *text* to a 16-kHz mono WAV using gTTS + ffmpeg."""
    from gtts import gTTS

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    gTTS(text=text, lang="en", slow=False).save(mp3_path)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "warning",
                "-i", mp3_path,
                "-ar", "16000",
                "-ac", "1",
                output_wav,
            ],
            check=True,
        )
    finally:
        os.unlink(mp3_path)

    return output_wav


@app.post("/api/animate-with-lipsync")
def animate_with_lipsync():
    """Generate a lip-synced talking-head video.

    Runs LivePortrait to animate the source image, synthesises speech from
    the hardcoded greeting text via gTTS, then applies MuseTalk lip-sync
    so the mouth matches the audio.

    JSON body options (same as /api/animate):
        {"duration_seconds": 5}
        {"text": "Custom text to speak"}   # optional override
    """
    try:
        from api.musetalk_lipsync import apply_lipsync

        if not Path(SOURCE_IMAGE).exists():
            raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE}")

        if not Path(TALKING_TEMPLATE).exists():
            app.logger.info("Driving template missing; generating %s", TALKING_TEMPLATE)
            with _template_lock:
                if not Path(TALKING_TEMPLATE).exists():
                    generate_talking_template()

        request_data = request.get_json(silent=True) or {}
        duration_seconds = float(
            request_data.get(
                "duration_seconds",
                request_data.get("duration", DEFAULT_ANIMATION_DURATION_SECONDS),
            )
        )
        text = request_data.get("text", DEFAULT_LIPSYNC_TEXT)

        # ── Step 1: Generate LivePortrait animated video ──────────────────
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration_token = str(duration_seconds).replace(".", "p")
        duration_template = output_dir / f"female_{duration_token}s.pkl"
        create_duration_template(
            template_path=TALKING_TEMPLATE,
            output_path=str(duration_template),
            duration_seconds=duration_seconds,
        )

        pipeline = get_pipeline()
        args = ArgumentConfig(
            source=SOURCE_IMAGE,
            driving=str(duration_template),
            output_dir=OUTPUT_DIR,
            flag_save_concat=False,
        )

        with _inference_lock:
            lp_video_path, _ = pipeline.execute(args)

        lp_video_file = Path(lp_video_path)
        if not lp_video_file.exists():
            raise FileNotFoundError(f"LivePortrait video not found: {lp_video_file}")

        app.logger.info("LivePortrait video: %s", lp_video_file)

        # ── Step 2: Text → speech ─────────────────────────────────────────
        wav_path = str(output_dir / "tts_speech.wav")
        _text_to_wav(text, wav_path)
        app.logger.info("TTS audio: %s", wav_path)

        # ── Step 3: MuseTalk lip-sync ──────────────────────────────────────
        lipsync_out = str(output_dir / f"lipsync_{lp_video_file.stem}.mp4")
        with _inference_lock:
            apply_lipsync(
                video_path=str(lp_video_file),
                audio_path=wav_path,
                output_path=lipsync_out,
            )

        video_file = Path(lipsync_out)
        if not video_file.exists():
            raise FileNotFoundError(f"Lip-synced video not found: {video_file}")

        video_bytes = video_file.read_bytes()

    except Exception as exc:
        app.logger.exception("Failed to animate with lip-sync")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return Response(
        video_bytes,
        mimetype="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename={video_file.name}",
            "Content-Length": str(len(video_bytes)),
            "X-Output-Path": str(video_file.resolve()),
            "X-Duration-Seconds": str(duration_seconds),
            "X-Text": text[:200],
        },
    )
