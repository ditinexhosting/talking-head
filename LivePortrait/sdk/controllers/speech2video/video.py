from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
import time

import cv2

from src.config.argument_config import ArgumentConfig
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import neutral_keyframes, build_template, add_blinks, add_talks

logger = logging.getLogger(__name__)

_pipeline_lock = threading.Lock()  # pipeline is not safe for concurrent execute_streaming calls

_IDLE_FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "idle_frame.png")


def capture_and_save_idle_frame() -> None:
    """Run once to capture the first LivePortrait frame and save it to disk."""
    gen = liveportrait_frame_gen()
    frame_rgb = next(gen)
    gen.close()
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(_IDLE_FRAME_PATH, bgr)
    logger.info("[video] idle frame saved to %s", _IDLE_FRAME_PATH)



_DEFAULT_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "uploads", "my_photo.png"
)

_FPS = 25

_VISEME_SEQUENCE = {
    "visemes": [
        {"start": 0.0,  "end": 0.01, "duration": 0.01, "viseme": "X"},
        {"start": 0.01, "end": 0.12, "duration": 0.11, "viseme": "B"},
        {"start": 0.12, "end": 0.19, "duration": 0.07, "viseme": "C"},
        {"start": 0.19, "end": 0.32, "duration": 0.13, "viseme": "A"},
        {"start": 0.32, "end": 0.59, "duration": 0.27, "viseme": "B"},
        {"start": 0.59, "end": 0.66, "duration": 0.07, "viseme": "G"},
        {"start": 0.66, "end": 0.8,  "duration": 0.14, "viseme": "E"},
        {"start": 0.8,  "end": 0.87, "duration": 0.07, "viseme": "F"},
        {"start": 0.87, "end": 0.94, "duration": 0.07, "viseme": "B"},
        {"start": 0.94, "end": 1.08, "duration": 0.14, "viseme": "C"},
        {"start": 1.08, "end": 1.22, "duration": 0.14, "viseme": "E"},
        {"start": 1.22, "end": 1.36, "duration": 0.14, "viseme": "B"},
        {"start": 1.36, "end": 1.44, "duration": 0.08, "viseme": "A"},
        {"start": 1.44, "end": 1.61, "duration": 0.17, "viseme": "E"},
        {"start": 1.61, "end": 1.69, "duration": 0.08, "viseme": "A"},
        {"start": 1.69, "end": 1.86, "duration": 0.17, "viseme": "C"},
        {"start": 1.86, "end": 1.93, "duration": 0.07, "viseme": "B"},
        {"start": 1.93, "end": 2.04, "duration": 0.11, "viseme": "A"},
        {"start": 2.04, "end": 2.1,  "duration": 0.06, "viseme": "B"},
        {"start": 2.1,  "end": 2.16, "duration": 0.06, "viseme": "G"},
        {"start": 2.16, "end": 2.23, "duration": 0.07, "viseme": "C"},
        {"start": 2.23, "end": 2.3,  "duration": 0.07, "viseme": "B"},
        {"start": 2.3,  "end": 2.38, "duration": 0.08, "viseme": "A"},
        {"start": 2.38, "end": 2.46, "duration": 0.08, "viseme": "E"},
        {"start": 2.46, "end": 2.53, "duration": 0.07, "viseme": "A"},
        {"start": 2.53, "end": 2.72, "duration": 0.19, "viseme": "F"},
        {"start": 2.72, "end": 2.79, "duration": 0.07, "viseme": "E"},
        {"start": 2.79, "end": 2.93, "duration": 0.14, "viseme": "C"},
        {"start": 2.93, "end": 3.0,  "duration": 0.07, "viseme": "B"},
        {"start": 3.0,  "end": 3.36, "duration": 0.36, "viseme": "X"},
        {"start": 3.36, "end": 3.62, "duration": 0.26, "viseme": "B"},
        {"start": 3.62, "end": 3.83, "duration": 0.21, "viseme": "C"},
        {"start": 3.83, "end": 3.9,  "duration": 0.07, "viseme": "B"},
        {"start": 3.9,  "end": 4.11, "duration": 0.21, "viseme": "C"},
        {"start": 4.11, "end": 4.18, "duration": 0.07, "viseme": "E"},
        {"start": 4.18, "end": 4.25, "duration": 0.07, "viseme": "C"},
        {"start": 4.25, "end": 4.39, "duration": 0.14, "viseme": "B"},
        {"start": 4.39, "end": 4.51, "duration": 0.12, "viseme": "A"},
        {"start": 4.51, "end": 4.58, "duration": 0.07, "viseme": "C"},
        {"start": 4.58, "end": 4.64, "duration": 0.06, "viseme": "E"},
        {"start": 4.64, "end": 4.78, "duration": 0.14, "viseme": "F"},
        {"start": 4.78, "end": 4.85, "duration": 0.07, "viseme": "E"},
        {"start": 4.85, "end": 4.92, "duration": 0.07, "viseme": "F"},
        {"start": 4.92, "end": 4.99, "duration": 0.07, "viseme": "H"},
        {"start": 4.99, "end": 5.09, "duration": 0.1,  "viseme": "D"},
        {"start": 5.09, "end": 5.13, "duration": 0.04, "viseme": "C"},
        {"start": 5.13, "end": 5.2,  "duration": 0.07, "viseme": "A"},
        {"start": 5.2,  "end": 5.27, "duration": 0.07, "viseme": "F"},
        {"start": 5.27, "end": 5.33, "duration": 0.06, "viseme": "A"},
        {"start": 5.33, "end": 5.41, "duration": 0.08, "viseme": "E"},
        {"start": 5.41, "end": 5.55, "duration": 0.14, "viseme": "D"},
        {"start": 5.55, "end": 5.62, "duration": 0.07, "viseme": "C"},
        {"start": 5.62, "end": 5.7,  "duration": 0.08, "viseme": "A"},
        {"start": 5.7,  "end": 6.01, "duration": 0.31, "viseme": "B"},
        {"start": 6.01, "end": 6.09, "duration": 0.08, "viseme": "A"},
        {"start": 6.09, "end": 6.45, "duration": 0.36, "viseme": "B"},
        {"start": 6.45, "end": 6.48, "duration": 0.03, "viseme": "X"},
    ]
}


def warmup_stream(pipeline) -> None:
    # 4 frames: warmup frames 1-2 do eager + graph capture; frames 3-4 absorb
    # the per-module re-recording that happens when GPU memory addresses shift
    # between sessions (cudagraph_trees check_invariants failure).
    keyframes = neutral_keyframes(seconds=4 / _FPS, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        frame_gen, _ = pipeline.execute_streaming(args)
        for _ in frame_gen:
            pass
    finally:
        os.unlink(tpl_file.name)



def liveportrait_frame_gen(viseme_sequence=None):
    """Yields raw RGB numpy frames from the LivePortrait pipeline."""
    vseq = viseme_sequence or _VISEME_SEQUENCE
    _total_frames = round(vseq["visemes"][-1]["end"] * _FPS) + 5
    _total_seconds = _total_frames / _FPS
    keyframes = neutral_keyframes(seconds=_total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, vseq["visemes"], fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()
        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        with _pipeline_lock:
            frame_gen, _ = _get_pipeline().execute_streaming(args)
            yield from frame_gen
    finally:
        os.unlink(tpl_file.name)
