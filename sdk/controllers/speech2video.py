from __future__ import annotations

import asyncio
import os
import os.path as osp
import pickle
import shutil
import sys
import tempfile
import threading
from pathlib import Path

_LIVE_PORTRAIT = Path(__file__).parents[2] / "LivePortrait"

from controllers.liveportrait.utils.motion import add_blinks, add_talks, build_template, neutral_keyframes

_FPS = 25
_UPLOADS = osp.abspath(osp.join(osp.dirname(__file__), "liveportrait", "uploads"))
_DEFAULT_SOURCE = osp.join(_UPLOADS, "my_photo.png")
_PIPELINE_LOCK = threading.Lock()


def _ensure_lp_on_path():
    if str(_LIVE_PORTRAIT) not in sys.path:
        sys.path.insert(0, str(_LIVE_PORTRAIT))


def _load_pipeline():
    _ensure_lp_on_path()
    from api.pipeline import get_pipeline
    return get_pipeline()


def _generate_video(visemes: list[dict], source: str) -> bytes:
    _ensure_lp_on_path()
    from src.config.argument_config import ArgumentConfig

    total_frames = round(visemes[-1]["end"] * _FPS) + 7
    total_seconds = total_frames / _FPS

    keyframes = neutral_keyframes(seconds=total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, visemes, fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], _FPS)

    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_pkl = osp.join(tmp_dir, "motion.pkl")
        output_dir = osp.join(tmp_dir, "outputs")
        os.makedirs(output_dir)

        with open(tmp_pkl, "wb") as f:
            pickle.dump(template, f)

        args = ArgumentConfig(source=source, driving=tmp_pkl, output_dir=output_dir)

        pipeline = _load_pipeline()
        with _PIPELINE_LOCK:
            wfp, _ = pipeline.execute(args)

        with open(wfp, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def speech_to_video(
    visemes: list[dict],
    source: str = _DEFAULT_SOURCE,
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generate_video, visemes, source)
