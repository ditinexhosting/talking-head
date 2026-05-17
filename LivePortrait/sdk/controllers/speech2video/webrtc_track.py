from __future__ import annotations

import asyncio
import fractions
import os
import pickle
import tempfile
import time

from aiortc import VideoStreamTrack
from av import VideoFrame

from src.config.argument_config import ArgumentConfig
from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import (
    neutral_keyframes,
    build_template,
    add_blinks,
    add_talks,
)

_FPS = 25
_VIDEO_CLOCK_RATE = 90000  # RTP video clock — fixed by spec
_VIDEO_TIME_BASE = fractions.Fraction(1, _VIDEO_CLOCK_RATE)
_TICKS_PER_FRAME = _VIDEO_CLOCK_RATE // _FPS


class LivePortraitTrack(VideoStreamTrack):
    """Wraps frame_gen from LivePortrait into a WebRTC video track.

    aiortc calls recv() on the encoder thread at the pace of the negotiated
    framerate. We block until frame_gen produces the next numpy frame, wrap
    it as an av.VideoFrame with rgb24 layout, then stamp a monotonically
    increasing pts on the 90 kHz RTP clock.
    """

    kind = "video"

    def __init__(self, visemes: list[dict], source_path: str):
        super().__init__()
        self._visemes = visemes
        self._source_path = source_path
        self._frame_gen = None
        self._tpl_path = None
        self._pts = 0
        self._started_at = None

    def _build_generator(self):
        total_frames = round(self._visemes[-1]["end"] * _FPS) + 15
        total_seconds = total_frames / _FPS
        keyframes = neutral_keyframes(seconds=total_seconds, fps=_FPS)
        keyframes = add_blinks(keyframes, fps=_FPS)
        keyframes = add_talks(keyframes, self._visemes, fps=_FPS)
        template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

        tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(template, tpl_file)
        tpl_file.close()
        self._tpl_path = tpl_file.name

        args = ArgumentConfig(
            source=self._source_path,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
            flag_pasteback=False,
        )
        frame_gen, _ = _get_pipeline().execute_streaming(args)
        return frame_gen

    async def recv(self) -> VideoFrame:
        if self._frame_gen is None:
            print("[track] recv() first call — building frame_gen…", flush=True)
            t0 = time.perf_counter()
            self._frame_gen = await asyncio.to_thread(self._build_generator)
            self._started_at = time.perf_counter()
            print(
                f"[track] frame_gen ready in {(self._started_at - t0) * 1000:.1f}ms",
                flush=True,
            )

        t_frame_start = time.perf_counter()
        frame_np = await asyncio.to_thread(self._next_or_none)
        gen_ms = (time.perf_counter() - t_frame_start) * 1000.0

        if frame_np is None:
            print("[track] frame_gen exhausted", flush=True)
            self.stop()
            raise ConnectionError("frame_gen exhausted")

        video_frame = VideoFrame.from_ndarray(frame_np, format="rgb24")
        video_frame.pts = self._pts
        video_frame.time_base = _VIDEO_TIME_BASE
        self._pts += _TICKS_PER_FRAME

        frame_idx = self._pts // _TICKS_PER_FRAME
        if frame_idx <= 5 or frame_idx % 25 == 0:
            print(
                f"[track] frame={frame_idx} shape={frame_np.shape} "
                f"gen={gen_ms:.1f}ms",
                flush=True,
            )
        return video_frame

    def _next_or_none(self):
        try:
            return next(self._frame_gen)
        except StopIteration:
            return None

    def stop(self):
        super().stop()
        if self._tpl_path and os.path.exists(self._tpl_path):
            try:
                os.unlink(self._tpl_path)
            except OSError:
                pass
            self._tpl_path = None
