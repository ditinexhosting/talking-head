from __future__ import annotations

import asyncio
import io
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from PIL import Image

from sdk.controllers.liveportrait import _get_pipeline
from sdk.controllers.speech2video.motion import neutral_keyframes, build_template, add_blinks, add_talks
from sdk.controllers.speech2video.video import _DEFAULT_SOURCE, _FPS, _VISEME_SEQUENCE
from src.config.argument_config import ArgumentConfig

if TYPE_CHECKING:
    from sdk.controllers.ws import WebSocketConnection

_executor = ThreadPoolExecutor(max_workers=1)


def _run_pipeline(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    total_frames = round(_VISEME_SEQUENCE["visemes"][-1]["end"] * _FPS) + 15
    total_seconds = total_frames / _FPS
    keyframes = neutral_keyframes(seconds=total_seconds, fps=_FPS)
    keyframes = add_blinks(keyframes, fps=_FPS)
    keyframes = add_talks(keyframes, _VISEME_SEQUENCE["visemes"], fps=_FPS)
    template = build_template([kf.to_dict() for kf in keyframes], fps=_FPS)

    tpl_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    try:
        pickle.dump(template, tpl_file)
        tpl_file.close()

        args = ArgumentConfig(
            source=_DEFAULT_SOURCE,
            driving=tpl_file.name,
            output_dir=tempfile.mkdtemp(),
        )
        frame_gen, _ = _get_pipeline().execute_streaming(args)
        for frame in frame_gen:
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="WEBP", quality=80, method=4)
            asyncio.run_coroutine_threadsafe(queue.put(buf.getvalue()), loop)
    finally:
        os.unlink(tpl_file.name)

    asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # signal end of stream


async def stream_frame(conn: WebSocketConnection) -> None:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)

    loop.run_in_executor(_executor, _run_pipeline, queue, loop)

    while True:
        frame_bytes = await queue.get()
        if frame_bytes is None:
            break
        await conn.send_bytes(frame_bytes)
