from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from sdk.controllers.speech2video.video import run_liveportrait_stream

if TYPE_CHECKING:
    from sdk.controllers.ws import WebSocketConnection

_executor = ThreadPoolExecutor(max_workers=1)


def _run_pipeline(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    for frame_bytes in run_liveportrait_stream():
        asyncio.run_coroutine_threadsafe(queue.put(frame_bytes), loop)
    asyncio.run_coroutine_threadsafe(queue.put(None), loop)  # signal end of stream


async def stream_frame(conn: WebSocketConnection) -> None:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)

    loop.run_in_executor(_executor, _run_pipeline, queue, loop)

    frame_count = 0
    t_start = time.perf_counter()
    while True:
        frame_bytes = await queue.get()
        if frame_bytes is None:
            break
        await conn.send_bytes(frame_bytes)
        frame_count += 1

    elapsed = time.perf_counter() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"[stream] {frame_count} frames in {elapsed:.2f}s → {fps:.1f} fps", flush=True)
