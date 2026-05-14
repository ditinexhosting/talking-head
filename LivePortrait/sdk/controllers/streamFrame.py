from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from sdk.controllers.speech2video.video import run_liveportrait_stream

if TYPE_CHECKING:
    from sdk.controllers.ws import WebSocketConnection

_END = object()


def _next_frame(gen):
    try:
        return next(gen)
    except StopIteration:
        return _END


async def stream_frame(conn: WebSocketConnection) -> None:
    gen = run_liveportrait_stream()
    chunk_count = 0
    bytes_sent = 0
    t_start = time.perf_counter()

    while True:
        t_gen_start = time.perf_counter()
        chunk_bytes = await asyncio.to_thread(_next_frame, gen)
        gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

        if chunk_bytes is _END:
            break

        t_send_start = time.perf_counter()
        await conn.send_bytes(chunk_bytes)
        send_ms = (time.perf_counter() - t_send_start) * 1000.0
        # Force an event-loop yield so the ASGI send drains immediately and
        # nothing further upstream can coalesce writes into the same tick.
        await asyncio.sleep(0)

        print(
            f"[ws] chunk={chunk_count:04d} gen={gen_ms:7.2f}ms "
            f"send={send_ms:7.2f}ms size={len(chunk_bytes)}B",
            flush=True,
        )
        chunk_count += 1
        bytes_sent += len(chunk_bytes)

    elapsed = time.perf_counter() - t_start
    # ~8 video frames per fMP4 fragment (server uses -g 8). This is an
    # approximation — for the authoritative frame count see the [stream] DONE
    # line printed by run_liveportrait_stream.
    approx_frames = chunk_count * 8
    fps = approx_frames / elapsed if elapsed > 0 else 0.0
    print(
        f"[ws] sent {chunk_count} fMP4 chunks ({bytes_sent}B) in "
        f"{elapsed * 1000.0:.2f}ms ({elapsed:.2f}s) → ~{fps:.1f} fps",
        flush=True,
    )
