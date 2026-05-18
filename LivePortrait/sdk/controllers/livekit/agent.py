from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import timedelta
from time import perf_counter

import numpy as np
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from sdk.controllers.livekit.util import idle_frame
from sdk.controllers.speech2video.video import liveportrait_frame_gen

logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livekit.ditinex.com")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret123changeme")

WIDTH, HEIGHT = 512, 512
FPS = 25

_RED_FRAME = bytes(np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), WIDTH * HEIGHT).tobytes())

def generate_livekit_token(identity: str, room_id: str) -> str:
    return (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(
            room_join=True,
            room=room_id,
            can_publish=True,
            can_subscribe=True,
        ))
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )


async def _run_until_disconnected(coro, disconnected: asyncio.Event) -> None:
    task = asyncio.create_task(coro)
    await disconnected.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _to_rtc_frame(frame_rgb: np.ndarray) -> rtc.VideoFrame:
    rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame_rgb
    rgba[:, :, 3] = 255
    return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, bytes(rgba.tobytes()))


async def _stream_loop(source: rtc.VideoSource, text_queue: asyncio.Queue[str]) -> None:
    """Show idle frame; on each queued text trigger a LivePortrait animation, then return to idle."""
    loop = asyncio.get_running_loop()
    _idle = idle_frame if idle_frame is not None else _RED_FRAME
    next_t = perf_counter()

    while True:
        # Check for a pending text without blocking
        try:
            text = text_queue.get_nowait()
        except asyncio.QueueEmpty:
            source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
            continue

        logger.info("[agent] animating for text: %r", text)

        frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=4)

        def _produce(q: asyncio.Queue) -> None:
            try:
                for frame_rgb in liveportrait_frame_gen():
                    asyncio.run_coroutine_threadsafe(q.put(frame_rgb), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

        threading.Thread(target=_produce, args=(frame_queue,), daemon=True).start()

        while True:
            try:
                frame_rgb = frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle))
                next_t += 1.0 / FPS
                await asyncio.sleep(max(0.0, next_t - perf_counter()))
                continue
            if frame_rgb is None:
                break
            source.capture_frame(_to_rtc_frame(frame_rgb))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))

        logger.info("[agent] animation done, returning to idle")


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    logger.info("[agent] starting room=%s", room_id)
    room = rtc.Room()
    loop = asyncio.get_running_loop()
    disconnected = asyncio.Event()
    text_queue: asyncio.Queue[str] = asyncio.Queue()

    @room.on("disconnected")
    def on_disconnected():
        loop.call_soon_threadsafe(disconnected.set)

    @room.on("reconnecting")
    def on_reconnecting():
        loop.call_soon_threadsafe(disconnected.set)

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            text = data.data.decode("utf-8")
        except Exception:
            return
        logger.debug("[agent] data topic=%r from %s: %r", data.topic, data.participant.identity, text)
        if data.topic == "chat":
            loop.call_soon_threadsafe(text_queue.put_nowait, text)

    try:
        token = generate_livekit_token(f"agent-{room_id}", room_id)
        await room.connect(LIVEKIT_URL, token)
        logger.info("[agent] connected room=%s", room_id)

        source = rtc.VideoSource(WIDTH, HEIGHT)
        track = rtc.LocalVideoTrack.create_video_track("agent-video", source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=FPS,
                max_bitrate=2_000_000,
            ),
        )
        pub = await room.local_participant.publish_track(track, options)
        logger.info("[agent] track published sid=%s room=%s", pub.sid, room_id)

        logger.info("[agent] starting stream loop room=%s", room_id)
        await _run_until_disconnected(_stream_loop(source, text_queue), disconnected)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
