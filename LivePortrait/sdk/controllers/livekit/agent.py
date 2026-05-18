from __future__ import annotations

import asyncio
import colorsys
import logging
import os
import threading
from datetime import timedelta
from time import perf_counter

import cv2
import numpy as np
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from sdk.controllers.livekit.data_handlers import TOPIC_HANDLERS
from sdk.controllers.livekit.util import idle_frame

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


async def stream_liveportrait(source: rtc.VideoSource) -> None:
    from sdk.controllers.speech2video.video import liveportrait_frame_gen

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _produce():
        try:
            for frame in liveportrait_frame_gen():
                loop.call_soon_threadsafe(queue.put_nowait, frame)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=_produce, daemon=True)
    thread.start()

    try:
        next_t = perf_counter()
        while True:
            frame_rgb = await queue.get()
            if frame_rgb is None:
                break
            h, w = frame_rgb.shape[:2]
            if h != HEIGHT or w != WIDTH:
                frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
            rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame_rgb
            rgba[:, :, 3] = 255
            source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, bytes(rgba.tobytes())))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
    finally:
        thread.join(timeout=10)


async def _stream_idle(source: rtc.VideoSource) -> None:
    next_t = perf_counter()
    while True:
        frame = idle_frame if idle_frame is not None else _RED_FRAME
        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, frame))
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    logger.info("[agent] starting room=%s", room_id)
    room = rtc.Room()
    loop = asyncio.get_running_loop()
    disconnected = asyncio.Event()

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
        handler = TOPIC_HANDLERS.get(data.topic)
        if handler:
            handler(data.participant.identity, text)
        else:
            logger.debug("[agent] unknown topic=%r from %s", data.topic, data.participant.identity)

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

        logger.info("[agent] streaming blank frames room=%s", room_id)
        await _run_until_disconnected(_stream_idle(source), disconnected)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
