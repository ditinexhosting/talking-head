from __future__ import annotations

import asyncio
import colorsys
import logging
import os
from datetime import timedelta
from time import perf_counter

import numpy as np
import httpx
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livekit.ditinex.com")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret123changeme")

WIDTH, HEIGHT = 640, 480
FPS = 25


def generate_livekit_token(identity: str, room_id: str) -> str:
    return (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(
            room_join=True,
            room=room_id,
            can_publish=True,
            can_subscribe=False,
        ))
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )


async def stream_hue(source: rtc.VideoSource) -> None:
    """Rainbow hue-cycling frames at FPS — swap with real pipeline frames later."""
    argb_frame = bytearray(WIDTH * HEIGHT * 4)
    arr = np.frombuffer(argb_frame, dtype=np.uint8)

    hue = 0.0
    next_frame_time = perf_counter()

    while True:
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        arr.flat[::4]  = int(r * 255)
        arr.flat[1::4] = int(g * 255)
        arr.flat[2::4] = int(b * 255)
        arr.flat[3::4] = 255

        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame))
        hue = (hue + (1 / FPS) / 3) % 1.0

        next_frame_time += 1 / FPS
        await asyncio.sleep(max(0.0, next_frame_time - perf_counter()))


async def notify_vps(callback_url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(callback_url)
        logger.info("[agent] notified VPS: %s", callback_url)
    except Exception as exc:
        logger.warning("[agent] VPS callback failed: %s", exc)


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    logger.info("[agent] starting room=%s", room_id)
    room = rtc.Room()
    hue_task: asyncio.Task | None = None

    @room.on("disconnected")
    def on_disconnect():
        if hue_task and not hue_task.done():
            hue_task.cancel()
        if callback_url:
            asyncio.create_task(notify_vps(callback_url))

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

        hue_task = asyncio.create_task(stream_hue(source))
        await hue_task  # runs until cancelled by on_disconnect

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        if hue_task and not hue_task.done():
            hue_task.cancel()
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
