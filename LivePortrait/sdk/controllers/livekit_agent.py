from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import timedelta

import numpy as np
import httpx
from livekit import rtc, api

logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livekit.ditinex.com")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret123changeme")

FRAME_W = 640
FRAME_H = 480
FRAME_FPS = 25

# Cycling background colours for the placeholder stream (RGB)
_COLOURS = [
    (220, 60,  60),   # red
    (60,  180, 60),   # green
    (60,  100, 220),  # blue
    (200, 160, 30),   # amber
]


def generate_livekit_token(identity: str, room_id: str) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=room_id,
        can_publish=True,
        can_subscribe=False,
    )
    token = (
        api.AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )
    return token


class SessionPipeline:
    """Placeholder pipeline that streams cycling solid-colour frames."""

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.interrupted = asyncio.Event()
        self._source: rtc.VideoSource | None = None
        self._done = asyncio.Event()

    def set_source(self, source: rtc.VideoSource) -> None:
        self._source = source

    async def handle_text(self, text: str) -> None:
        logger.info("[pipeline] room=%s text received: %s", self.room_id, text)
        # TODO: drive TTS → lip-sync → LivePortrait frames from here

    async def run_until_done(self) -> None:
        colour_idx = 0
        frame_interval = 1.0 / FRAME_FPS

        while not self.interrupted.is_set():
            if self._source is not None:
                r, g, b = _COLOURS[colour_idx % len(_COLOURS)]
                # RGBA flat array — LiveKit VideoFrame expects RGBA
                rgba = np.full((FRAME_H, FRAME_W, 4), (r, g, b, 255), dtype=np.uint8)
                frame = rtc.VideoFrame(
                    width=FRAME_W,
                    height=FRAME_H,
                    type=rtc.VideoBufferType.RGBA,
                    data=rgba.tobytes(),
                )
                self._source.capture_frame(frame)

            await asyncio.sleep(frame_interval)

            # Advance colour every 60 frames (~2 s)
            colour_idx += 1
            if colour_idx % 60 == 0:
                colour_idx = (colour_idx // 60 + 1) * 60

        self._done.set()


async def setup_tracks(room: rtc.Room, pipeline: SessionPipeline) -> None:
    source = rtc.VideoSource(width=FRAME_W, height=FRAME_H)
    pipeline.set_source(source)

    track = rtc.LocalVideoTrack.create_video_track("agent-video", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(track, options)
    logger.info("[agent] video track published in room=%s", pipeline.room_id)


async def notify_vps(callback_url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(callback_url)
        logger.info("[agent] notified VPS at %s", callback_url)
    except Exception as exc:
        logger.warning("[agent] VPS callback failed: %s", exc)


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    logger.info("[agent] starting for room=%s", room_id)
    room = rtc.Room()
    pipeline = SessionPipeline(room_id)

    @room.on("data_received")
    def on_data(packet: rtc.DataPacket):
        try:
            import json
            msg = json.loads(packet.data)
            if msg.get("type") == "user_text":
                asyncio.create_task(pipeline.handle_text(msg["text"]))
            elif msg.get("type") == "interrupt":
                pipeline.interrupted.set()
        except Exception as exc:
            logger.warning("[agent] bad data packet: %s", exc)

    @room.on("disconnected")
    def on_disconnect():
        pipeline.interrupted.set()
        if callback_url:
            asyncio.create_task(notify_vps(callback_url))

    try:
        token = generate_livekit_token(f"agent-{room_id}", room_id)
        await room.connect(LIVEKIT_URL, token)
        logger.info("[agent] connected to room=%s", room_id)

        await setup_tracks(room, pipeline)
        await pipeline.run_until_done()

    except Exception as exc:
        logger.error("[agent] room=%s crashed: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] disconnected from room=%s", room_id)
