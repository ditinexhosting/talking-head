from __future__ import annotations

import asyncio
import colorsys
import logging
import os
import threading
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


async def stream_rainbow(source: rtc.VideoSource) -> None:
    argb_frame = bytearray(WIDTH * HEIGHT * 4)
    arr = np.frombuffer(argb_frame, dtype=np.uint8)
    hue = 0.0
    next_t = perf_counter()
    while True:
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        arr.flat[::4]  = int(r * 255)
        arr.flat[1::4] = int(g * 255)
        arr.flat[2::4] = int(b * 255)
        arr.flat[3::4] = 255
        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame))
        hue = (hue + (1 / FPS) / 3) % 1.0
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


async def stream_liveportrait(source: rtc.VideoSource) -> None:
    """Stream LivePortrait frames to a LiveKit VideoSource at FPS rate."""
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
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = frame_rgb
            rgba[:, :, 3] = 255
            source.capture_frame(rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, rgba.tobytes()))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
    finally:
        thread.join(timeout=10)


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
    loop = asyncio.get_running_loop()
    chat_queue: asyncio.Queue[str] = asyncio.Queue()
    disconnected = asyncio.Event()

    @room.on("data_received")
    def on_data(packet: rtc.DataPacket) -> None:
        if packet.topic == "chat":
            try:
                loop.call_soon_threadsafe(chat_queue.put_nowait, packet.data.decode("utf-8"))
            except Exception:
                pass

    @room.on("disconnected")
    def on_disconnect() -> None:
        loop.call_soon_threadsafe(disconnected.set)
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

        stop_task = asyncio.create_task(disconnected.wait())
        stream_task = asyncio.create_task(stream_rainbow(source))

        while not disconnected.is_set():
            chat_task = asyncio.create_task(chat_queue.get())
            done, _ = await asyncio.wait(
                {stream_task, chat_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stop_task in done:
                stream_task.cancel()
                chat_task.cancel()
                break

            if chat_task in done:
                text = chat_task.result()
                logger.info("[agent] chat: %r", text)
                # Cancel rainbow
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass
                # Run LivePortrait, interruptible by disconnect
                lp_task = asyncio.create_task(stream_liveportrait(source))
                done2, _ = await asyncio.wait({lp_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
                if stop_task in done2:
                    lp_task.cancel()
                    try:
                        await lp_task
                    except asyncio.CancelledError:
                        pass
                    break
                # Drain any messages that arrived during liveportrait
                while not chat_queue.empty():
                    chat_queue.get_nowait()
                # Back to rainbow
                stream_task = asyncio.create_task(stream_rainbow(source))
            else:
                # stream_task finished unexpectedly — restart rainbow
                chat_task.cancel()
                try:
                    await chat_task
                except asyncio.CancelledError:
                    pass
                stream_task = asyncio.create_task(stream_rainbow(source))

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
