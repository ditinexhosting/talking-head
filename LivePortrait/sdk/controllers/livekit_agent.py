from __future__ import annotations

import asyncio
import colorsys
import json
import logging
import os
import threading
from datetime import timedelta
from time import perf_counter

import cv2
import numpy as np
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livekit.ditinex.com")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret123changeme")

WIDTH, HEIGHT = 512, 512
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
            can_subscribe=True,
        ))
        .with_ttl(timedelta(hours=2))
        .to_jwt()
    )


def _handle_chat(participant: str, text: str) -> None:
    logger.info("[agent] chat from %s: %s", participant, text)


def _handle_command(participant: str, text: str) -> None:
    try:
        payload = json.loads(text)
        logger.info("[agent] command from %s: %s", participant, payload)
    except json.JSONDecodeError:
        logger.warning("[agent] invalid command JSON from %s", participant)


_TOPIC_HANDLERS = {
    "chat": _handle_chat,
    "command": _handle_command,
}


async def _wait_first(*events: asyncio.Event) -> None:
    tasks = [asyncio.create_task(e.wait()) for e in events]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()


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


_BLANK_FRAME = bytes(WIDTH * HEIGHT * 4)


async def _stream_hold(source: rtc.VideoSource) -> None:
    next_t = perf_counter()
    while True:
        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _BLANK_FRAME))
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


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
        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, bytes(argb_frame)))
        hue = (hue + (1 / FPS) / 3) % 1.0
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    logger.info("[agent] starting room=%s", room_id)
    room = rtc.Room()
    loop = asyncio.get_running_loop()
    disconnected = asyncio.Event()
    chat_event = asyncio.Event()

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
        if data.topic == "chat":
            loop.call_soon_threadsafe(chat_event.set)
        handler = _TOPIC_HANDLERS.get(data.topic)
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

        source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, bytes(WIDTH * HEIGHT * 4)))

        while not disconnected.is_set():
            logger.info("[agent] waiting for chat room=%s", room_id)
            hold_task = asyncio.create_task(_stream_hold(source))
            await _wait_first(chat_event, disconnected)
            hold_task.cancel()
            try:
                await hold_task
            except asyncio.CancelledError:
                pass

            if disconnected.is_set():
                break

            chat_event.clear()
            logger.info("[agent] chat received, starting liveportrait room=%s", room_id)

            stream_task = asyncio.create_task(stream_liveportrait(source))
            disconnect_task = asyncio.create_task(disconnected.wait())
            _, pending = await asyncio.wait(
                [stream_task, disconnect_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
