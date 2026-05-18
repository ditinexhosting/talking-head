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
from sdk.controllers.tts import text_to_speech_kokoro

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


_AUDIO_CHUNK_MS = 20  # push audio in 20ms chunks so capture_frame paces at real playback rate


async def _push_audio(audio_source: rtc.AudioSource, pcm_bytes: bytes, sample_rate: int) -> None:
    """Push PCM audio in 20ms chunks so this coroutine takes the actual audio duration to complete."""
    bytes_per_chunk = sample_rate * _AUDIO_CHUNK_MS // 1000 * 2  # int16 = 2 bytes/sample
    offset = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset:offset + bytes_per_chunk]
        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(chunk) // 2,
        )
        await audio_source.capture_frame(frame)
        offset += bytes_per_chunk


async def _play_sentence(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource,
    pcm_bytes: bytes,
    sample_rate: int,
    idle: bytes,
    next_t: float,
    loop: asyncio.AbstractEventLoop,
) -> float:
    """Play one sentence: audio + video run concurrently; returns updated next_t.
    Both must fully complete before this returns — sentence 2 cannot start until then."""
    frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=4)

    def _produce(q: asyncio.Queue) -> None:
        try:
            for frame_rgb in liveportrait_frame_gen():
                asyncio.run_coroutine_threadsafe(q.put(frame_rgb), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

    threading.Thread(target=_produce, args=(frame_queue,), daemon=True).start()

    # Audio runs as a timed task — takes the real playback duration to complete
    audio_task = asyncio.create_task(_push_audio(audio_source, pcm_bytes, sample_rate))

    # Drive video frames until the animation generator is exhausted
    while True:
        try:
            frame_rgb = frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
            continue
        if frame_rgb is None:
            break
        video_source.capture_frame(_to_rtc_frame(frame_rgb))
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))

    # Wait for audio playback to finish — whichever finishes last, the other already done
    await audio_task
    return next_t


async def _stream_loop(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource,
    text_queue: asyncio.Queue[str],
) -> None:
    """Idle until text arrives, then play each sentence fully (audio + video) before starting the next."""
    loop = asyncio.get_running_loop()
    _idle = idle_frame if idle_frame is not None else _RED_FRAME
    next_t = perf_counter()

    while True:
        try:
            text = text_queue.get_nowait()
        except asyncio.QueueEmpty:
            video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
            continue

        logger.info("[agent] received text: %r", text)

        async for pcm_bytes, sample_rate, visemes in text_to_speech_kokoro(text):
            logger.info("[agent] playing sentence — audio=%.2fs", len(pcm_bytes) / 2 / sample_rate)
            next_t = await _play_sentence(video_source, audio_source, pcm_bytes, sample_rate, _idle, next_t, loop)
            logger.info("[agent] sentence done, moving to next")

        logger.info("[agent] all sentences done, returning to idle")


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

        video_source = rtc.VideoSource(WIDTH, HEIGHT)
        video_track = rtc.LocalVideoTrack.create_video_track("agent-video", video_source)
        video_pub = await room.local_participant.publish_track(
            video_track,
            rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_encoding=rtc.VideoEncoding(max_framerate=FPS, max_bitrate=2_000_000),
            ),
        )
        logger.info("[agent] video track published sid=%s room=%s", video_pub.sid, room_id)

        audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
        audio_pub = await room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        logger.info("[agent] audio track published sid=%s room=%s", audio_pub.sid, room_id)

        logger.info("[agent] starting stream loop room=%s", room_id)
        await _run_until_disconnected(_stream_loop(video_source, audio_source, text_queue), disconnected)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
