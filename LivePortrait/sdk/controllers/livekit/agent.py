from __future__ import annotations

import asyncio
import logging
import os
import threading
import time as _time
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

# How long to delay audio relative to the first generated video frame.
# WebRTC video has ~150-300 ms more pipeline latency than audio (H.264 encoding +
# jitter buffer), so audio must start later to stay in sync at the receiver.
# Tune this value if audio arrives noticeably before or after lip movement.
AUDIO_VIDEO_OFFSET_SECONDS = float(os.getenv("AUDIO_VIDEO_OFFSET_SECONDS", "0.3"))

_AUDIO_CHUNK_MS = 20   # ms per audio chunk — matches LiveKit's expected cadence
_PREFILL_FRAMES = 10   # video frames to buffer before starting playback


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
    return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, rgba.tobytes())


def _start_audio_thread(
    audio_source: rtc.AudioSource,
    pcm_bytes: bytes,
    sample_rate: int,
    main_loop: asyncio.AbstractEventLoop,
    start_delay: float = 0.0,
) -> threading.Thread:
    """Stream audio on a dedicated OS thread, completely decoupled from the asyncio event loop.

    Why a thread instead of asyncio.create_task:
    - asyncio is single-threaded: even a 'concurrent' Task still occupies the event loop
      thread while capture_frame awaits its 20 ms window, stealing time from the video loop.
    - This thread blocks on fut.result() at the OS level.  During that OS-level wait,
      the event loop is 100 % free to push video frames at precise 25 fps intervals.
    - start_delay uses time.sleep (OS-level) so the event loop is never involved.
    """
    chunk_size = sample_rate * _AUDIO_CHUNK_MS // 1000 * 2  # int16 bytes per chunk

    def _run() -> None:
        if start_delay > 0:
            _time.sleep(start_delay)
        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset : offset + chunk_size]
            if not chunk:
                break
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    audio_source.capture_frame(frame), main_loop
                )
                fut.result()  # OS-level block; event loop stays free
            except Exception as exc:
                logger.warning("[audio-thread] stopping early: %s", exc)
                break
            offset += chunk_size
        logger.debug("[audio-thread] done")

    t = threading.Thread(target=_run, daemon=True, name="audio-stream")
    t.start()
    return t


async def _play_sentence(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource | None,
    pcm_bytes: bytes,
    sample_rate: int,
    visemes: list,
    idle: bytes,
    next_t: float,
    loop: asyncio.AbstractEventLoop,
) -> float:
    """Video drives the clock at a steady 25 fps; audio streams on its own OS thread."""
    viseme_sequence = {"visemes": visemes}

    # Unlimited queue — let LP generate at full GPU speed without being throttled
    frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=0)

    def _produce(q: asyncio.Queue) -> None:
        try:
            for frame_rgb in liveportrait_frame_gen(viseme_sequence):
                asyncio.run_coroutine_threadsafe(q.put(frame_rgb), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

    threading.Thread(target=_produce, args=(frame_queue,), daemon=True).start()

    # Phase 1 — Prefill: accumulate frames while showing idle video
    prefill: list[np.ndarray] = []
    generator_done = False
    while len(prefill) < _PREFILL_FRAMES and not generator_done:
        try:
            val = frame_queue.get_nowait()
            if val is None:
                generator_done = True
            else:
                prefill.append(val)
        except asyncio.QueueEmpty:
            video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))

    logger.info(
        "[agent] prefill done frames=%d generator_done=%s audio_delay=%.2fs",
        len(prefill),
        generator_done,
        AUDIO_VIDEO_OFFSET_SECONDS,
    )

    # Launch audio on a dedicated OS thread with the configured delay.
    # The delay compensates for the extra pipeline latency video has vs audio in WebRTC.
    audio_thread: threading.Thread | None = None
    if audio_source is not None and pcm_bytes:
        audio_thread = _start_audio_thread(
            audio_source, pcm_bytes, sample_rate, loop,
            start_delay=AUDIO_VIDEO_OFFSET_SECONDS,
        )

    # Phase 2 — Play buffered frames; video clock is authoritative
    for frame_rgb in prefill:
        video_source.capture_frame(_to_rtc_frame(frame_rgb))
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))

    # Phase 3 — Stream remaining frames from the generator
    if not generator_done:
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

    # Phase 4 — Keep video alive with idle frames while audio thread drains
    if audio_thread is not None:
        while audio_thread.is_alive():
            video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))

    return next_t


async def _stream_loop(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource | None,
    text_queue: asyncio.Queue[str],
) -> None:
    """Continuously push idle frames; background TTS task pre-fills play_queue so
    there are no gaps between sentences or during TTS computation."""
    loop = asyncio.get_running_loop()
    _idle = idle_frame if idle_frame is not None else _RED_FRAME
    next_t = perf_counter()
    play_queue: asyncio.Queue[tuple[bytes, int, list] | None] = asyncio.Queue()

    while True:
        # Play next sentence if TTS already has one ready
        try:
            item = play_queue.get_nowait()
            if item is not None:
                pcm_bytes, sample_rate, visemes, sentence = item
                logger.info(
                    "[agent] playing sentence — audio=%.2fs visemes=%d sentence=%r",
                    len(pcm_bytes) / 2 / sample_rate,
                    len(visemes),
                    sentence,
                )
                next_t = await _play_sentence(
                    video_source, audio_source, pcm_bytes, sample_rate,
                    visemes, _idle, next_t, loop,
                )
                logger.info("[agent] sentence done")
            # item is None → TTS batch sentinel, just consume and loop
            continue
        except asyncio.QueueEmpty:
            pass

        # Kick off background TTS for any new text
        try:
            text = text_queue.get_nowait()
            logger.info("[agent] received text: %r", text)

            async def _run_tts(t: str = text) -> None:
                async for res in text_to_speech_kokoro(t):
                    await play_queue.put(res)
                await play_queue.put(None)
                logger.info("[agent] tts batch done")

            asyncio.create_task(_run_tts())
            continue
        except asyncio.QueueEmpty:
            pass

        # Nothing ready — push idle frame to keep the track alive
        video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle))
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


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
