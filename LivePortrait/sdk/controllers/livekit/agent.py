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
from sdk.controllers.tts import text_to_speech_kokoro, _split_into_sentences

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
    return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, rgba.tobytes())


_AUDIO_CHUNK_MS = 20  # ms per chunk — matches LiveKit's expected cadence
_MIN_VIDEO_BUFFER = 10  # frames to pre-buffer before starting audio+video


async def _play_sentence(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource | None,
    pcm_bytes: bytes,
    sample_rate: int,
    visemes: list,
    next_t: float,
    loop: asyncio.AbstractEventLoop,
    idle_bytes: bytes | None = None,
) -> float:
    """Video-driven playback with upfront frame buffering.

    Phase 1 — Buffer : Show idle frames until frame_queue has ≥ _MIN_VIDEO_BUFFER
              frames (or frame generation finishes, whichever comes first).
    Phase 2 — Play   : Start audio thread and pump video frames at 25 fps in sync.
    Phase 3 — Drain  : After all frames are displayed, wait for audio to finish.

    Returns the updated next_t so the video clock stays continuous.
    """
    _idle = idle_bytes if idle_bytes is not None else _RED_FRAME
    idle = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle)
    chunk_size = sample_rate * _AUDIO_CHUNK_MS // 1000 * 2  # int16 bytes per 20 ms chunk

    # Wrap the viseme list in the dict format expected by liveportrait_frame_gen.
    # An empty/missing visemes list falls back to the built-in _VISEME_SEQUENCE.
    viseme_sequence = {"visemes": visemes} if visemes else None

    # ------------------------------------------------------------------
    # Frame generation — runs in a background thread so the GPU work
    # doesn't block the event loop.  A sentinel None marks the end.
    # ------------------------------------------------------------------
    frame_queue: asyncio.Queue[rtc.VideoFrame | None] = asyncio.Queue()
    gen_done = threading.Event()

    def _generate_frames() -> None:
        try:
            for frame_rgb in liveportrait_frame_gen(viseme_sequence):
                loop.call_soon_threadsafe(frame_queue.put_nowait, _to_rtc_frame(frame_rgb))
        except Exception as exc:
            logger.warning("[video] frame gen error: %s", exc)
        finally:
            loop.call_soon_threadsafe(frame_queue.put_nowait, None)
            gen_done.set()
            logger.debug("[video] frame gen done")

    threading.Thread(target=_generate_frames, daemon=True, name="frame-gen").start()

    # ------------------------------------------------------------------
    # Phase 1 — Buffer: show idle frames until we have enough video frames
    # or the generator finishes (handles clips shorter than MIN_BUFFER).
    # ------------------------------------------------------------------
    while frame_queue.qsize() < _MIN_VIDEO_BUFFER and not gen_done.is_set():
        video_source.capture_frame(idle)
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))
    logger.debug("[video] buffer ready: %d frames queued", frame_queue.qsize())

    # ------------------------------------------------------------------
    # Phase 2 — Start audio at the same moment we begin pumping video.
    # ------------------------------------------------------------------
    audio_done = threading.Event()

    def _stream_audio() -> None:
        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset: offset + chunk_size]
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
                    audio_source.capture_frame(frame), loop
                )
                fut.result()  # blocks OS thread; event loop stays free
            except Exception as exc:
                logger.warning("[audio] stopping early: %s", exc)
                break
            offset += chunk_size
        audio_done.set()
        logger.debug("[audio] done")

    if audio_source is not None and pcm_bytes:
        threading.Thread(target=_stream_audio, daemon=True, name="audio-stream").start()
    else:
        audio_done.set()

    while True:
        frame = await frame_queue.get()
        if frame is None:  # sentinel — all frames consumed
            break
        video_source.capture_frame(frame)
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))

    # ------------------------------------------------------------------
    # Phase 3 — Drain: video finished; keep pumping idle frames at 25 fps
    # while we wait for audio to finish so the video source stays active.
    # ------------------------------------------------------------------
    while not audio_done.is_set():
        video_source.capture_frame(idle)
        next_t += 1.0 / FPS
        await asyncio.sleep(max(0.0, next_t - perf_counter()))

    return next_t


async def _stream_loop(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource | None,
    text_queue: asyncio.Queue[str],
) -> None:
    """Two independent loops running concurrently:
    - Text watcher: awaits new text, splits into sentences, fills sentence_queue.
    - Video loop:   shows idle frame when sentence_queue has items, idle frame otherwise.
    """
    _idle = idle_frame if idle_frame is not None else _RED_FRAME
    next_t = perf_counter()
    # Each item: (pcm_bytes, sample_rate, visemes, sentence)
    sentence_queue: asyncio.Queue[tuple[bytes, int, list, str]] = asyncio.Queue()

    async def _text_watcher() -> None:
        while True:
            text = await text_queue.get()                   # suspends until text arrives
            # logger.info("[agent] text received: %r", text)
            async for result in text_to_speech_kokoro(text):
                pcm_bytes, sample_rate, visemes, sentence = result
                # logger.info("[agent] tts ready, queuing sentence: %r", sentence)
                await sentence_queue.put((pcm_bytes, sample_rate, visemes, sentence))

    loop = asyncio.get_running_loop()
    asyncio.create_task(_text_watcher())                    # runs independently

    while True:
        try:
            pcm_bytes, sample_rate, visemes, sentence = sentence_queue.get_nowait()
            logger.info("[agent] processing sentence: %r", sentence)
            next_t = await _play_sentence(
                video_source, audio_source, pcm_bytes, sample_rate, visemes, next_t, loop,
                idle_bytes=_idle,
            )
        except asyncio.QueueEmpty:
            video_source.capture_frame(rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, _idle))
            next_t += 1.0 / FPS
            await asyncio.sleep(max(0.0, next_t - perf_counter()))


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    # logger.info("[agent] starting room=%s", room_id)
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
        # logger.debug("[agent] data topic=%r from %s: %r", data.topic, data.participant.identity, text)
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
        # logger.info("[agent] video track published sid=%s room=%s", video_pub.sid, room_id)

        audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", audio_source)
        audio_pub = await room.local_participant.publish_track(
            audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        # logger.info("[agent] audio track published sid=%s room=%s", audio_pub.sid, room_id)

        # logger.info("[agent] starting stream loop room=%s", room_id)
        await _run_until_disconnected(_stream_loop(video_source, audio_source, text_queue), disconnected)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("[agent] room=%s error: %s", room_id, exc)
    finally:
        await room.disconnect()
        logger.info("[agent] done room=%s", room_id)
