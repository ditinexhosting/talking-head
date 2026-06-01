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

_FRAME_PERIOD = 1.0 / FPS   # 40 ms
_BUFFER_FRAMES = 1         # pre-roll frames before playback starts


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


def _make_idle_rtc() -> rtc.VideoFrame | None:
    if idle_frame is None:
        return None
    return rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, idle_frame)


async def _stream_loop(
    video_source: rtc.VideoSource,
    audio_source: rtc.AudioSource | None,
    text_queue: asyncio.Queue[tuple[str, float]],
) -> None:
    # sentence_queue: (idx, pcm_bytes, sample_rate, sentence, frame_q, t_received|None)
    # t_received is set only on the first sentence of each message for TTFF logging
    sentence_queue: asyncio.Queue[tuple[int, bytes, int, str, asyncio.Queue, float | None]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    idle_rtc = _make_idle_rtc()

    async def _producer() -> None:
        idx = 0
        while True:
            text, t_received = await text_queue.get()
            is_first_sentence = True
            for sentence in _split_into_sentences(text):
                pcm_bytes, sample_rate, visemes, sentence = await text_to_speech_kokoro(sentence)

                viseme_sequence = {"visemes": visemes} if visemes else None
                per_frame_q: asyncio.Queue[rtc.VideoFrame | None] = asyncio.Queue()
                gen_done: asyncio.Event = asyncio.Event()
                await sentence_queue.put((idx, pcm_bytes, sample_rate, sentence, per_frame_q,
                                          t_received if is_first_sentence else None))
                is_first_sentence = False

                def _gen_frames(sidx: int = idx, vs=viseme_sequence, fq=per_frame_q, done=gen_done) -> None:
                    try:
                        for frame_rgb in liveportrait_frame_gen(vs):
                            loop.call_soon_threadsafe(fq.put_nowait, _to_rtc_frame(frame_rgb))
                    except Exception as exc:
                        logger.warning("[video] frame gen error: %s", exc)
                    finally:
                        loop.call_soon_threadsafe(fq.put_nowait, None)
                        loop.call_soon_threadsafe(done.set)
                        logger.debug("[video] frame gen done idx=%d", sidx)

                threading.Thread(target=_gen_frames, daemon=True, name=f"frame-gen-{idx}").start()
                idx += 1

    asyncio.create_task(_producer())

    # single frame-pump: runs at 25 FPS; shows idle whenever there is nothing ready
    current: tuple | None = None
    frame_idx = 0
    buffering = False
    next_t = perf_counter()

    while True:
        # --- acquire sentence ---
        if current is None:
            try:
                current = await asyncio.wait_for(sentence_queue.get(), timeout=_FRAME_PERIOD)
                frame_idx = 0
                buffering = True
            except asyncio.TimeoutError:
                if idle_rtc:
                    video_source.capture_frame(idle_rtc)
                next_t += _FRAME_PERIOD
                await asyncio.sleep(max(0.0, next_t - perf_counter()))
                continue

        idx, pcm_bytes, sample_rate, sentence, frame_q, t_received = current

        # --- pre-roll: wait until 10 frames buffered or generation done ---
        if buffering:
            qsize = frame_q.qsize()
            gen_done = qsize > 0 and any(item is None for item in frame_q._queue)
            if qsize >= _BUFFER_FRAMES or gen_done:
                buffering = False
            else:
                if idle_rtc:
                    video_source.capture_frame(idle_rtc)
                next_t += _FRAME_PERIOD
                await asyncio.sleep(max(0.0, next_t - perf_counter()))
                continue
        samples_per_frame = sample_rate // FPS
        bytes_per_frame = samples_per_frame * 2  # int16

        # --- acquire next frame (show idle if gen is behind) ---
        try:
            frame = await asyncio.wait_for(frame_q.get(), timeout=_FRAME_PERIOD)
        except asyncio.TimeoutError:
            if idle_rtc:
                video_source.capture_frame(idle_rtc)
            next_t += _FRAME_PERIOD
            await asyncio.sleep(max(0.0, next_t - perf_counter()))
            continue

        if frame is None:
            current = None   # sentence exhausted; pick up next one next iteration
            buffering = False
            continue

        # --- emit paired video + audio ---
        if frame_idx == 0 and t_received is not None:
            logger.info("[agent] time-to-first-frame: %.3fs", perf_counter() - t_received)

        video_source.capture_frame(frame)

        if audio_source is not None and pcm_bytes:
            start = frame_idx * bytes_per_frame
            chunk = pcm_bytes[start: start + bytes_per_frame]
            if len(chunk) < bytes_per_frame:
                chunk = chunk + bytes(bytes_per_frame - len(chunk))
            await audio_source.capture_frame(rtc.AudioFrame(
                data=chunk, sample_rate=sample_rate,
                num_channels=1, samples_per_channel=samples_per_frame,
            ))
        frame_idx += 1
        next_t += _FRAME_PERIOD
        await asyncio.sleep(max(0.0, next_t - perf_counter()))


async def run_agent(room_id: str, callback_url: str | None = None) -> None:
    # logger.info("[agent] starting room=%s", room_id)
    room = rtc.Room()
    loop = asyncio.get_running_loop()
    disconnected = asyncio.Event()
    text_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()

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
            loop.call_soon_threadsafe(text_queue.put_nowait, (text, perf_counter()))

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
