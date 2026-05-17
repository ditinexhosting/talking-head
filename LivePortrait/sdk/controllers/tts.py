from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from sdk.controllers.ws import WebSocketConnection

_ASSETS = Path(__file__).parent.parent / "assets"
_MODEL_PATH = str(_ASSETS / "kokoro-v1.0.onnx")
_VOICES_PATH = str(_ASSETS / "voices-v1.0.bin")
_RHUBARB = str(_ASSETS / "rhubarb-1.14.0" / "rhubarb")
_SAMPLE_DIR = _ASSETS / "sample"
_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(_MODEL_PATH, _VOICES_PATH)
    return _kokoro


def _split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx)\.', r'\1<DOT>', text)
    pattern = r'(?<=[.!?])\s+'
    raw = re.split(pattern, text)
    sentences = []
    for chunk in raw:
        chunk = chunk.strip().replace('<DOT>', '.')
        if chunk:
            sentences.append(chunk)
    return sentences


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _enrich_visemes(mouth_cues: list[dict]) -> list[dict]:
    result = []
    for cue in mouth_cues:
        start = round(cue["start"], 3)
        end = round(cue["end"], 3)
        viseme = cue["value"]
        result.append({
            "start":       start,
            "end":         end,
            "duration":    round(end - start, 3),
            "viseme":      viseme
        })
    return result


async def _run_rhubarb(wav_path: str, transcript: str) -> list[dict]:
    dialog_fd, dialog_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(dialog_fd, "w") as f:
            f.write(transcript)
        proc = await asyncio.create_subprocess_exec(
            _RHUBARB, "-f", "json", "--machineReadable", "-q",
            "--recognizer", "phonetic", "-d", dialog_path, wav_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
    finally:
        os.unlink(dialog_path)

    if proc.returncode != 0:
        return []
    mouth_cues = json.loads(stdout).get("mouthCues", [])
    return _enrich_visemes(mouth_cues)


_TAG_AUDIO = b"\x01"
_TAG_VIDEO = b"\x02"


async def _send_interleaved(conn: WebSocketConnection, audio_bytes: bytes, video_bytes: bytes | None, chunk: int = 64 * 1024):
    a_len = len(audio_bytes)
    v_len = len(video_bytes) if video_bytes else 0
    a_off = v_off = 0
    while a_off < a_len or v_off < v_len:
        if a_off < a_len:
            await conn.send_bytes(_TAG_AUDIO + audio_bytes[a_off:a_off + chunk])
            a_off += chunk
        if v_off < v_len:
            await conn.send_bytes(_TAG_VIDEO + video_bytes[v_off:v_off + chunk])
            v_off += chunk


async def text_to_speech_kokoro_sample(conn: WebSocketConnection):
    for name in sorted(p.name for p in _SAMPLE_DIR.glob("[0-9][0-9][0-9].json")):
        payload = json.loads((_SAMPLE_DIR / name).read_text())
        audio_bytes = base64.b64decode(payload["audio"]["data"])
        video_bytes = base64.b64decode(payload["video"]["data"]) if payload.get("video") else None

        await conn.send_json({
            "session_id": conn.session_id,
            "event": "chunk_start",
            "audio": {"sample_rate": payload["audio"]["sample_rate"], "encoding": payload["audio"]["encoding"], "size": len(audio_bytes)},
            "video": {"encoding": payload["video"]["encoding"], "size": len(video_bytes)} if video_bytes else None,
        })

        await _send_interleaved(conn, audio_bytes, video_bytes)

        await conn.send_json({
            "session_id": conn.session_id,
            "event": "chunk_end"
        })
        await asyncio.sleep(1)


async def text_to_speech_kokoro(
    conn: WebSocketConnection,
    text: str,
    voice: str = "am_adam",
    speed: float = 1.0,
    lang: str = "en-us",
):
    kokoro = _get_kokoro()

    for idx, sentence in enumerate(_split_into_sentences(text)):
        samples, sample_rate = kokoro.create(sentence, voice=voice, speed=speed, lang=lang)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(_samples_to_wav_bytes(samples, sample_rate))
            visemes = await _run_rhubarb(tmp_path, sentence)
        finally:
            os.unlink(tmp_path)

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        (_SAMPLE_DIR / "visemes_debug.json").write_text(json.dumps({"visemes": visemes}, indent=2))
        #video_bytes = await speech_to_video(visemes) if visemes else None
        video_bytes = None

        sample_payload = {
            "audio": {
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            },
            "video": {
                "encoding": "mp4",
                "data": base64.b64encode(video_bytes).decode("ascii"),
            } if video_bytes else None,
        }
        (_SAMPLE_DIR / f"{idx:03d}.json").write_text(json.dumps(sample_payload))

        await conn.send_json({
            "session_id": conn.session_id,
            "event": "chunk_start",
            "audio": {"sample_rate": sample_rate, "encoding": "pcm_s16le", "size": len(audio_bytes)},
            "video": {"encoding": "mp4", "size": len(video_bytes)} if video_bytes else None,
        })

        await _send_interleaved(conn, audio_bytes, video_bytes)

        await conn.send_json({
            "session_id": conn.session_id,
            "event": "chunk_end"
        })
