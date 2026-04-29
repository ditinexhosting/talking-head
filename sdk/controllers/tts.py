from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from controllers.ws import WebSocketConnection

_ASSETS = Path(__file__).parent.parent / "assets"
_MODEL_PATH = str(_ASSETS / "kokoro-v1.0.onnx")
_VOICES_PATH = str(_ASSETS / "voices-v1.0.bin")

_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(_MODEL_PATH, _VOICES_PATH)
    return _kokoro


def split_into_sentences(text: str) -> list[str]:
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


async def text_to_speech_kokoro(
    conn: WebSocketConnection,
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    lang: str = "en-us",
):
    kokoro = _get_kokoro()
    sentences = split_into_sentences(text)

    for sentence in sentences:
        async for samples, sample_rate in kokoro.create_stream(
            sentence, voice=voice, speed=speed, lang=lang
        ):
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(pcm.tobytes()).decode()
            await conn.send_json({
                "session_id": conn.session_id,
                "event": "audio",
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
                "data": audio_b64,
            })
