from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).parent.parent / "assets"
_MODEL_PATH = str(_ASSETS / "kokoro-v1.0.onnx")
_VOICES_PATH = str(_ASSETS / "voices-v1.0.bin")
_RHUBARB = str(_ASSETS / "rhubarb-1.14.0" / "rhubarb")


_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        import onnxruntime as rt
        from kokoro_onnx import Kokoro
        available = rt.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        logger.info("[tts] onnxruntime providers: %s", providers)
        session = rt.InferenceSession(_MODEL_PATH, providers=providers)
        _kokoro = Kokoro.from_session(session, _VOICES_PATH)
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



async def text_to_speech_kokoro(
    text: str,
    voice: str = "am_adam",
    speed: float = 1.0,
    lang: str = "en-us",
):
    """Async generator — yields (pcm_bytes, sample_rate, visemes) per sentence."""
    kokoro = _get_kokoro()
    loop = asyncio.get_event_loop()
    total_start = time.perf_counter()

    for idx, sentence in enumerate(_split_into_sentences(text)):
        t = time.perf_counter()
        samples, sample_rate = await loop.run_in_executor(
            None, lambda s=sentence: kokoro.create(s, voice=voice, speed=speed, lang=lang)
        )
        kokoro_s = time.perf_counter() - t
        logger.info("[tts] s%d kokoro.create=%.3fs", idx, kokoro_s)

        t = time.perf_counter()
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(_samples_to_wav_bytes(samples, sample_rate))
        wav_s = time.perf_counter() - t
        logger.info("[tts] s%d wav_write=%.3fs", idx, wav_s)

        try:
            t = time.perf_counter()
            visemes = await _run_rhubarb(tmp_path, sentence)
            rhubarb_s = time.perf_counter() - t
            logger.info("[tts] s%d rhubarb=%.3fs  visemes=%d", idx, rhubarb_s, len(visemes))
        finally:
            os.unlink(tmp_path)

        t = time.perf_counter()
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        pcm_s = time.perf_counter() - t
        logger.info("[tts] s%d pcm_convert=%.3fs", idx, pcm_s)

        logger.info("[tts] s%d total=%.3fs | %r", idx, kokoro_s + wav_s + rhubarb_s + pcm_s, sentence)

        yield audio_bytes, sample_rate, visemes

    logger.info("[tts] grand_total=%.3fs for full text", time.perf_counter() - total_start)
