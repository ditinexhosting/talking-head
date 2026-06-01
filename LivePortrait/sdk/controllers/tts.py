from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from sdk.controllers.viseme import get_visemes

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).parent.parent / "assets"
_MODEL_PATH = str(_ASSETS / "kokoro-v1.0.onnx")
_VOICES_PATH = str(_ASSETS / "voices-v1.0.bin")


_kokoro = None


def _preload_cudnn():
    """Preload cuDNN .so files so onnxruntime's CUDA provider can find them.

    The nvidia-cudnn-cu12 wheel installs into site-packages/nvidia/cudnn/lib/,
    which is not on LD_LIBRARY_PATH. ctypes.CDLL with RTLD_GLOBAL makes each
    library visible to subsequent dlopen() calls (including libonnxruntime_providers_cuda.so).
    """
    import ctypes
    import site
    from pathlib import Path as _Path

    for sp in site.getsitepackages():
        cudnn_lib = _Path(sp) / "nvidia" / "cudnn" / "lib"
        if not cudnn_lib.exists():
            continue
        load_order = [
            "libcudnn.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
        ]
        for name in load_order:
            p = cudnn_lib / name
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    logger.warning("[tts] failed to preload %s: %s", name, e)
        break


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        _preload_cudnn()
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




async def text_to_speech_kokoro(
    text: str,
    voice: str = "am_adam",
    speed: float = 1.0,
    lang: str = "en-us",
) -> tuple:
    """Returns (audio_bytes, sample_rate, visemes, sentence) for a single sentence."""
    kokoro = _get_kokoro()
    loop = asyncio.get_event_loop()
    sentence = text.strip()
    total_start = time.perf_counter()

    t = time.perf_counter()
    samples, sample_rate = await loop.run_in_executor(
        None, lambda: kokoro.create(sentence, voice=voice, speed=speed, lang=lang)
    )
    logger.info("[tts] kokoro.create=%.3fs", time.perf_counter() - t)

    t = time.perf_counter()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(tmp_fd, "wb") as f:
        f.write(_samples_to_wav_bytes(samples, sample_rate))

    try:
        t = time.perf_counter()
        visemes = await loop.run_in_executor(None, get_visemes, tmp_path, sentence)
        logger.info("[tts] visemes=%.3fs  count=%d", time.perf_counter() - t, len(visemes))
    finally:
        os.unlink(tmp_path)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    audio_bytes = pcm.tobytes()

    logger.info("[tts] total=%.3fs | %r", time.perf_counter() - total_start, sentence)
    return audio_bytes, sample_rate, visemes, sentence
