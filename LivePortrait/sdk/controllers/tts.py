from __future__ import annotations

import asyncio
import logging
import re
from time import perf_counter

import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_emotion_classifier = None


def _get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        from transformers import pipeline as hf_pipeline
        logger.info("[tts] loading emotion classifier")
        _emotion_classifier = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0,  # cuda:0
        )
        logger.info("[tts] emotion classifier ready")
    return _emotion_classifier

_LANG_CODE_MAP = {
    "en-us": "a",
    "en-gb": "b"
}

_SAMPLE_RATE = 24000

_VOICE = "am_puck"

_pipeline_cache: dict[str, object] = {}


def _get_pipeline(lang_code: str):
    if lang_code not in _pipeline_cache:
        import warnings
        from kokoro import KPipeline
        logger.info("[tts] loading KPipeline lang_code=%s", lang_code)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = KPipeline(lang_code=lang_code, device="cuda", repo_id="hexgrad/Kokoro-82M")
        import torch
        model_device = next(pipe.model.parameters()).device if hasattr(pipe, "model") else "unknown"
        print(f"[tts] KPipeline loaded lang={lang_code} model_device={model_device} cuda_available={torch.cuda.is_available()}")
        _t_warm = perf_counter()
        for _ in pipe("hi", voice=_VOICE):
            pass
        print(f"[tts] warmup took {perf_counter() - _t_warm:.3f}s")
        _pipeline_cache[lang_code] = pipe
    return _pipeline_cache[lang_code]


def _split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx)\.', r'\1<DOT>', text)
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [chunk.strip().replace('<DOT>', '.') for chunk in raw if chunk.strip()]


def _split_into_clauses(sentence: str) -> list[str]:
    sentence = sentence.strip()
    if not sentence:
        return []
    raw = re.split(
        r'[,;:\-–—]+\s*|\s+(?=(?:because|although|while|when|if|unless|since|so|but|and|or|nor|yet)\s)',
        sentence,
        flags=re.IGNORECASE,
    )
    return [chunk.strip() for chunk in raw if chunk.strip()]


def _assign_clause_timings(emotions: list[dict], timings: list[dict]) -> list[dict]:
    t_idx = 0
    out = []
    for entry in emotions:
        words = [w.lower() for w in re.findall(r"[a-zA-Z']+", entry["clause"])]
        w_idx = 0
        clause_start = None
        clause_end = None
        scan = t_idx
        while scan < len(timings) and w_idx < len(words):
            gs_clean = re.sub(r"[^a-zA-Z']", "", timings[scan]["gs"]).lower()
            if gs_clean == words[w_idx]:
                if w_idx == 0:
                    clause_start = timings[scan]["start"]
                clause_end = timings[scan]["end"]
                w_idx += 1
            scan += 1
        if w_idx > 0:
            t_idx = scan
        out.append({**entry, "start": clause_start, "end": clause_end})
    return out


def _run_tts_sync(sentence: str, voice: str, speed: float, lang: str):
    lang_code = _LANG_CODE_MAP.get(lang, "a")
    sentence = sentence.strip()

    def _do_tts():
        pipeline = _get_pipeline(lang_code)
        gs_parts, ps_parts, audio_parts = [], [], []
        timings: list[dict] = []
        time_offset = 0.0
        for result in pipeline(sentence, voice=voice, speed=speed):
            gs_parts.append(result.graphemes)
            ps_parts.append(result.phonemes)
            if result.audio is not None:
                audio_parts.append(result.audio)
            for token in (result.tokens or []):
                if not token.phonemes:
                    continue
                start = getattr(token, 'start_ts', None)
                end = getattr(token, 'end_ts', None)
                if start is not None and end is not None:
                    timings.append({
                        "gs": token.text,
                        "ps": token.phonemes,
                        "start": round(time_offset + start, 4),
                        "end": round(time_offset + end, 4),
                    })
            if result.audio is not None:
                time_offset += len(result.audio) / 24000.0
        all_gs = " ".join(gs_parts)
        all_ps = " ".join(ps_parts)
        if audio_parts:
            all_audio = (np.clip(np.concatenate(audio_parts), -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        else:
            all_audio = b""
        return all_gs, all_ps, all_audio, timings

    def _do_emotion():
        try:
            clauses = _split_into_clauses(sentence)
            if not clauses:
                return []
            classifier = _get_emotion_classifier()
            out = []
            for clause, scores in zip(clauses, classifier(clauses)):
                top = sorted(scores, key=lambda x: x["score"], reverse=True)[0]
                out.append({"clause": clause, "emotion": top["label"], "score": round(top["score"], 4)})
            return out
        except Exception as exc:
            logger.warning("[tts] emotion failed: %s", exc)
            return []

    _t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        tts_future = ex.submit(_do_tts)
        emotion_future = ex.submit(_do_emotion)
        all_gs, all_ps, all_audio, timings = tts_future.result()
        emotion = emotion_future.result()
    emotion = _assign_clause_timings(emotion, timings)
    print(f"[tts] pipeline took {perf_counter() - _t0:.3f}s")
    print(all_gs, all_ps, timings)
    print(f"[tts] emotion: {emotion}")
    return all_gs, all_ps, all_audio, timings, emotion


async def text_to_speech_kokoro(
    text: str,
    voice: str = _VOICE,
    speed: float = 1.0,
    lang: str = "en-us",
) -> tuple[str, str, bytes, list[dict], list[dict]]:
    """Returns (gs, ps, pcm, timings, emotion) — graphemes, phonemes, int16 PCM bytes at 24 kHz, per-token timings, emotion scores."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: _run_tts_sync(text.strip(), voice, speed, lang)
    )
