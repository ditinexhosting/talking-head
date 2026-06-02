from __future__ import annotations

import logging
import re
import time
import wave

logger = logging.getLogger(__name__)


# Checked before single chars (longest-first greedy match)
_IPA_MULTI: list[str] = sorted(
    [
        "tʃ", "dʒ",
        "iː", "uː", "ɑː", "ɔː", "ɜː",
        "eɪ", "oʊ", "aɪ", "aʊ", "ɔɪ",
    ],
    key=len,
    reverse=True,
)

# IPA phoneme → Rhubarb viseme shape (A-X)
IPA_TO_VISEME: dict[str, str] = {
    # A — closed lips (bilabials)
    "m": "A", "b": "A", "p": "A",
    # F — labiodental
    "f": "F", "v": "F",
    # H — labial rounded
    "w": "H", "uː": "H", "u": "H", "ʊ": "H",
    # G — front close
    "iː": "G", "i": "G", "ɪ": "G", "j": "G",
    # C — spread mid
    "eɪ": "C", "e": "C", "ɛ": "C", "æ": "C",
    # D — open mouth
    "ɑː": "D", "ɑ": "D", "a": "D", "aɪ": "D", "aʊ": "D", "ʌ": "D",
    # E — rounded mid-back
    "ɔː": "E", "ɔ": "E", "ɔɪ": "E", "oʊ": "E", "o": "E",
    # B — default (alveolars, velars, glottals, schwa)
    "t": "B", "d": "B", "s": "B", "z": "B",
    "n": "B", "l": "B", "r": "B", "ɹ": "B",
    "θ": "B", "ð": "B", "ʃ": "B", "ʒ": "B",
    "tʃ": "B", "dʒ": "B", "h": "B",
    "k": "B", "ɡ": "B", "g": "B", "ŋ": "B",
    "ə": "B", "ɜː": "B", "ɜ": "B",
    # X — silence / idle
    "": "X",
}

# Characters to skip during IPA tokenisation
_SKIP = frozenset("ˈˌ̩ ,.!?;:-()'\"[]{}\n\t")


def _tokenize_ipa(ipa: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] in _SKIP:
            i += 1
            continue
        matched = False
        for multi in _IPA_MULTI:
            end = i + len(multi)
            if ipa[i:end] == multi:
                tokens.append(multi)
                i = end
                matched = True
                break
        if not matched:
            tokens.append(ipa[i])
            i += 1
    return tokens


def _ipa_to_visemes(ipa_str: str) -> list[str]:
    tokens = _tokenize_ipa(ipa_str)
    visemes = [IPA_TO_VISEME.get(tok, "B") for tok in tokens]
    return visemes or ["B"]


def _split_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z'-]+", text)


def _wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def get_visemes(wav_path: str, transcript: str) -> list[dict]:
    """
    Return Rhubarb-format mouth cues:
      [{"start": float, "end": float, "duration": float, "viseme": str}, ...]

    Pipeline:
      1. Even word distribution — words spread uniformly across audio duration
      2. phonemizer            — word text → IPA per word  (~1 ms, CPU)
      3. Merge                 — distribute each word's phonemes across its time span
    """
    total_duration = _wav_duration(wav_path)
    words = _split_words(transcript)

    if not words:
        return [{
            "start": 0.0,
            "end": round(total_duration, 3),
            "duration": round(total_duration, 3),
            "viseme": "X",
        }]

    # Evenly distribute words across the audio duration
    word_dur = total_duration / len(words)
    aligned_words = [
        {"word": w, "start": i * word_dur, "end": (i + 1) * word_dur}
        for i, w in enumerate(words)
    ]

    _t1 = time.perf_counter()
    from phonemizer import phonemize as _phonemize
    ipa_list: list[str] = _phonemize(
        words,
        backend="espeak",
        language="en-us",
        with_stress=True,
        language_switch="remove-flags",
    )
    logger.info("[viseme] phonemizer: %.3fs (%d words)", time.perf_counter() - _t1, len(words))
    word_visemes: list[list[str]] = [_ipa_to_visemes(ipa) for ipa in ipa_list]

    # Distribute visemes across each word's time span
    cues: list[dict] = []
    for word_info, visemes in zip(aligned_words, word_visemes):
        start = word_info["start"]
        end = word_info["end"]
        n = len(visemes)
        step = (end - start) / n
        for k, v in enumerate(visemes):
            cues.append({
                "start":    round(start + k * step, 3),
                "end":      round(start + (k + 1) * step, 3),
                "duration": round(step, 3),
                "viseme":   v,
            })

    if not cues:
        return [{
            "start": 0.0,
            "end": round(total_duration, 3),
            "duration": round(total_duration, 3),
            "viseme": "X",
        }]

    # Pad with silence at start / end
    if cues[0]["start"] > 0.05:
        gap = cues[0]["start"]
        cues.insert(0, {"start": 0.0, "end": gap, "duration": round(gap, 3), "viseme": "X"})

    if cues[-1]["end"] < total_duration - 0.05:
        tail_start = cues[-1]["end"]
        tail_dur = round(total_duration - tail_start, 3)
        cues.append({
            "start": tail_start,
            "end": round(total_duration, 3),
            "duration": tail_dur,
            "viseme": "X",
        })

    return cues
