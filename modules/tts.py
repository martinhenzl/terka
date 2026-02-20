"""
TTS modul — Text to Speech pomocí kokoro-onnx.
Generuje řeč dívčím hlasem a přehraje ji.
"""

from __future__ import annotations
import os
import re
import numpy as np

from config import TTS_MODEL_PATH, TTS_VOICES_PATH, TTS_VOICE, TTS_SPEED, TTS_LANG
from modules.audio import play_audio

_kokoro = None
_MAX_CHUNK = 300  # znaky — kokoro má lepší výsledky s kratšími texty


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        if not os.path.exists(TTS_MODEL_PATH):
            raise FileNotFoundError(
                f"TTS model nenalezen: {TTS_MODEL_PATH}\n"
                "Spusť: python setup.py"
            )
        if not os.path.exists(TTS_VOICES_PATH):
            raise FileNotFoundError(
                f"TTS hlasy nenalezeny: {TTS_VOICES_PATH}\n"
                "Spusť: python setup.py"
            )
        from kokoro_onnx import Kokoro
        print("  Načítám Kokoro TTS model...")
        _kokoro = Kokoro(TTS_MODEL_PATH, TTS_VOICES_PATH)
        print("  TTS připraven.")
    return _kokoro


def _split_into_chunks(text: str) -> list[str]:
    """Rozdělí text na věty vhodné pro TTS."""
    # Rozdělení na věty podle interpunkce
    sentences = re.split(r"(?<=[.!?~])\s+", text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < _MAX_CHUNK:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


def speak(text: str) -> None:
    """Převede text na řeč a přehraje ho."""
    text = text.strip()
    if not text:
        return

    kokoro = _get_kokoro()
    chunks = _split_into_chunks(text)

    for chunk in chunks:
        if not chunk.strip():
            continue
        samples, sample_rate = kokoro.create(
            chunk,
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            lang=TTS_LANG,
        )
        play_audio(np.array(samples, dtype=np.float32), sample_rate)
