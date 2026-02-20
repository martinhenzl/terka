"""
STT modul — Speech to Text pomocí faster-whisper.
"""

from __future__ import annotations
import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"  Načítám Whisper '{WHISPER_MODEL_SIZE}' ({device}, {compute_type})...")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
        print("  Whisper připraven.")
    return _model


def transcribe(audio: np.ndarray) -> str:
    """
    Přepíše numpy audio pole na text.
    Vrací prázdný řetězec pokud nic nebylo rozpoznáno.
    """
    model = _get_model()

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    text = " ".join(seg.text for seg in segments).strip()

    if text:
        lang_info = f" [{info.language}, {info.language_probability:.0%}]"
        print(f"  Rozpoznáno{lang_info}: {text}")
    else:
        print("  Nic nerozpoznáno.")

    return text
