"""
STT module — Speech to Text using faster-whisper.
"""

from __future__ import annotations
import gc
import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_BEAM_SIZE, WHISPER_INITIAL_PROMPT

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        print(f"  Loading Whisper '{WHISPER_MODEL_SIZE}' ({device}, {compute_type})...")
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
        print("  Whisper ready.")
    return _model


def transcribe(audio: np.ndarray, silent: bool = False) -> str:
    """
    Transcribe a numpy audio array to text.
    Returns empty string if nothing was recognized.
    Pass silent=True to suppress all console output (e.g. during warmup).
    """
    model = _get_model()

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,  # prevents hallucination loops
        no_speech_threshold=0.6,           # drop near-silence segments
    )

    text = " ".join(seg.text for seg in segments).strip()

    if not silent:
        if text:
            lang_info = f" [{info.language}, {info.language_probability:.0%}]"
            print(f"  Recognized{lang_info}: {text}")
        else:
            print("  Nothing recognized.")

    return text


def shutdown() -> None:
    """Explicitly destroy WhisperModel to free CUDA memory before process exit.
    gc.collect() forces the CTranslate2 destructor to run NOW, while the CUDA
    context is still alive. Without this, ctranslate2's CudaAsyncAllocator throws
    during DLL_PROCESS_DETACH (after ExitProcess) → STATUS_STACK_BUFFER_OVERRUN."""
    global _model
    if _model is not None:
        _model = None
        gc.collect()
