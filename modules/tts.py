"""
TTS modul — Fish Speech (OpenAudio S1 Mini) přes lokální HTTP API server.
Při prvním použití spustí Fish Speech API server jako subprocess.
"""

from __future__ import annotations
import sys
import os
import io
import base64
import json
import time
import subprocess
import urllib.request
import urllib.error

import numpy as np
import soundfile as sf

from config import (
    FISH_SPEECH_DIR, FISH_SPEECH_HOST, FISH_SPEECH_PORT,
    FISH_SPEECH_DEVICE, VOICE_SAMPLES_DIR,
)
from modules.audio import play_audio

_server_proc: subprocess.Popen | None = None


# ── Pomocné funkce ────────────────────────────────────────────────────────────

def _url(path: str = "") -> str:
    return f"http://{FISH_SPEECH_HOST}:{FISH_SPEECH_PORT}{path}"


def _is_server_alive() -> bool:
    try:
        with urllib.request.urlopen(_url("/v1/health"), timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _load_references() -> list[dict]:
    """Načte referenční audio vzorky ze složky voice_samples/ (pokud existují)."""
    if not os.path.isdir(VOICE_SAMPLES_DIR):
        return []
    refs = []
    for fname in sorted(os.listdir(VOICE_SAMPLES_DIR)):
        if not fname.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        audio_path = os.path.join(VOICE_SAMPLES_DIR, fname)
        txt_path = os.path.splitext(audio_path)[0] + ".txt"
        text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        refs.append({"audio": audio_b64, "text": text})
    return refs


# ── Spuštění serveru ──────────────────────────────────────────────────────────

def initialize() -> None:
    """Spustí Fish Speech API server pokud ještě neběží."""
    global _server_proc

    if _is_server_alive():
        print("  Fish Speech server již běží.")
        return

    if not os.path.isdir(FISH_SPEECH_DIR):
        raise FileNotFoundError(
            f"Fish Speech repozitář nenalezen: {FISH_SPEECH_DIR}\n"
            "Spusť: git clone https://github.com/fishaudio/fish-speech.git fish-speech"
        )

    checkpoints = os.path.join(FISH_SPEECH_DIR, "checkpoints", "openaudio-s1-mini")
    if not os.path.isdir(checkpoints):
        raise FileNotFoundError(
            f"Fish Speech model nenalezen: {checkpoints}\n"
            "Stáhni model: cd fish-speech && python tools/download_models.py"
        )

    print(f"  Spouštím Fish Speech server (port {FISH_SPEECH_PORT}, zařízení: {FISH_SPEECH_DEVICE})...")
    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "tools.api_server",
         "--device", FISH_SPEECH_DEVICE,
         "--listen", f"{FISH_SPEECH_HOST}:{FISH_SPEECH_PORT}"],
        cwd=FISH_SPEECH_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Čekáme na start (max 90 sekund — model se načítá)
    for _ in range(90):
        time.sleep(1)
        if _server_proc.poll() is not None:
            err = _server_proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Fish Speech server se ukončil předčasně:\n{err[-2000:]}")
        if _is_server_alive():
            print("  Fish Speech připraven.")
            return

    raise RuntimeError("Fish Speech server nenaběhl do 90 sekund.")


def shutdown() -> None:
    """Ukončí Fish Speech server (voláno při zavírání Terky)."""
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc = None


# ── Syntéza řeči ─────────────────────────────────────────────────────────────

def speak(text: str) -> bool:
    """
    Převede text na řeč pomocí Fish Speech a přehraje ho.
    Vrací True pokud bylo přerušeno Escem.
    """
    text = text.strip()
    if not text:
        return False

    refs = _load_references()

    payload = {
        "text": text,
        "references": refs,
        "format": "wav",
        "streaming": False,
        "normalize": True,
        "temperature": 0.8,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url("/v1/tts"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            wav_bytes = resp.read()
    except urllib.error.URLError as e:
        print(f"\r  [TTS chyba] {e}")
        return False
    except Exception as e:
        print(f"\r  [TTS chyba] {e}")
        return False

    # Dekóduj WAV → numpy float32
    audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # stereo → mono

    return play_audio(audio_data, sample_rate)
