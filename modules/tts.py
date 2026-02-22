"""
TTS module — supports two backends selectable via config.TTS_BACKEND:

  "kokoro" — local neural TTS (Kokoro), ~0.3–1 s latency, GPU accelerated.
             install: pip install kokoro soundfile
  "fish"   — Fish Speech openaudio-s1-mini via HTTP API, ~5–15 s, voice cloning.
"""

from __future__ import annotations
import sys
import os
import re
import base64
import json
import time
import msvcrt
import threading
import subprocess
import urllib.request
import urllib.error

import numpy as np
import sounddevice as sd

from config import (
    TTS_BACKEND,
    # Kokoro
    KOKORO_VOICE, KOKORO_SPEED,
    # Fish Speech
    FISH_SPEECH_DIR, FISH_SPEECH_HOST, FISH_SPEECH_PORT,
    FISH_SPEECH_DEVICE, FISH_SPEECH_WSL, FISH_SPEECH_COMPILE, VOICE_SAMPLES_DIR,
    FISH_SPEECH_SAMPLE_RATE, FISH_SPEECH_CHANNELS,
    VOICE_REFERENCES_MAX,
)

# ── Emotion tags ───────────────────────────────────────────────────────────────
_EMOTION_RE = re.compile(r"^\[(\w+)\]\s*")


def extract_emotion(text: str) -> tuple[str, str]:
    """
    Parse leading [emotion] tag from LLM output.
    Returns (label, clean_text) — tag is stripped from clean_text.
    """
    m = _EMOTION_RE.match(text)
    if m:
        return m.group(1).lower(), text[m.end():]
    return "neutral", text


def _clean_for_tts(text: str) -> str:
    """Strip emotion tag and action markers from text before sending to TTS."""
    _, clean = extract_emotion(text)
    return re.sub(r'\*[^*]+\*', '', clean).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Kokoro backend
# ══════════════════════════════════════════════════════════════════════════════

_kokoro_pipeline = None   # KPipeline instance, loaded once


def _get_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        print("  Loading Kokoro TTS model...", flush=True)
        _kokoro_pipeline = KPipeline(lang_code="a")  # "a" = American English
        print("  Kokoro ready.")
    return _kokoro_pipeline


def _speak_kokoro(text: str) -> bool:
    """Speak text via Kokoro. Returns True if interrupted by Esc."""
    tts_text = _clean_for_tts(text)
    if not tts_text:
        return False

    pipeline = _get_kokoro()
    t_start = time.time()
    first = True

    try:
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32") as stream:
            for _gs, _ps, audio in pipeline(tts_text, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
                if first:
                    first = False
                    print(f"  [TTS first audio: {time.time() - t_start:.1f}s]", flush=True)
                if msvcrt.kbhit() and msvcrt.getch() == b"\x1b":
                    return True
                # Kokoro yields float32 mono at 24 kHz
                stream.write(audio.reshape(-1, 1) if audio.ndim == 1 else audio)
    except Exception as e:
        print(f"\r  [TTS error] {e}")

    return False


# ══════════════════════════════════════════════════════════════════════════════
# Fish Speech backend
# ══════════════════════════════════════════════════════════════════════════════

_server_proc: subprocess.Popen | None = None
_references_cache: list[dict] | None = None


def _url(path: str = "") -> str:
    return f"http://{FISH_SPEECH_HOST}:{FISH_SPEECH_PORT}{path}"


def _is_server_alive() -> bool:
    for path in ("/health", "/v1/health", "/"):
        try:
            with urllib.request.urlopen(_url(path), timeout=2) as r:
                if r.status < 500:
                    return True
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return True
        except Exception:
            pass
    return False


def _load_references() -> list[dict]:
    global _references_cache
    if _references_cache is not None:
        return _references_cache

    if not os.path.isdir(VOICE_SAMPLES_DIR):
        _references_cache = []
        return _references_cache

    audio_exts = {".wav", ".mp3", ".flac"}
    files = [
        f for f in os.listdir(VOICE_SAMPLES_DIR)
        if os.path.splitext(f)[1].lower() in audio_exts
    ]
    files.sort(key=lambda f: os.path.getsize(os.path.join(VOICE_SAMPLES_DIR, f)))
    files = files[:VOICE_REFERENCES_MAX]

    refs = []
    for fname in files:
        audio_path = os.path.join(VOICE_SAMPLES_DIR, fname)
        txt_path = os.path.splitext(audio_path)[0] + ".txt"
        text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        refs.append({"audio": audio_b64, "text": text})

    total_kb = sum(
        os.path.getsize(os.path.join(VOICE_SAMPLES_DIR, f)) for f in files
    ) // 1024
    print(f"  Voice references cached: {len(refs)} file(s), {total_kb} KB")
    _references_cache = refs
    return _references_cache


def _start_fish_server() -> None:
    global _server_proc

    if _is_server_alive():
        print("  Fish Speech server already running.")
        _load_references()
        return

    if not os.path.isdir(FISH_SPEECH_DIR):
        raise FileNotFoundError(
            f"Fish Speech repository not found: {FISH_SPEECH_DIR}\n"
            "Run: git clone https://github.com/fishaudio/fish-speech.git fish-speech"
        )

    checkpoints = os.path.join(FISH_SPEECH_DIR, "checkpoints", "openaudio-s1-mini")
    if not os.path.isdir(checkpoints):
        raise FileNotFoundError(
            f"Fish Speech model not found: {checkpoints}\n"
            "Download: cd fish-speech && python tools/download_models.py"
        )

    compile_note = " +compile" if FISH_SPEECH_COMPILE else ""
    wsl_note = " [WSL2]" if FISH_SPEECH_WSL else ""
    print(f"  Starting Fish Speech server (port {FISH_SPEECH_PORT}, device: {FISH_SPEECH_DEVICE}{compile_note}{wsl_note})...")

    if FISH_SPEECH_WSL:
        # Convert Windows path to WSL2 path (e.g. C:\Terka\fish-speech → /mnt/c/Terka/fish-speech)
        drive, rest = FISH_SPEECH_DIR.replace("\\", "/").split(":/", 1)
        wsl_dir = f"/mnt/{drive.lower()}/{rest}"
        shell_cmd = (
            f"cd {wsl_dir!r} && "
            f"/root/.local/bin/uv run python3 -m tools.api_server "
            f"--device {FISH_SPEECH_DEVICE} "
            f"--listen 0.0.0.0:{FISH_SPEECH_PORT}"
        )
        if FISH_SPEECH_COMPILE:
            shell_cmd += " --compile"
        _server_proc = subprocess.Popen(
            ["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", "-c", shell_cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    else:
        cmd = [sys.executable, "-m", "tools.api_server",
               "--device", FISH_SPEECH_DEVICE,
               "--listen", f"{FISH_SPEECH_HOST}:{FISH_SPEECH_PORT}"]
        if FISH_SPEECH_COMPILE:
            cmd.append("--compile")
        _server_proc = subprocess.Popen(
            cmd,
            cwd=FISH_SPEECH_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    _stderr_lines: list[str] = []
    def _read_stderr() -> None:
        try:
            for line in _server_proc.stderr:
                _stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass
    threading.Thread(target=_read_stderr, daemon=True).start()

    TIMEOUT = 240
    print(f"  Waiting for model load (up to {TIMEOUT}s on first start)...", flush=True)
    for i in range(TIMEOUT):
        time.sleep(1)
        if _server_proc.poll() is not None:
            err = "\n".join(_stderr_lines[-40:])
            raise RuntimeError(f"Fish Speech server exited early:\n{err or '(no output)'}")
        if _is_server_alive():
            print(f"  Fish Speech ready. ({i + 1}s)")
            _load_references()
            return
        if (i + 1) % 15 == 0:
            print(f"  Still loading... ({i + 1}/{TIMEOUT}s)", flush=True)

    err = "\n".join(_stderr_lines[-40:])
    raise RuntimeError(
        f"Fish Speech server did not respond within {TIMEOUT}s.\n"
        f"Last server output:\n{err or '(none)'}"
    )


def _speak_fish(text: str) -> bool:
    """Speak text via Fish Speech. Returns True if interrupted by Esc."""
    tts_text = _clean_for_tts(text)
    if not tts_text:
        return False

    refs = _load_references()
    payload = {
        "text": tts_text,
        "references": refs,
        "format": "wav",
        "streaming": True,
        "normalize": True,
        "temperature": 0.7,
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

    t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            CHUNK = 4096
            with sd.OutputStream(
                samplerate=FISH_SPEECH_SAMPLE_RATE,
                channels=FISH_SPEECH_CHANNELS,
                dtype="int16",
            ) as stream:
                first = True
                while True:
                    if msvcrt.kbhit() and msvcrt.getch() == b"\x1b":
                        return True
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    if first:
                        first = False
                        if chunk[:4] == b"RIFF":
                            chunk = chunk[44:]
                        if not chunk:
                            continue
                        print(f"  [TTS first audio: {time.time() - t_start:.1f}s]", flush=True)
                    stream.write(np.frombuffer(chunk, dtype=np.int16))
    except urllib.error.URLError as e:
        print(f"\r  [TTS error] {e}")
        return False
    except Exception as e:
        print(f"\r  [TTS error] {e}")
        return False

    return False


def fetch_audio(text: str) -> list[np.ndarray]:
    """Fetch Fish Speech audio into memory buffer (for prefetch use)."""
    tts_text = _clean_for_tts(text)
    if not tts_text:
        return []

    refs = _load_references()
    payload = {
        "text": tts_text, "references": refs, "format": "wav",
        "streaming": True, "normalize": True,
        "temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.1,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url("/v1/tts"), data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    chunks: list[np.ndarray] = []
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            first = True
            while True:
                raw = resp.read(4096)
                if not raw:
                    break
                if first:
                    first = False
                    if raw[:4] == b"RIFF":
                        raw = raw[44:]
                    if not raw:
                        continue
                chunks.append(np.frombuffer(raw, dtype=np.int16).copy())
    except Exception as e:
        print(f"\r  [TTS prefetch error] {e}")
    return chunks


def play_audio(chunks: list[np.ndarray]) -> bool:
    """Play pre-fetched Fish Speech audio. Returns True if interrupted."""
    if not chunks:
        return False
    with sd.OutputStream(
        samplerate=FISH_SPEECH_SAMPLE_RATE,
        channels=FISH_SPEECH_CHANNELS,
        dtype="int16",
    ) as stream:
        for chunk in chunks:
            if msvcrt.kbhit() and msvcrt.getch() == b"\x1b":
                return True
            stream.write(chunk)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def initialize() -> None:
    """Load / start the TTS backend."""
    if TTS_BACKEND == "kokoro":
        _get_kokoro()   # pre-load model so first speak() is fast
    else:
        _start_fish_server()


def shutdown() -> None:
    """Clean up TTS resources on exit."""
    global _server_proc
    if TTS_BACKEND == "fish" and _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc = None


def speak(text: str) -> bool:
    """Convert text to speech and play it. Returns True if interrupted by Esc."""
    text = text.strip()
    if not text:
        return False
    if TTS_BACKEND == "kokoro":
        return _speak_kokoro(text)
    return _speak_fish(text)
