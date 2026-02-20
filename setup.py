"""
Terka — setup skript
Nainstaluje závislosti, stáhne TTS modely a ověří Ollama.
Spusť jednou před prvním spuštěním: python setup.py
"""

import os
import sys
import subprocess
import urllib.request

# ── Cesty ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── TTS modely (kokoro-onnx) ───────────────────────────────────────────────────
KOKORO_FILES = {
    "kokoro-v1.0.int8.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/kokoro-v1.0.int8.onnx"
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.0/voices-v1.0.bin"
    ),
}

# ── Python závislosti ──────────────────────────────────────────────────────────
REQUIREMENTS = [
    "faster-whisper>=1.0.0",
    "ollama>=0.4.0",
    "sounddevice>=0.4.6",
    "soundfile>=0.12.1",
    "numpy>=1.24.0",
    "kokoro-onnx>=0.3.0",
    "scipy>=1.11.0",
]

# ── LLM model pro Ollama ───────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2:3b"


# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}")
    print("─" * 50)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def err(msg: str) -> None:
    print(f"  ✗ {msg}")


def _reporthook(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        done = count * block_size
        pct  = min(done * 100 // total_size, 100)
        mb   = done / 1024 / 1024
        tot  = total_size / 1024 / 1024
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {mb:.1f}/{tot:.1f} MB", end="", flush=True)


def install_requirements() -> None:
    step(1, 3, "Instalace Python závislostí")
    for req in REQUIREMENTS:
        print(f"  pip install {req} ...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", req, "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            print("✓")
        else:
            print("✗")
            warn(f"Selhalo: {result.stderr.decode()[:200]}")


def download_models() -> None:
    step(2, 3, "Stahování TTS modelů (kokoro-onnx)")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in KOKORO_FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            size_mb = os.path.getsize(dest) / 1024 / 1024
            ok(f"{filename} již existuje ({size_mb:.1f} MB)")
            continue

        print(f"\n  Stahuji {filename}...")
        try:
            urllib.request.urlretrieve(url, dest, _reporthook)
            print()  # newline po progress baru
            size_mb = os.path.getsize(dest) / 1024 / 1024
            ok(f"{filename} stažen ({size_mb:.1f} MB)")
        except Exception as e:
            print()
            err(f"Selhalo stahování {filename}: {e}")
            err("Zkus stáhnout ručně:")
            err(f"  {url}")
            err(f"  → ulož do {dest}")


def check_ollama() -> None:
    step(3, 3, "Kontrola Ollama + stažení LLM modelu")

    # Zkontroluj binárku ollama
    result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
    if result.returncode != 0:
        err("Ollama není nainstalována!")
        print()
        print("  Stáhni Ollama z: https://ollama.com/download")
        print("  Po instalaci spusť znovu: python setup.py")
        return

    ok(f"Ollama nalezena: {result.stdout.strip()}")

    # Stáhni model
    print(f"\n  Stahuji model '{OLLAMA_MODEL}' (může trvat i minuty)...")
    print("  (Zrušit: Ctrl+C)")
    result = subprocess.run(["ollama", "pull", OLLAMA_MODEL])
    if result.returncode == 0:
        ok(f"Model '{OLLAMA_MODEL}' připraven")
    else:
        warn(f"Stahování modelu selhalo. Zkus ručně: ollama pull {OLLAMA_MODEL}")


# ── Hlavní funkce ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 50)
    print("   Terka — AI hlasová asistentka")
    print("   Setup / první spuštění")
    print("=" * 50)

    install_requirements()
    download_models()
    check_ollama()

    print("\n" + "=" * 50)
    print("  Hotovo! Spusť Terku příkazem:")
    print()
    print("    python main.py")
    print()
    print("  Tip: Pro lepší kvalitu odpovědí zkus model:")
    print("    ollama pull mistral:7b")
    print("  a v config.py nastav OLLAMA_MODEL = 'mistral:7b'")
    print("=" * 50)


if __name__ == "__main__":
    main()
