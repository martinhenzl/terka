"""
Terka — AI voice assistant
Run: python main.py
"""

import sys
import os
import msvcrt
import time

# Přidej CUDA DLLs do PATH (nvidia-cublas-cu12, nvidia-cudnn-cu12 z pip)
# Musíme přidat do PATH (ne jen os.add_dll_directory), aby ONNX Runtime
# mohl najít závislé DLLy při načítání CUDA provideru.
def _add_cuda_dlls() -> None:
    import site
    extra: list[str] = []
    for sp in site.getsitepackages():
        for pkg in ("cublas", "cudnn", "cuda_runtime", "cufft", "curand"):
            bin_path = os.path.join(sp, "nvidia", pkg, "bin")
            if os.path.isdir(bin_path):
                os.add_dll_directory(bin_path)
                extra.append(bin_path)
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")

_add_cuda_dlls()

BANNER = """
╔══════════════════════════════════════════════╗
║           T E R K A   —   AI chat           ║
║           Voice assistant (EN)               ║
╠══════════════════════════════════════════════╣
║  Enter         → record voice               ║
║  Esc           → interrupt (anytime)        ║
║  Type text     → skip microphone            ║
║  reset         → clear history              ║
║  devices       → list microphones           ║
║  quit / exit   → exit                       ║
╚══════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)

    # Import modulů až po zobrazení banneru
    from modules import stt, llm, tts
    from modules.audio import record_until_silence, list_input_devices

    # ── Initialization ────────────────────────────────────────────────────────
    print("[Initializing...]\n")

    # Check Ollama
    if not llm.check_connection():
        print("\nOllama not available. Check settings and try again.")
        sys.exit(1)

    # Pre-warm Whisper model (downloads automatically if missing)
    import numpy as np
    print()
    stt.transcribe(np.zeros(1600, dtype="float32"))

    # Start / verify TTS (Fish Speech server)
    print()
    try:
        tts.initialize()
    except Exception as e:
        print(f"\n[TTS ERROR] {e}")
        sys.exit(1)

    print("\n[Terka is ready!]\n")
    audio_device: int | None = None

    # ── Hlavní smyčka ─────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            tts.shutdown()
            break

        # ── Commands ──────────────────────────────────────────────────────────
        if not user_input:
            # Empty Enter → record voice
            audio = record_until_silence(device=audio_device)
            if audio is None:
                continue

            print("  Transcribing...")
            text = stt.transcribe(audio)
            if not text.strip():
                print("  Didn't catch that, try again.")
                continue
            user_input = text

        elif user_input.lower() in ("quit", "exit", "q"):
            print("Terka: Bye bye! Saving my evil plans for later. Heheh~")
            tts.speak("Bye bye! Saving my evil plans for later. Heheh.")
            tts.shutdown()
            break

        elif user_input.lower() == "reset":
            llm.reset_history()
            continue

        elif user_input.lower() in ("devices",):
            list_input_devices()
            try:
                idx = input("  Device number (Enter = default): ").strip()
                audio_device = int(idx) if idx else None
            except ValueError:
                audio_device = None
            continue

        elif user_input.lower() in ("help", "?"):
            print(BANNER)
            continue

        # ── Process message ───────────────────────────────────────────────────
        print(f"\nYou: {user_input}")
        print("Terka: ...", end="", flush=True)

        _esc = [False]
        def _stop_check() -> bool:
            if not _esc[0] and msvcrt.kbhit():
                if msvcrt.getch() == b'\x1b':
                    _esc[0] = True
            return _esc[0]

        t_start = time.time()
        try:
            reply = llm.chat(user_input, stop_check=_stop_check)
        except Exception as e:
            print(f"\r[LLM ERROR]: {e}")
            continue
        t_llm = time.time() - t_start

        if _esc[0] or not reply:
            print("\r  Interrupted.                      ")
            continue

        emotion, clean_reply = tts.extract_emotion(reply)
        print(f"\rTerka: {clean_reply}")
        print(f"  [LLM: {t_llm:.1f}s | {len(clean_reply.split())} words | {emotion}]")

        tts.speak(reply)


if __name__ == "__main__":
    main()
