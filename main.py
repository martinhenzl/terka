"""
Terka — AI voice assistant
Run:  python main.py             — manual mode (Enter to record)
      python main.py -c          — continuous mode (auto-listen after each reply)
"""

import sys
import os
import msvcrt
import time
import argparse

# Add CUDA DLLs to PATH (nvidia-cublas-cu12, nvidia-cudnn-cu12 from pip).
# Must add to PATH (not just os.add_dll_directory) so ONNX Runtime
# can find dependent DLLs when loading the CUDA provider.
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

def _make_banner() -> str:
    W = 46
    top = "╔" + "═" * W + "╗"
    mid = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def c(s): return "║" + s.center(W) + "║"
    def r(s): return "║  " + s.ljust(W - 2) + "║"
    return "\n".join([
        "", top,
        c("T E R K A   —   AI chat"),
        c("Voice assistant (EN)"),
        mid,
        r("Enter         → record voice"),
        r("Esc           → interrupt (anytime)"),
        r("Type text     → skip microphone"),
        r("reset         → clear history"),
        r("devices       → list microphones"),
        r("quit / exit   → exit"),
        mid,
        r("-c flag        → continuous listening mode"),
        r("(Esc pauses, Enter resumes)"),
        r("-voice jahoda  → switch voice character"),
        r("-voice amber   → use Amber's voice"),
        bot, "",
    ])

BANNER = _make_banner()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python main.py", add_help=False)
    parser.add_argument("-c", "--continuous", action="store_true",
                        help="Continuous listening mode")
    parser.add_argument("-voice", dest="voice", default=None,
                        help="Voice character to use (e.g. jahoda, amber)")
    args, _ = parser.parse_known_args()

    if args.voice:
        import config as _cfg
        _cfg.CHARACTER_VOICE = args.voice.lower()

    print(BANNER)

    # Import modules after banner is shown
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
    stt.transcribe(np.zeros(1600, dtype="float32"), silent=True)

    # Start / verify TTS (Fish Speech server)
    print()
    try:
        tts.initialize()
    except Exception as e:
        print(f"\n[TTS ERROR] {e}")
        sys.exit(1)

    print("\n[Terka is ready!]\n")

    greeting = llm.greet()
    if greeting:
        _, clean_greeting = tts.extract_emotion(greeting)
        print(f"Terka: {clean_greeting}\n")
        tts.speak(greeting)

    audio_device: int | None = None
    continuous = args.continuous

    if continuous:
        print("[Continuous mode active — Esc to pause, Enter to resume]\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        user_input: str | None = None

        # ── Input acquisition ─────────────────────────────────────────────────
        if continuous:
            esc_flag = [False]
            audio = record_until_silence(device=audio_device, esc_out=esc_flag)

            if esc_flag[0]:
                continuous = False
                print("[Continuous paused — type text, or press Enter to resume]\n")
                continue

            if audio is None:
                continue  # too short / quiet — listen again

            print("  Transcribing...", flush=True)
            text = stt.transcribe(audio)
            if not text.strip():
                print("  Nothing recognized — listening again...")
                continue

            user_input = text
            print()

        else:
            # Manual mode
            try:
                raw = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                stt.shutdown()
                tts.shutdown()
                os._exit(0)

            if not raw:
                if args.continuous:
                    # Resume continuous mode with Enter
                    continuous = True
                    print("[Continuous mode resumed]\n")
                    continue
                else:
                    # Single recording
                    esc_flag = [False]
                    audio = record_until_silence(device=audio_device, esc_out=esc_flag)
                    if audio is None:
                        continue
                    print("  Transcribing...")
                    text = stt.transcribe(audio)
                    if not text.strip():
                        print("  Didn't catch that, try again.")
                        continue
                    user_input = text
                    print()

            elif raw.lower() in ("quit", "exit", "q"):
                farewell = llm.chat("(The user is leaving. Say a short playful goodbye — one sentence.)")
                _, clean_farewell = tts.extract_emotion(farewell)
                print(f"Terka: {clean_farewell}")
                tts.speak(farewell)
                stt.shutdown()
                tts.shutdown()
                os._exit(0)

            elif raw.lower() == "reset":
                llm.reset_history()
                continue

            elif raw.lower() == "devices":
                list_input_devices()
                try:
                    idx = input("  Device number (Enter = default): ").strip()
                    audio_device = int(idx) if idx else None
                except ValueError:
                    audio_device = None
                continue

            elif raw.lower() in ("help", "?"):
                print(BANNER)
                continue

            else:
                user_input = raw
                print(f"\nYou: {user_input}")

        # ── Process message ───────────────────────────────────────────────────
        assert user_input is not None
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
            if continuous:
                continuous = False
                print("[Continuous paused — press Enter to resume]\n")
            continue

        emotion, clean_reply = tts.extract_emotion(reply)
        print(f"\rTerka: {clean_reply}")
        print(f"  [LLM: {t_llm:.1f}s | {len(clean_reply.split())} words | {emotion}]")

        interrupted = tts.speak(reply)

        if interrupted and continuous:
            continuous = False
            print("\n[Continuous paused — press Enter to resume]\n")


if __name__ == "__main__":
    main()
