"""
Terka — AI hlasová asistentka
Spusť: python main.py
"""

import sys
import os
import msvcrt

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
║         Hlasová asistentka (open-source)     ║
╠══════════════════════════════════════════════╣
║  Enter         → nahrát hlas (mluv)          ║
║  Esc           → přerušit (kdykoli)          ║
║  Napsat text   → přeskočit mikrofon          ║
║  reset         → smazat historii             ║
║  zařízení      → zobrazit mikrofonů          ║
║  quit / exit   → ukončit                     ║
╚══════════════════════════════════════════════╝
"""


def main() -> None:
    print(BANNER)

    # Import modulů až po zobrazení banneru
    from modules import stt, llm, tts
    from modules.audio import record_until_silence, list_input_devices

    # ── Inicializace ─────────────────────────────────────────────────────────
    print("[Inicializace...]\n")

    # Zkontroluj Ollama
    if not llm.check_connection():
        print("\nOllama není dostupná. Zkontroluj nastavení a zkus znovu.")
        sys.exit(1)

    # Přednahraj Whisper model (stáhne se automaticky pokud chybí)
    import numpy as np
    print()
    stt.transcribe(np.zeros(1600, dtype="float32"))  # pre-warm — stáhne model pokud chybí

    # Přednahraj / ověř TTS (Fish Speech server)
    print()
    try:
        tts.initialize()
    except Exception as e:
        print(f"\n[CHYBA TTS] {e}")
        sys.exit(1)

    print("\n[Terka je připravena!]\n")
    audio_device: int | None = None

    # ── Hlavní smyčka ─────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("Ty: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nNa shledanou!")
            tts.shutdown()
            break

        # ── Příkazy ───────────────────────────────────────────────────────────
        if not user_input:
            # Prázdný Enter → záznam hlasu
            audio = record_until_silence(device=audio_device)
            if audio is None:
                continue

            print("  Přepisuji...")
            text = stt.transcribe(audio)
            if not text.strip():
                print("  Nic jsem neslyšela, zkus to znovu.")
                continue
            user_input = text

        elif user_input.lower() in ("quit", "exit", "konec", "q"):
            print("Terka: Čau čau! Zlé plány si nechám na jindy. Fufufu~")
            tts.speak("Čau čau! Zlé plány si nechám na jindy. Fufufu.")
            tts.shutdown()
            break

        elif user_input.lower() == "reset":
            llm.reset_history()
            continue

        elif user_input.lower() in ("zařízení", "zarizeni", "devices"):
            list_input_devices()
            try:
                idx = input("  Číslo zařízení (Enter = výchozí): ").strip()
                audio_device = int(idx) if idx else None
            except ValueError:
                audio_device = None
            continue

        elif user_input.lower() in ("help", "pomoc", "?"):
            print(BANNER)
            continue

        # ── Zpracování zprávy ─────────────────────────────────────────────────
        print(f"\nTy: {user_input}")
        print("Terka: ...", end="", flush=True)

        # Esc přeruší generování mezi tokeny
        _esc = [False]
        def _stop_check() -> bool:
            if not _esc[0] and msvcrt.kbhit():
                if msvcrt.getch() == b'\x1b':
                    _esc[0] = True
            return _esc[0]

        try:
            reply = llm.chat(user_input, stop_check=_stop_check)
        except Exception as e:
            print(f"\r[CHYBA LLM]: {e}")
            continue

        if _esc[0] or not reply:
            print("\r  Přerušeno.                        ")
            continue

        print(f"\rTerka: {reply}\n")
        tts.speak(reply)


if __name__ == "__main__":
    main()
