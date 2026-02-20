"""
Terka — AI hlasová asistentka
Spusť: python main.py
"""

import sys

BANNER = """
╔══════════════════════════════════════════════╗
║           T E R K A   —   AI chat           ║
║         Hlasová asistentka (open-source)     ║
╠══════════════════════════════════════════════╣
║  Enter         → nahrát hlas (mluv)          ║
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

    # Přednahraj TTS model
    print()
    try:
        tts._get_kokoro()
    except FileNotFoundError as e:
        print(f"\n[CHYBA] {e}")
        sys.exit(1)

    print("\n[Terka je připravena!]\n")
    audio_device: int | None = None

    # ── Hlavní smyčka ─────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("Ty: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nNa shledanou!")
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
            tts.speak("Bye bye! I'll keep my evil plans for later. Fufufu.")
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

        try:
            reply = llm.chat(user_input)
        except Exception as e:
            print(f"\r[CHYBA LLM]: {e}")
            continue

        print(f"\rTerka: {reply}\n")
        tts.speak(reply)


if __name__ == "__main__":
    main()
