"""
Audio modul — záznam z mikrofonu s detekcí ticha.
"""

from __future__ import annotations
import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    CHANNELS,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MIN_RECORD_DURATION,
)

CHUNK_SIZE = 1024  # vzorků na blok


def list_input_devices() -> None:
    """Vypíše dostupná vstupní audio zařízení."""
    print("\nDostupná vstupní zařízení:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " ← výchozí" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()


def record_until_silence(device: int | None = None) -> np.ndarray | None:
    """
    Nahrává z mikrofonu dokud není detekováno ticho.
    Vrací numpy float32 pole, nebo None pokud nic nezachytilo.
    """
    chunks: list[np.ndarray] = []
    silence_chunks   = 0
    speaking_started = False

    chunks_per_sec   = SAMPLE_RATE // CHUNK_SIZE
    required_silence = int(SILENCE_DURATION * chunks_per_sec)
    min_chunks       = int(MIN_RECORD_DURATION * chunks_per_sec)

    print("\n  Poslouchám... (mluv, po tichu se automaticky zastavím)")

    try:
        stream_kw: dict = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
        )
        if device is not None:
            stream_kw["device"] = device

        with sd.InputStream(**stream_kw) as stream:
            dots = 0
            while True:
                data, _ = stream.read(CHUNK_SIZE)
                rms = float(np.sqrt(np.mean(data ** 2)))

                if rms > SILENCE_THRESHOLD:
                    speaking_started = True
                    silence_chunks   = 0
                    chunks.append(data.copy())
                    bars = min(int(rms / SILENCE_THRESHOLD), 20)
                    print(f"\r  [{'█' * bars:<20}]", end="", flush=True)

                elif speaking_started:
                    chunks.append(data.copy())
                    silence_chunks += 1
                    pct = int(silence_chunks / required_silence * 20)
                    print(f"\r  [ticho {'░' * pct:<20}]", end="", flush=True)
                    if silence_chunks >= required_silence:
                        break

                else:
                    dots = (dots + 1) % 4
                    print(f"\r  Čekám na řeč{'.' * dots}   ", end="", flush=True)

    except KeyboardInterrupt:
        print()
        return None
    except Exception as e:
        print(f"\n  [audio chyba]: {e}")
        return None

    print()

    if len(chunks) < min_chunks:
        print("  Příliš krátké nebo tiché — zkus to znovu.")
        return None

    audio = np.concatenate(chunks, axis=0).flatten()
    print(f"  Zachyceno {len(audio) / SAMPLE_RATE:.1f}s zvuku.")
    return audio


def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    """Přehraje numpy pole jako audio a počká na dokončení."""
    sd.play(samples, samplerate=sample_rate)
    sd.wait()
