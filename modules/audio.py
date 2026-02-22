"""
Audio module — microphone recording with silence detection.
"""

from __future__ import annotations
import msvcrt
import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    CHANNELS,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    MIN_RECORD_DURATION,
)

CHUNK_SIZE = 1024  # samples per block


def list_input_devices() -> None:
    """Print available input audio devices."""
    print("\nAvailable input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()


def record_until_silence(
    device: int | None = None,
    esc_out: list[bool] | None = None,
) -> np.ndarray | None:
    """
    Record from microphone until silence is detected.
    Returns numpy float32 array, or None if nothing was captured.
    If Esc is pressed, sets esc_out[0] = True (if provided) and returns None.
    """
    chunks: list[np.ndarray] = []
    silence_chunks   = 0
    speaking_started = False

    chunks_per_sec   = SAMPLE_RATE // CHUNK_SIZE
    required_silence = int(SILENCE_DURATION * chunks_per_sec)
    min_chunks       = int(MIN_RECORD_DURATION * chunks_per_sec)

    print("\n  Listening... (speak, stops automatically on silence — Esc = cancel)")

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
                # Esc = cancel recording
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x1b':
                        if esc_out is not None:
                            esc_out[0] = True
                        print("\r  Recording cancelled.          ")
                        return None

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
                    print(f"\r  [silence {'░' * pct:<20}]", end="", flush=True)
                    if silence_chunks >= required_silence:
                        break

                else:
                    dots = (dots + 1) % 4
                    print(f"\r  Waiting for speech{'.' * dots}   ", end="", flush=True)

    except KeyboardInterrupt:
        print()
        return None
    except Exception as e:
        print(f"\n  [audio error]: {e}")
        return None

    print()

    if len(chunks) < min_chunks:
        print("  Too short or quiet — try again.")
        return None

    audio = np.concatenate(chunks, axis=0).flatten()
    print(f"  Captured {len(audio) / SAMPLE_RATE:.1f}s of audio.")
    return audio


def play_audio(samples: np.ndarray, sample_rate: int) -> bool:
    """Play a numpy array as audio. Returns True if interrupted by Esc."""
    import time
    sd.play(samples, samplerate=sample_rate)
    try:
        while sd.get_stream().active:
            if msvcrt.kbhit():
                if msvcrt.getch() == b'\x1b':
                    sd.stop()
                    return True
            time.sleep(0.05)
    except Exception:
        pass
    sd.wait()
    return False
