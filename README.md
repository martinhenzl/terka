# Terka — AI Voice Assistant

Open-source AI voice chatbot with a custom cloned voice, playful personality, and fully local stack. No API keys, no cloud.

- **Listens** via microphone (auto silence detection)
- **Transcribes** speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper turbo)
- **Replies** via local LLM ([Ollama](https://ollama.com))
- **Speaks** with a cloned voice via [Fish Speech](https://github.com/fishaudio/fish-speech)

---

## Requirements

| Requirement | Notes |
|---|---|
| Windows 10/11 | Tested on Windows 10 |
| Python 3.10+ | [python.org](https://www.python.org/downloads/) |
| NVIDIA GPU | CUDA required — Whisper + Fish Speech both run on GPU |
| [Ollama](https://ollama.com/download) | Local LLM server |
| [Git](https://git-scm.com/) | For cloning Fish Speech |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Fish Speech package manager |
| WSL2 + Ubuntu *(optional)* | Enables `--compile` for ~30–50% faster TTS |

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/yourname/terka.git
cd terka
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama + pull a model

Download and install [Ollama](https://ollama.com/download), then pull a model:

```bash
ollama pull llama3.2:3b       # fast, default (~2 GB)
ollama pull llama3.1:8b       # better quality (~5 GB)
```

Ollama must be running in the background when you start Terka (`ollama serve`).

### 4. Install Fish Speech

```bash
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
uv sync --extra cu128          # CUDA 12.x — use cu126 for older drivers
cd ..
```

Then download the Fish Speech checkpoint (openaudio-s1-mini):

```bash
cd fish-speech
python -m tools.download_models --model openaudio-s1-mini
cd ..
```

> Alternatively, download manually from [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini)
> and place it into `fish-speech/checkpoints/openaudio-s1-mini/`.

### 5. Add voice samples *(optional — for voice cloning)*

Put pairs of `.wav` + `.txt` files in `voice_samples/`:

```
voice_samples/
├── speaker_00.wav    # audio clip (5–15 seconds, clean speech)
├── speaker_00.txt    # exact transcript of that clip
├── speaker_01.wav
├── speaker_01.txt
└── ...
```

The `.txt` must contain the exact words spoken in the `.wav`. Terka auto-discovers all pairs on startup. If no samples are provided, Fish Speech uses its default voice.

### 6. Configure `config.py`

Open `config.py` and adjust at minimum:

```python
OLLAMA_MODEL = "llama3.2:3b"    # LLM model you pulled in step 3
TTS_BACKEND  = "fish"           # "fish" (default) or "kokoro" (faster, no cloning)
```

Fish Speech path defaults to `<project root>/fish-speech` — change `FISH_SPEECH_DIR` if you cloned it elsewhere.

### 7. Run

```bash
python main.py
```

---

## Optional: WSL2 acceleration for Fish Speech

Fish Speech supports `--compile` (PyTorch/Triton) which gives **30–50% faster** TTS inference, but Triton only works on Linux. You can run the Fish Speech server inside WSL2 while Terka runs on Windows.

**One-time setup:**

```powershell
# Install Ubuntu in WSL2 (run in PowerShell as admin)
wsl --install -d Ubuntu

# Install Fish Speech dependencies inside WSL2
wsl -d Ubuntu -u root -- bash -c "
  apt-get update -qq &&
  apt-get install -y portaudio19-dev &&
  curl -LsSf https://astral.sh/uv/install.sh | sh
"
wsl -d Ubuntu -u root -- bash -c "
  cd /mnt/c/path/to/fish-speech &&
  /root/.local/bin/uv sync --extra cu128
"
```

**Enable in `config.py`:**

```python
FISH_SPEECH_WSL     = True   # start server in Ubuntu WSL2
FISH_SPEECH_COMPILE = True   # enable torch.compile (Triton)
```

First startup will take ~30–60 extra seconds to compile. Subsequent starts are fast.

---

## Usage

| Action | How |
|---|---|
| Speak | Press **Enter**, talk, stop — auto-detects silence |
| Type instead | Just type your message and press Enter |
| Interrupt | Press **Esc** at any time |
| Clear history | Type `reset` |
| List microphones | Type `devices` |
| Quit | Type `quit` or `exit` |

---

## Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model name |
| `WHISPER_MODEL_SIZE` | `turbo` | Whisper model — `tiny`/`base`/`small`/`turbo`/`large-v3` |
| `WHISPER_LANGUAGE` | `"en"` | `"en"`, `"cs"`, or `None` for auto-detect |
| `TTS_BACKEND` | `"fish"` | `"fish"` (voice cloning) or `"kokoro"` (fast, no cloning) |
| `VOICE_REFERENCES_MAX` | `3` | Voice samples used for cloning — `0` = default voice (fastest) |
| `FISH_SPEECH_WSL` | `False` | Run Fish Speech server in WSL2 Ubuntu |
| `FISH_SPEECH_COMPILE` | `False` | Enable `--compile` (requires WSL2) |
| `SILENCE_THRESHOLD` | `0.08` | Mic sensitivity — lower if your mic is quiet |
| `KOKORO_VOICE` | `af_heart` | Voice for Kokoro backend — `af_heart`, `af_sky`, `af_bella`, … |

---

## Project structure

```
terka/
├── main.py              # main loop
├── config.py            # all settings
├── requirements.txt     # Python dependencies
├── modules/
│   ├── audio.py         # microphone recording
│   ├── stt.py           # speech-to-text (Whisper)
│   ├── llm.py           # language model (Ollama)
│   └── tts.py           # text-to-speech (Fish Speech / Kokoro)
├── voice_samples/       # .wav + .txt pairs for voice cloning
├── fish-speech/         # Fish Speech repo (cloned separately, git-ignored)
└── models/              # Kokoro model files (if using Kokoro backend)
```

---

## Tech stack

| Component | Technology | License |
|---|---|---|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper turbo) | MIT |
| LLM | [Ollama](https://ollama.com) + llama3.2 | MIT / Meta Llama |
| TTS | [Fish Speech](https://github.com/fishaudio/fish-speech) (openaudio-s1-mini) | CC BY-NC-SA 4.0 |
| TTS fallback | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | MIT |
| Audio | [sounddevice](https://python-sounddevice.readthedocs.io/) | MIT |

---

## Troubleshooting

**Ollama not available:**
```bash
ollama serve   # run in a separate terminal
```

**Fish Speech server won't start:**
- Check that `fish-speech/` exists in the project folder (or update `FISH_SPEECH_DIR` in config)
- Check that checkpoints are downloaded: `fish-speech/checkpoints/openaudio-s1-mini/`
- Try `TTS_BACKEND = "kokoro"` as a quick fallback

**Mic not working / nothing recorded:**
- Type `devices` in chat to list microphones and select the right one
- Lower `SILENCE_THRESHOLD` in `config.py` (e.g. `0.03`) if mic is quiet

**CUDA out of memory:**
- Use a smaller Whisper model: `WHISPER_MODEL_SIZE = "base"`
- Lower `VOICE_REFERENCES_MAX` to `1` or `0`
- Use `FISH_SPEECH_DEVICE = "cpu"` (slow)
