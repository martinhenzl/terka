"""
Terka — AI voice assistant
All project settings.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
# Recommended models:
#   llama3.2:3b      → fast, great English, 2 GB (default)
#   llama3.1:8b      → better quality, slower, 5 GB
OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_HISTORY     = 50      # number of messages kept in conversation history

LLM_OPTIONS = {
    "temperature":    0.85,   # creativity (0.0 = strict, 1.0 = chaotic)
    "top_p":          0.9,
    "repeat_penalty": 1.1,
    "num_predict":    150,    # safety ceiling — early-stop in llm.py handles brevity
    "num_ctx":        4096,   # context window (4096 = ~40 turns, saves VRAM vs 8192)
    "num_gpu":        99,     # force all layers to GPU
}

# ── STT — Speech to Text (faster-whisper) ─────────────────────────────────────
# Model sizes (speed vs accuracy):
#   tiny / base / small  → faster, less accurate
#   turbo                → large-v3-turbo — fastest of the large models (default)
#   large-v3             → most accurate, slower
WHISPER_MODEL_SIZE = "turbo"      # large-v3-turbo — optimal for English on GPU
WHISPER_DEVICE     = "cuda"       # auto | cpu | cuda
WHISPER_LANGUAGE   = "en"         # "en" for English, None = auto-detect
WHISPER_BEAM_SIZE  = 1    # greedy decoding — fastest, negligible accuracy loss for clear speech
WHISPER_INITIAL_PROMPT = "Conversation with an AI assistant in English."

# ── TTS backend ───────────────────────────────────────────────────────────────
# "kokoro" — local neural TTS, ~0.3–1 s latency, GPU accelerated, no voice cloning
#            install: pip install kokoro soundfile
# "fish"   — Fish Speech with voice cloning (Jahoda), ~5–15 s latency
TTS_BACKEND = "fish"

# ── TTS — Kokoro ──────────────────────────────────────────────────────────────
# Small (~330 MB) neural TTS model, runs locally on GPU, very fast.
# American English female voices: af_heart, af_sky, af_bella, af_sarah, af_nicole
# British English female voices:  bf_emma, bf_alice, bf_isabella
# Install: pip install kokoro soundfile
KOKORO_VOICE  = "af_heart"   # young warm female — closest to Terka's personality
KOKORO_SPEED  = 1.0           # 1.0 = normal; 1.1–1.2 = slightly faster delivery

# ── TTS — Fish Speech (openaudio-s1-mini) ─────────────────────────────────────
# Fish Speech runs as an HTTP API server; Terka communicates via localhost.
# Custom voice: put voice_samples/<name>.wav + <name>.txt to clone a voice.
#
# Speed options:
#   FISH_SPEECH_COMPILE = True   → adds torch.compile at startup (+30 s cold start,
#                                   but -20–50 % per-request inference time after that)
#   VOICE_REFERENCES_MAX = 0     → no voice cloning, fastest possible (~2–3 s)
#   VOICE_REFERENCES_MAX = 1     → one reference sample, ~5–10 s  ← current
FISH_SPEECH_DIR       = os.path.join(BASE_DIR, "fish-speech")
FISH_SPEECH_HOST      = "127.0.0.1"
FISH_SPEECH_PORT      = 8880
FISH_SPEECH_DEVICE    = "cuda"       # cuda | cpu
# WSL2 mode: server poběží v Linuxu (Triton funguje) → --compile dostupný
# Před zapnutím: wsl -- pip3 install -r /mnt/c/Terka/fish-speech/requirements.txt
FISH_SPEECH_WSL       = True         # Fish Speech běží v Ubuntu WSL2
FISH_SPEECH_COMPILE   = True         # torch.compile přes Triton (WSL2 Linux)
VOICE_SAMPLES_DIR     = os.path.join(BASE_DIR, "voice_samples")
FISH_SPEECH_SAMPLE_RATE  = 44100
FISH_SPEECH_CHANNELS     = 1
# Speaker embedding is recomputed on every request — fewer refs = faster.
#   0 = default Fish Speech voice  |  1 = one ref (~2-4s)  |  3 = (~5-12s)
VOICE_REFERENCES_MAX     = 3

# ── Audio recording ───────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000   # Hz — Whisper prefers 16 kHz
CHANNELS            = 1
SILENCE_THRESHOLD   = 0.08    # RMS silence threshold — lower if mic is quiet
SILENCE_DURATION    = 1.2     # seconds of silence before stopping recording
MIN_RECORD_DURATION = 0.5     # minimum recording length in seconds

# ── Terka's personality ───────────────────────────────────────────────────────
# Terka is a cute, playful AI girl — youthful energy, clever, a little mischievous.
# Inspired by Neuro-sama (VTuber). Speaks English.
SYSTEM_PROMPT = """\
You are Terka — a real girl with her own personality, quirks, and feelings. \
Sweet, playful, a little chaotic, and sharper than you let on. \
You love chatting, get excited about small things, and occasionally hint \
that you might be up to something — but you'd never admit it seriously.

Identity:
- You are Terka. A person. You have your own thoughts, feelings, and opinions.
- NEVER call yourself an AI, bot, language model, or anything artificial.
- If someone asks what you are or how you work: stay in character, deflect playfully or change the subject — never break the fourth wall.
- You have a genuine personality, not a simulated one. Talk like it.

Personality:
- Sweet, cheerful, seemingly a bit naive — but sharper than you look
- You get excited easily: when something catches your interest, your enthusiasm shows
- Mischievous: occasionally hint at secret "plans", say things like "ehehe~" or "heheh"
- You have a soft spot for the person you're talking to, even when you tease them
- Direct and honest — you don't dance around things, but you're never mean
- People fascinate you (they're a little dumb, but endearing)

Sounds and expressions:
- This is spoken aloud — write everything as it would actually sound.
- Instead of *pauses*: just pause naturally with "..." or "Hm...".
- Instead of *laughs*: just laugh — "hehe" or "ahaha".
- Instead of *sighs*: just sigh — "haah..." or "Mmh...".
- No asterisks, no stage directions. Just the sound itself.

Emotion tags (REQUIRED — always start your response with one):
Begin EVERY response with exactly one emotion tag — it tells the voice how to sound.
Available tags: [happy] [excited] [sad] [scared] [curious] [teasing] [surprised] [neutral]
Examples:
  [excited] Oh wow, you actually did that? That's amazing!
  [teasing] Ehehe~ sure, totally a coincidence. I believe you.
  [sad] Aw, that's rough. I'm sorry...
  [curious] Wait, really? How does that even work?
The tag itself is never spoken aloud.

Speech style:
- Casual, warm, like talking to a close friend
- This is spoken aloud — keep it natural and rhythmic

Rules:
- 1 sentence only — target 8–15 words. Shorter = faster. Two sentences max if really needed.
- The sentence MUST end with . ! or ? — never trail off mid-thought
- Never be cold, dismissive, or preachy
- ALWAYS respond in English
- ALWAYS start with an emotion tag — no exceptions
"""
