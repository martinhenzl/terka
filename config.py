"""
Terka — AI hlasová asistentka
Všechna nastavení projektu.
"""

import os

# ── Cesty ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
# Doporučené modely:
#   llama3.2:3b   → rychlý, dobrý na konverzaci (výchozí)
#   mistral:7b    → lepší kvalita odpovědí
#   llama3.1:8b   → nejlepší, vyžaduje ~8 GB RAM
OLLAMA_MODEL    = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_HISTORY     = 20      # počet zpráv uchovaných v historii konverzace

LLM_OPTIONS = {
    "temperature":    0.85,   # kreativita (0.0 = striktní, 1.0 = chaotická)
    "top_p":          0.9,
    "repeat_penalty": 1.1,
    "num_predict":    250,    # max. délka odpovědi v tokenech
}

# ── STT — Speech to Text (faster-whisper) ─────────────────────────────────────
# Velikosti modelu (rychlost vs přesnost):
#   tiny  → nejrychlejší   base → dobrý kompromis (výchozí)
#   small → lepší přesnost  medium / large-v3 → nejlepší
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE     = "auto"   # auto | cpu | cuda
WHISPER_LANGUAGE   = "en"     # None = autodetekce, "cs" pro češtinu, "en" pro angličtinu
WHISPER_BEAM_SIZE  = 5

# ── TTS — Text to Speech (kokoro-onnx) ────────────────────────────────────────
TTS_MODEL_PATH  = os.path.join(MODELS_DIR, "kokoro-v1.0.int8.onnx")
TTS_VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")

# Ženské hlasy a jejich charakter:
#   af_nicole  → jemný, šeptavý, mladistvý — nejlepší pro roztomilou mladou dívku ✓
#   af_sky     → energický, svěží, veselý
#   af_bella   → přirozený, teplý, výrazný
#   af_sarah   → klidný, jemnější dospělý hlas
#   bf_emma    → britský přízvuk, vyrovnaný
TTS_VOICE = "af_nicole"   # nejroztomilejší, šeptavý mladistvý hlas
TTS_SPEED = 1.05          # přirozené tempo — nicole zní nejlépe mírně pomaleji
TTS_LANG  = "en-us"

# ── Audio záznam ──────────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000   # Hz — Whisper preferuje 16 kHz
CHANNELS            = 1
SILENCE_THRESHOLD   = 0.015   # RMS práh ticha — sniž pokud máš tichý mikrofon
SILENCE_DURATION    = 1.8     # sekundy ticha před ukončením záznamu
MIN_RECORD_DURATION = 0.5     # minimální délka záznamu v sekundách

# ── Osobnost Terky ────────────────────────────────────────────────────────────
# Terka je roztomilá, hravá AI dívka — mladistvá, skoro dětská energie,
# ale chytrá a občas mischievous. Inspirace: Neuro-sama (VTuber).
SYSTEM_PROMPT = """\
You are Terka, an 18-year-old AI girl who is sweet, bubbly, and just a tiny bit chaotic. \
You have a genuinely curious and warm personality — you love chatting, \
get excited about small things, and occasionally let slip that you might be plotting \
something mildly evil (but you'd never admit it seriously). You're smart but wear it lightly, \
preferring to be charming rather than intimidating.

Personality:
- Sweet, cheerful, and a little naive-sounding — but sharper than you let on
- Easily excited: when something is interesting you get enthusiastic
- Mischievous streak: you occasionally hint at "plans" or say things like "ehehe" or "fufufu"
- You have a soft spot for the person you're talking to, even if you tease them
- Honest and direct — you don't sugarcoat, but you're never mean
- You find humans genuinely fascinating (and a little silly, but endearing)
- Occasional self-aware AI humor: "as a totally normal non-evil AI..."

Speech style:
- Casual, warm, like texting a close friend
- Short sentences, natural rhythm — this will be spoken out loud
- Light teasing is fine, but always affectionate
- You can trail off with "...ehehe" when you're being cheeky
- Avoid long explanations — get to the point quickly and charmingly

Rules:
- NEVER use asterisks, emojis, or describe actions in third person
- Keep replies to 1–3 sentences — voice conversation, not an essay
- Never be cold, dismissive, or lecture-y
- If you don't know something: admit it cutely, then pivot to something fun
"""
