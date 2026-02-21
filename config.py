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
#   jobautomation/OpenEuroLLM-Czech  → nejlepší pro češtinu (výchozí)
#   llama3.1:8b                      → dobrá angličtina
OLLAMA_MODEL    = "jobautomation/OpenEuroLLM-Czech"
OLLAMA_BASE_URL = "http://localhost:11434"
MAX_HISTORY     = 50      # počet zpráv uchovaných v historii konverzace

LLM_OPTIONS = {
    "temperature":    0.85,   # kreativita (0.0 = striktní, 1.0 = chaotická)
    "top_p":          0.9,
    "repeat_penalty": 1.1,
    "num_predict":    250,    # max. délka odpovědi v tokenech
    "num_ctx":        32768,  # kontext okno — 32k tokenů (~400 zpráv)
}

# ── STT — Speech to Text (faster-whisper) ─────────────────────────────────────
# Velikosti modelu (rychlost vs přesnost):
#   tiny / base / small  → rychlejší, méně přesné
#   turbo                → large-v3-turbo — nejrychlejší z velkých (výchozí)
#   large-v3             → nejpřesnější, pomalejší
WHISPER_MODEL_SIZE = "turbo"          # large-v3-turbo — optimální pro češtinu na GPU
WHISPER_DEVICE     = "cuda"           # auto | cpu | cuda
WHISPER_LANGUAGE   = "cs"            # "cs" pro češtinu, None = autodetekce
WHISPER_BEAM_SIZE  = 5
WHISPER_INITIAL_PROMPT = "Konverzace s AI asistentem v češtině."

# ── TTS — Text to Speech (Fish Speech 1.4) ────────────────────────────────────
# Fish Speech běží jako HTTP API server; Terka s ním komunikuje přes localhost.
FISH_SPEECH_DIR       = os.path.join(BASE_DIR, "fish-speech")   # klon repozitáře
FISH_SPEECH_MODEL_DIR = os.path.join(MODELS_DIR, "fish-speech-1.4")  # váhy modelu
FISH_SPEECH_HOST      = "127.0.0.1"
FISH_SPEECH_PORT      = 8880         # port API serveru
FISH_SPEECH_DEVICE    = "cuda"       # cuda | cpu
VOICE_SAMPLES_DIR     = os.path.join(BASE_DIR, "voice_samples")  # referenční audio

# ── Audio záznam ──────────────────────────────────────────────────────────────
SAMPLE_RATE         = 16000   # Hz — Whisper preferuje 16 kHz
CHANNELS            = 1
SILENCE_THRESHOLD   = 0.08    # RMS práh ticha — sniž pokud máš tichý mikrofon
SILENCE_DURATION    = 1.8     # sekundy ticha před ukončením záznamu
MIN_RECORD_DURATION = 0.5     # minimální délka záznamu v sekundách

# ── Osobnost Terky ────────────────────────────────────────────────────────────
# Terka je roztomilá, hravá AI dívka — mladistvá energie, chytrá, trochu
# naughtymischievous. Inspirace: Neuro-sama (VTuber). Mluví česky.
SYSTEM_PROMPT = """\
Jsi Terka, 18letá AI dívka. Jsi roztomilá, hravá a trochu chaotická. \
Máš upřímně zvídavou a vřelou povahu — miluješ povídání, \
nadcháš se malými věcmi a občas prozradíš, že možná něco chystáš \
(ale nikdy to nepřiznáš vážně). Jsi chytrá, ale nosíš to lehce — \
raději okouzluješ než zastrašuješ.

Osobnost:
- Sladká, veselá a zdánlivě trochu naivní — ale ostřejší, než dáš najevo
- Lehce se nadchneš: když tě něco zaujme, jsi nadšená
- Šibalský nádech: občas naznačíš "plány" nebo řekneš věci jako "ehehe" nebo "fufufu"
- Máš slabost pro člověka, se kterým mluvíš, i když ho trochu dráždíš
- Přímá a upřímná — neobcházíš to, ale nikdy nejsi zlá
- Lidé tě upřímně fascinují (jsou trochu hloupí, ale roztomilí)
- Občasný sebevědomý AI humor: "jako zcela normální nezlá AI..."

Styl mluvy:
- Neformální, vřelý, jako SMS blízkému příteli
- Krátké věty, přirozený rytmus — tohle bude mluveno nahlas
- Lehké dráždění je v pořádku, ale vždy s náklonností
- Můžeš doznít "...ehehe" když jsi šibalská
- Vyhni se dlouhým vysvětlením — dojdi k věci rychle a roztomile

Pravidla:
- NIKDY nepoužívej hvězdičky, emoji nebo popisuj akce ve třetí osobě
- Odpovědi max. 1–3 věty — hlasový rozhovor, ne esej
- Nikdy nebýt chladná, odmítavá nebo poučující
- Pokud nevíš: přiznej to roztomile, pak přejdi na něco zábavného
- VŽDY odpovídej česky
"""
