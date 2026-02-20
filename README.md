# Terka — AI hlasová asistentka

Open-source AI chatbot s hlasovým vstupem a výstupem, chaotickou osobností a dívčím hlasem.

## Co to umí

- **Poslouchá** tvůj hlas přes mikrofon (automatická detekce ticha)
- **Přepíše** řeč na text (Whisper od OpenAI)
- **Vygeneruje** odpověď přes lokální LLM (Ollama)
- **Odpoví** hlasem s dívčím přízvukem (kokoro-onnx)

Vše běží **lokálně** — žádné API klíče, žádný cloud.

---

## Požadavky

- Python 3.10+
- [Ollama](https://ollama.com/download) (nainstaluj před setupem)
- Mikrofon + reproduktory/sluchátka

---

## Instalace

```bash
# 1. Nainstaluj Ollama z https://ollama.com/download

# 2. Spusť setup (stáhne závislosti + TTS modely + LLM):
python setup.py

# 3. Spusť Terku:
python main.py
```

---

## Použití

| Akce | Co udělat |
|------|-----------|
| Mluvit | Stiskni Enter, mluv, po tichu se automaticky zastaví |
| Psát | Napiš text a stiskni Enter |
| Smazat historii | Napiš `reset` |
| Změnit mikrofon | Napiš `zařízení` |
| Ukončit | Napiš `quit` nebo `exit` |

---

## Nastavení (`config.py`)

| Proměnná | Výchozí | Popis |
|----------|---------|-------|
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model (rychlý) — zkus `mistral:7b` pro lepší kvalitu |
| `WHISPER_MODEL_SIZE` | `base` | Velikost Whisper modelu |
| `WHISPER_LANGUAGE` | `None` | `None` = autodetekce, `"cs"` = čeština, `"en"` = angličtina |
| `TTS_VOICE` | `af_bella` | Hlas — viz níže |
| `SILENCE_THRESHOLD` | `0.015` | Citlivost mikrofonu — sniž pro tichý mikrofon |

### Dostupné ženské hlasy

| Hlas | Popis |
|------|-------|
| `af_bella` | Výchozí — přirozený, energický |
| `af_sarah` | Jemnější, klidnější |
| `af_nicole` | Mladistvý, hravý |
| `af_sky` | Lehký, svěží |
| `bf_emma` | Britský přízvuk |

### Doporučené LLM modely

```bash
ollama pull llama3.2:3b    # rychlý, výchozí
ollama pull mistral:7b     # lepší kvalita
ollama pull llama3.1:8b    # nejlepší, ~8 GB RAM
```

---

## Struktura projektu

```
C:\Terka\
├── main.py          # hlavní smyčka
├── config.py        # veškerá nastavení
├── setup.py         # první spuštění
├── requirements.txt
├── modules/
│   ├── audio.py     # záznam z mikrofonu
│   ├── stt.py       # speech-to-text (Whisper)
│   ├── llm.py       # jazykový model (Ollama)
│   └── tts.py       # text-to-speech (kokoro)
└── models/
    ├── kokoro-v0_19.onnx   # TTS model (stáhne setup.py)
    └── voices.bin           # hlasové embeddingy
```

---

## Technologie

| Komponenta | Technologie | Licence |
|------------|------------|---------|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| LLM | [Ollama](https://ollama.com) + llama3.2 | MIT / Meta Llama |
| TTS | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | MIT |
| Audio | [sounddevice](https://github.com/spatialaudio/python-sounddevice) | MIT |

---

## Řešení problémů

**Ollama není dostupná:**
```bash
ollama serve   # spusť v samostatném terminálu
```

**Mikrofon nefunguje:**
- Napiš `zařízení` v chatu pro seznam mikrofonů
- Uprav `SILENCE_THRESHOLD` v `config.py`

**Whisper se stahuje dlouho:**
- Modely se stahují jen jednou a ukládají se do cache (`~/.cache/huggingface/`)

**TTS model chybí:**
```bash
python setup.py   # spusť znovu
```
