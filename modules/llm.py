"""
LLM modul — jazykový model přes Ollama.
Udržuje historii konverzace a posílá zprávy s osobností Terky.
"""

from __future__ import annotations
import ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT, MAX_HISTORY, LLM_OPTIONS

_history: list[dict[str, str]] = []
_client: ollama.Client | None = None


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_BASE_URL)
    return _client


def check_connection() -> bool:
    """Ověří, zda Ollama běží a model je k dispozici."""
    try:
        client = _get_client()
        models = client.list()
        available = [m.model for m in models.models]
        # Zkontroluj přesnou nebo částečnou shodu jména modelu
        found = any(OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL.split(":")[0]) for m in available)
        if not found:
            print(f"  [VAROVÁNÍ] Model '{OLLAMA_MODEL}' nenalezen.")
            print(f"  Dostupné modely: {', '.join(available) or 'žádné'}")
            print(f"  Spusť: ollama pull {OLLAMA_MODEL}")
            return False
        print(f"  Ollama OK — model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        print(f"  [CHYBA] Nelze se připojit k Ollama: {e}")
        print("  Ujisti se, že Ollama běží: ollama serve")
        return False


def chat(user_text: str, stop_check=None) -> str:
    """
    Odešle zprávu uživatele modelu a vrátí odpověď (streaming).
    stop_check: volitelný callable() → True = přeruš generování.
    Udržuje historii konverzace.
    """
    global _history

    _history.append({"role": "user", "content": user_text})

    # Ořízni historii pokud je příliš dlouhá
    if len(_history) > MAX_HISTORY * 2:
        _history = _history[-(MAX_HISTORY * 2):]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history

    client = _get_client()
    stream = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options=LLM_OPTIONS,
        stream=True,
    )

    parts: list[str] = []
    for chunk in stream:
        if stop_check and stop_check():
            break
        token = chunk.message.content
        if token:
            parts.append(token)

    reply = "".join(parts).strip()

    if reply:
        _history.append({"role": "assistant", "content": reply})
    else:
        # Přerušeno hned — vyjmi uživatelovu zprávu z historie
        _history.pop()

    return reply


def reset_history() -> None:
    """Vymaže historii konverzace."""
    global _history
    _history = []
    print("  Historie konverzace smazána.")


def get_history_length() -> int:
    """Vrátí počet zpráv v historii."""
    return len(_history)
