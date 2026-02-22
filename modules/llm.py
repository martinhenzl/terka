"""
LLM module — language model via Ollama.
Maintains conversation history and sends messages with Terka's personality.
"""

from __future__ import annotations
import re
import ollama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT, MAX_HISTORY, LLM_OPTIONS

_history: list[dict[str, str]] = []
_client: ollama.Client | None = None

# Sentence boundary: .!?~… followed by whitespace + capital letter (or opening bracket/quote).
# ~ = playful tone marker; … = pause/ellipsis — both valid sentence endings for Terka.
# Note: "Hm..." also splits correctly — the last dot in "..." is followed by space + capital.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?~…])\s+(?=[A-Z"\[])')

# Characters that count as a complete sentence ending
_SENTENCE_END = frozenset('.!?…~')


def _ensure_complete(text: str) -> str:
    """
    Trim trailing incomplete sentence — safeguard for num_predict ceiling hits.
    If the text ends with a sentence-ending character it's already complete.
    Otherwise cut at the last sentence boundary found.
    If no boundary exists, return the text as-is (short phrase or single word).
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] in _SENTENCE_END:
        return text
    m = re.search(r'[.!?…~](?!.*[.!?…~])', text)  # last sentence-ending punctuation
    return text[:m.end()].strip() if m else text


def _get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_BASE_URL)
    return _client


def check_connection() -> bool:
    """Verify that Ollama is running and the model is available."""
    try:
        client = _get_client()
        models = client.list()
        available = [m.model for m in models.models]
        found = any(OLLAMA_MODEL in m or m.startswith(OLLAMA_MODEL.split(":")[0]) for m in available)
        if not found:
            print(f"  [WARNING] Model '{OLLAMA_MODEL}' not found.")
            print(f"  Available models: {', '.join(available) or 'none'}")
            print(f"  Run: ollama pull {OLLAMA_MODEL}")
            return False
        print(f"  Ollama OK — model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def chat_sentences(user_text: str, stop_check=None):
    """
    Generator — yields complete sentences as the LLM generates them.
    The first sentence includes the [emotion] tag if the LLM added one.
    History is updated with the full response when the generator finishes or is closed.
    """
    global _history

    _history.append({"role": "user", "content": user_text})
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

    all_parts: list[str] = []
    buffer = ""

    try:
        for chunk in stream:
            if stop_check and stop_check():
                break
            token = chunk.message.content or ""
            all_parts.append(token)
            buffer += token

            m = _SENTENCE_SPLIT.search(buffer)
            if m:
                sentence = buffer[:m.start() + 1].strip()
                buffer = buffer[m.end():]
                if sentence:
                    yield sentence

        # Yield any remaining text (last sentence, or everything if single-sentence response)
        remaining = _ensure_complete(buffer.strip())
        if remaining:
            yield remaining

    finally:
        # Save full response to history (or pop user message if nothing was generated)
        full_reply = _ensure_complete("".join(all_parts).strip())
        if full_reply:
            _history.append({"role": "assistant", "content": full_reply})
        else:
            _history.pop()


def chat(user_text: str, stop_check=None) -> str:
    """
    Send a user message to the model and return the full response.
    Wrapper around chat_sentences() for callers that want a single string.
    """
    return " ".join(chat_sentences(user_text, stop_check=stop_check))


def reset_history() -> None:
    """Clear conversation history."""
    global _history
    _history = []
    print("  Conversation history cleared.")


def get_history_length() -> int:
    """Vrátí počet zpráv v historii."""
    return len(_history)
