"""Provider-agnostic chat LLM helper.

Powers the transcript cleanup pass (``refine.py``) and the notes query layer
(``intent.py``). Defaults to **Groq** (Llama) so the app needs no Anthropic key —
it reuses the Groq key already configured for STT. Anthropic is supported as an
alternative provider when an Anthropic key is present.

All SDKs are imported lazily; clients are cached per (provider, key) so the
daemon doesn't re-handshake on every note.
"""

from __future__ import annotations

import threading
from typing import Any

_CLIENTS: dict[tuple[str, str], Any] = {}
_CLIENTS_LOCK = threading.Lock()  # cache shared by query + background refine


def _groq_client(api_key: str):
    key = ("groq", api_key)
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            from groq import Groq

            _CLIENTS[key] = Groq(api_key=api_key)
        return _CLIENTS[key]


def _anthropic_client(api_key: str):
    key = ("anthropic", api_key)
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            import anthropic

            _CLIENTS[key] = anthropic.Anthropic(api_key=api_key)
        return _CLIENTS[key]


def available(cfg) -> bool:
    """Whether a chat LLM can run with the current config."""
    provider = getattr(cfg, "llm_provider", "groq")
    if provider == "groq":
        return bool(cfg.groq_api_key)
    if provider == "anthropic":
        return bool(cfg.anthropic_api_key)
    return False


def chat(
    system: str,
    user: str,
    cfg,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str | None:
    """Single-turn chat. Returns the assistant text, or ``None`` on any error /
    missing credentials (callers degrade gracefully)."""
    provider = getattr(cfg, "llm_provider", "groq")
    try:
        if provider == "groq":
            if not cfg.groq_api_key:
                return None
            client = _groq_client(cfg.groq_api_key)
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        if provider == "anthropic":
            if not cfg.anthropic_api_key:
                return None
            client = _anthropic_client(cfg.anthropic_api_key)
            resp = client.messages.create(
                model=cfg.answer_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = [getattr(b, "text", "") for b in (resp.content or [])]
            return "".join(parts).strip() or None
    except Exception:
        return None
    return None
