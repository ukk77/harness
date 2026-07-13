"""Thin LLM client for A2 run summaries.

Supports two providers:
  - ``ollama``  — local Ollama server (default, no API key required)
  - ``openai``  — OpenAI chat completions (requires OPENAI_API_KEY in env)

All public functions return ``None`` on any failure and never raise, so the
caller can always treat ``None`` as "summary unavailable" and continue normally.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120  # seconds


def call_llm(
    prompt: str,
    provider: str = "ollama",
    model: str = "llama3.2:3b",
    base_url: str = "http://localhost:11434",
) -> Optional[str]:
    """Call the configured LLM and return the response text.

    Parameters
    ----------
    prompt:
        The full user prompt to send.
    provider:
        ``"ollama"`` or ``"openai"``.
    model:
        Model name (e.g. ``"llama3.2:3b"`` for Ollama, ``"gpt-4o-mini"`` for OpenAI).
    base_url:
        Ollama server base URL.  Ignored for the ``openai`` provider.

    Returns
    -------
    str | None
        Response text stripped of leading/trailing whitespace, or ``None`` on
        any failure (connection error, timeout, API error, missing package,
        unknown provider).
    """
    try:
        if provider == "ollama":
            return _call_ollama(prompt, model, base_url)
        if provider == "openai":
            return _call_openai(prompt, model)
        log.warning(
            "[llm_client] Unknown provider '%s' — supported: ollama, openai", provider
        )
        return None
    except Exception as exc:
        log.warning("[llm_client] LLM call failed (%s / %s): %s", provider, model, exc)
        return None


# ── Provider implementations ──────────────────────────────────────────────────


def _call_ollama(prompt: str, model: str, base_url: str) -> Optional[str]:
    """POST to Ollama /api/chat — returns response text or None."""
    import requests  # standard in the platform (already a dependency)

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def _call_openai(prompt: str, model: str) -> Optional[str]:
    """Call OpenAI chat completions — returns response text or None."""
    try:
        import openai
    except ImportError:
        log.warning(
            "[llm_client] 'openai' package not installed — run: pip install openai"
        )
        return None

    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()
