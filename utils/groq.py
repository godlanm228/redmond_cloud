"""Groq chat completions — the one low-level client (like utils/gemini).

Before Sep 29, 2026 four modules talked to Groq on their own: the response
generator (through the SDK), the router (SDK), vision and speech (requests).
None of them read the rate-limit headers Groq sends on every reply, so the
limits were only discovered by hitting them. Here every reply, good or bad,
is reported to utils.llm_gate.

Errors are returned as values, not raised: the generator is shared by four
bots running in threads, and an exception-as-control-flow per model in a
fallback chain is how errors used to overwrite each other.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils import failures, llm_gate

logger = logging.getLogger(__name__)

API_BASE = "https://api.groq.com"
TIMEOUT_SEC = 40.0
# Groq answers 403 to the default urllib/requests user agent (Sep 28, 2026).
_UA = "redmond-hub/1.0"


def api_key_from_env() -> str:
    return os.getenv("REDMOND_GROQ_API_KEY", "")


def model_options(model: str) -> Dict[str, Any]:
    """Per-family request options.

    qwen3 is a reasoning model: without reasoning_effort=none it thinks in a
    <think> block and burns max_tokens on it. gpt-oss always reasons and its
    reasoning counts against max_tokens; short service calls (router,
    translation) pass reasoning_effort=low themselves through `extra`."""
    if "qwen" in model:
        return {"reasoning_effort": "none"}
    return {}


def chat(model: str, messages: List[Dict[str, Any]], *, tools: Optional[list] = None,
         tool_choice: Any = None, temperature: float = 0.5, max_tokens: int = 800,
         api_key: str = "", base_url: str = "", timeout: float = TIMEOUT_SEC,
         extra: Optional[Dict[str, Any]] = None) -> Tuple[Optional[dict], str]:
    """(raw JSON reply | None, error text). Reports the outcome to llm_gate."""
    key = api_key or api_key_from_env()
    if not key:
        return None, "no Groq API key"
    payload: Dict[str, Any] = {"model": model, "messages": messages,
                               "temperature": temperature, "max_tokens": max_tokens}
    payload.update(model_options(model))
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if extra:
        payload.update(extra)
    url = f"{(base_url or API_BASE).rstrip('/')}/openai/v1/chat/completions"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "User-Agent": _UA},
                             json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — network, timeout
        err = f"{e.__class__.__name__}: {e}"
        llm_gate.report(model, None)
        failures.remember(f"Groq {model}", err)
        return None, err

    llm_gate.report(model, resp.status_code, dict(resp.headers),
                    None if resp.status_code == 200 else _body(resp))
    if resp.status_code == 200:
        try:
            return resp.json(), ""
        except ValueError as e:
            return None, f"bad JSON from Groq: {e}"
    err = f"HTTP {resp.status_code}: {resp.text[:600]}"
    failures.remember(f"Groq {model}", err)
    return None, err


def _body(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


def text_of(completion: Optional[dict]) -> str:
    """Assistant text of a completion ('' if none)."""
    try:
        return (completion["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ""
