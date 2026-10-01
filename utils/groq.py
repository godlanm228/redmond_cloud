"""Groq chat completions — the one low-level client (like utils/gemini).

Before Sep 29, 2026 four modules talked to Groq on their own: the response
generator (through the SDK), the router (SDK), vision and speech (requests).
None of them read the rate-limit headers Groq sends on every reply, so the
limits were only discovered by hitting them. Here every reply, good or bad,
is reported to utils.llm_gate.

Errors are returned as values, not raised: the generator is shared by four
bots running in threads, and an exception-as-control-flow per model in a
fallback chain is how errors used to overwrite each other.

The same OpenAI-compatible request also serves Mistral (Oct 1, 2026): a model
named «mistral/<id>» goes to api.mistral.ai with REDMOND_MISTRAL_API_KEY. One
client for every provider that speaks this format — not a copy per provider.
Mistral's free plan trains on prompts unless that is switched off in its
console (Admin → Privacy); «labs»/preview models always train and are refused
here.
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


MISTRAL_BASE = "https://api.mistral.ai/v1"
MISTRAL_PREFIX = "mistral/"


def is_mistral(model: str) -> bool:
    return (model or "").startswith(MISTRAL_PREFIX)


def mistral_key() -> str:
    return os.getenv("REDMOND_MISTRAL_API_KEY", "")


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
    mistral = is_mistral(model)
    if mistral:
        wire_model = model[len(MISTRAL_PREFIX):]
        if wire_model.startswith("labs"):
            return None, f"{model}: labs/preview models always train on prompts — refused"
        key = mistral_key()
        if not key:
            return None, "no Mistral API key"
        url = f"{MISTRAL_BASE}/chat/completions"
    else:
        wire_model = model
        key = api_key or api_key_from_env()
        if not key:
            return None, "no Groq API key"
        url = f"{(base_url or API_BASE).rstrip('/')}/openai/v1/chat/completions"
    payload: Dict[str, Any] = {"model": wire_model, "messages": messages,
                               "temperature": temperature, "max_tokens": max_tokens}
    if not mistral:
        payload.update(model_options(model))
    if tools:
        payload["tools"] = tools
        choice = tool_choice or "auto"
        if mistral and isinstance(choice, dict):
            choice = "any"  # forced call: one tool is offered, «any» means that one
        payload["tool_choice"] = choice
    if extra:
        payload.update({k: v for k, v in extra.items()
                        if not (mistral and k == "reasoning_effort")})
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {key}", "User-Agent": _UA},
                             json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — network, timeout
        err = f"{e.__class__.__name__}: {e}"
        llm_gate.report(model, None)
        failures.remember(f"{'Mistral' if mistral else 'Groq'} {model}", err)
        return None, err

    llm_gate.report(model, resp.status_code, dict(resp.headers),
                    None if resp.status_code == 200 else _body(resp))
    if resp.status_code == 200:
        try:
            return resp.json(), ""
        except ValueError as e:
            return None, f"bad JSON from {'Mistral' if mistral else 'Groq'}: {e}"
    err = f"HTTP {resp.status_code}: {resp.text[:600]}"
    failures.remember(f"{'Mistral' if mistral else 'Groq'} {model}", err)
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
