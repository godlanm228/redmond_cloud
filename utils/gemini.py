"""
Gemini API (generativelanguage.googleapis.com) — единый низкоуровневый клиент.

Ключ REDMOND_GEMINI_API_KEY = free-tier проект RedmondFree (~10 RPM / 250 RPD
на gemini-2.5-flash) — для наших объёмов с запасом. ВАЖНО: проект должен быть
БЕЗ биллинга — проект с prepay-биллингом при нуле кредитов отдаёт 429 без
отката на free tier (так «не работал» старый ключ).

Используется для:
  • vision-разбора фото (logic/vision.py) — качество сильно выше Groq scout
  • web-поиска через Google Search grounding (logic/tools.py) — настоящий
    Google вместо мёртвого CSE (403) и DDG-костылей
  • перевода заголовков дайджеста (logic/digest.py)
  • fallback-генерации когда Groq исчерпал дневной TPD (response_generator)

thinkingBudget=0 во всех вызовах: 2.5-flash по умолчанию «думает» и молча
сжигает выходные токены на рассуждения — для наших коротких задач это вред.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


def api_key_from_env() -> str:
    return os.getenv("REDMOND_GEMINI_API_KEY", "")


def generate(
    parts: List[Dict[str, Any]],
    *,
    model: str = "",
    system: str = "",
    tools: Optional[List[dict]] = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout: float = 45.0,
    api_key: str = "",
) -> Optional[dict]:
    """Сырой generateContent. parts — список частей ({'text': …} и/или
    {'inline_data': {'mime_type': …, 'data': b64}}). None при любой ошибке."""
    key = api_key or api_key_from_env()
    if not key:
        return None

    body: Dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools

    try:
        resp = requests.post(
            f"{_API_BASE}/models/{model or DEFAULT_MODEL}:generateContent",
            params={"key": key},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return None


def extract_text(data: Optional[dict]) -> str:
    """Склеенный текст из ответа generateContent ('' если его нет)."""
    if not data:
        return ""
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    return "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()


def generate_text(
    prompt: str,
    *,
    system: str = "",
    model: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1024,
    api_key: str = "",
) -> str:
    """Текст-в-текст. '' при ошибке."""
    return extract_text(generate(
        [{"text": prompt}],
        model=model, system=system,
        temperature=temperature, max_tokens=max_tokens, api_key=api_key,
    ))


def grounded_search(
    query: str,
    *,
    max_tokens: int = 800,
    api_key: str = "",
) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """
    Поиск через Google Search grounding: модель сама гуглит и отвечает фактурой.
    Возвращает (answer_text, [(source_title, url), …]) или None при ошибке.
    URL источников — redirect-ссылки Google (vertexaisearch…), они рабочие.
    """
    data = generate(
        [{"text": query}],
        system=(
            "You are a web search assistant. Answer factually based on Google "
            "Search results, concise and specific. Reply in the language of "
            "the query. No preamble."
        ),
        tools=[{"google_search": {}}],
        temperature=0.2,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    text = extract_text(data)
    if not text:
        return None

    sources: List[Tuple[str, str]] = []
    try:
        chunks = data["candidates"][0].get("groundingMetadata", {}).get("groundingChunks", [])
    except (KeyError, IndexError, TypeError):
        chunks = []
    for ch in chunks:
        web = ch.get("web") or {}
        uri = web.get("uri", "")
        if uri:
            sources.append((web.get("title", "") or "источник", uri))
    return text, sources
