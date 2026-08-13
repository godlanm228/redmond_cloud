"""
Gemini API (generativelanguage.googleapis.com) — единый низкоуровневый клиент.

Ключ REDMOND_GEMINI_API_KEY = free-tier проект RedmondFree (~10 RPM / сотни RPD)
— для наших объёмов с запасом (реальный расход ~16 запросов в день). ВАЖНО:
проект должен быть БЕЗ биллинга — проект с prepay-биллингом при нуле кредитов
отдаёт 429 без отката на free tier (так «не работал» старый ключ).
У google_search grounding квота ОТДЕЛЬНАЯ и заметно меньше — см. GROUNDING_MODEL.

Модель по умолчанию — REDMOND_GEMINI_MODEL из env, иначе DEFAULT_MODEL.

Используется для:
  • vision-разбора фото (logic/vision.py) — качество сильно выше Groq scout
  • web-поиска через Google Search grounding (logic/tools.py) — настоящий
    Google вместо мёртвого CSE (403) и DDG-костылей
  • перевода заголовков дайджеста (logic/digest.py)
  • fallback-генерации когда Groq исчерпал дневной TPD (response_generator)

Думание отключается во всех вызовах: модель по умолчанию «думает» и молча
сжигает выходные токены на рассуждения — для наших коротких задач это вред.
Параметр разный по семействам (см. _thinking_config): 2.5 понимает только
thinkingBudget, 3.x — только thinkingLevel, и каждое отвечает 400 на чужое.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = os.getenv("REDMOND_GEMINI_MODEL", "gemini-3.6-flash")


def api_key_from_env() -> str:
    return os.getenv("REDMOND_GEMINI_API_KEY", "")


def _thinking_config(model: str) -> Dict[str, Any]:
    """Как выключить размышления у конкретного семейства моделей.

    Проверено на живом API 13.08.2026:
      • 3.x + thinkingBudget=0 → 400 INVALID_ARGUMENT; без конфига думает всегда
        и съедает весь maxOutputTokens (ответ приходит пустым, finish=MAX_TOKENS)
      • 2.5 + thinkingLevel   → 400 «Thinking level is not supported for this model»
    Незнакомое семейство → thinkingBudget (поведение до 3.x, безопасный дефолт).
    """
    return ({"thinkingLevel": "minimal"} if (model or DEFAULT_MODEL).startswith("gemini-3")
            else {"thinkingBudget": 0})


def _bump_usage() -> None:
    """Инкремент дневного счётчика запросов Gemini (RPD-гард free-tier:
    1500/день на проект, пул общий с vision/поиском/дайджестом/Iris). Единый
    чокпоинт — здесь проходят ВСЕ вызовы. Никогда не роняет генерацию."""
    try:
        from logic import coach_storage
        coach_storage.gemini_bump()
    except Exception:
        pass


def _post_generate(key: str, body: Dict[str, Any], model: str, timeout: float) -> Optional[dict]:
    """POST generateContent с одним ретраем. Засчитывает запрос в RPD-гард при успехе.
    Один ретрай: транзиентный 429/timeout не должен рушить fallback/поиск."""
    import time
    last_err = None
    for _attempt in range(2):
        try:
            # Ключ в заголовке, НЕ в query-параметре: requests включает полный URL
            # в текст исключения, и ?key=… месяцами светился в v2.log плейнтекстом.
            resp = requests.post(
                f"{_API_BASE}/models/{model or DEFAULT_MODEL}:generateContent",
                headers={"x-goog-api-key": key},
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            _bump_usage()
            return resp.json()
        except Exception as e:
            last_err = e
            if _attempt == 0:
                time.sleep(1.0)
    logger.warning("Gemini call failed (2 attempts): %s", last_err)
    return None


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
            "thinkingConfig": _thinking_config(model),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools

    return _post_generate(key, body, model, timeout)


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


def generate_contents(
    contents: List[Dict[str, Any]],
    *,
    model: str = "",
    system: str = "",
    tools: Optional[List[dict]] = None,
    tool_config: Optional[dict] = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout: float = 45.0,
    api_key: str = "",
) -> Optional[dict]:
    """generateContent с полным списком contents (multi-turn function calling).
    contents — список {'role': 'user'|'model', 'parts': [...]}. None при ошибке."""
    key = api_key or api_key_from_env()
    if not key:
        return None
    body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": _thinking_config(model),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    if tool_config:
        body["toolConfig"] = tool_config
    return _post_generate(key, body, model, timeout)


def extract_function_calls(data: Optional[dict]) -> List[Dict[str, Any]]:
    """functionCall-части ответа → [{name, id, args}, …] ([] если их нет)."""
    if not data:
        return []
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return []
    calls = []
    for p in parts:
        fc = p.get("functionCall")
        if fc and fc.get("name"):
            calls.append({"name": fc["name"], "id": fc.get("id", ""), "args": fc.get("args") or {}})
    return calls


# Конвертация наших OpenAI-style TOOL_SCHEMAS → Gemini functionDeclarations.
# Gemini не принимает union-типы (["string","null"]); lowercase-типов достаточно.
def _conv_schema(s: Any) -> Dict[str, Any]:
    if not isinstance(s, dict):
        return {}
    out: Dict[str, Any] = {}
    t = s.get("type")
    if isinstance(t, list):  # ["string","null"] → базовый не-null тип
        t = next((x for x in t if x != "null"), "string")
    if t:
        out["type"] = t
    if s.get("description"):
        out["description"] = s["description"]
    if s.get("enum"):
        out["enum"] = [e for e in s["enum"] if e is not None]
    if "items" in s:
        out["items"] = _conv_schema(s["items"])
    if "properties" in s:
        out["properties"] = {k: _conv_schema(v) for k, v in s["properties"].items()}
    if s.get("required"):
        out["required"] = s["required"]
    return out


def tool_schemas_to_gemini(schemas: List[dict]) -> Optional[List[dict]]:
    """[{type:function, function:{name,description,parameters}}] → Gemini-формат.
    Пустой parameters Gemini не любит — опускаем (tool без аргументов)."""
    decls = []
    for t in schemas:
        fn = t.get("function", {})
        if not fn.get("name"):
            continue
        decl: Dict[str, Any] = {"name": fn["name"], "description": fn.get("description", "")}
        params = fn.get("parameters") or {}
        if params.get("properties"):
            decl["parameters"] = _conv_schema(params)
        decls.append(decl)
    return [{"functionDeclarations": decls}] if decls else None


# Grounding намеренно прибит к 2.5, а не к DEFAULT_MODEL: у google_search своя
# квота, и 13.08.2026 она была выжжена до того, как удалось проверить 3.6-flash.
# 2.5 этот путь отработал 20 раз из 23 за всё время — не меняем вслепую.
# TODO: проверить grounding на DEFAULT_MODEL и убрать пин.
GROUNDING_MODEL = "gemini-2.5-flash"


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
        model=GROUNDING_MODEL,
        system=(
            "You are a web search assistant. Answer factually based on Google "
            "Search results, concise and specific. Reply in the language of "
            "the query. No preamble. "
            "The user is Ukrainian: do NOT use, cite, or repeat claims from Russian "
            "state / propaganda / aggregator sources (ria, tass, rt, lenta, rbc, "
            "gazeta, vesti, regnum, iz.ru, kp.ru, etc.). For any Russia/Ukraine war "
            "or politics topic rely ONLY on Western, Ukrainian or neutral international "
            "sources."
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
