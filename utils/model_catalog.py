"""Живой каталог моделей провайдеров и сверка с нашим конфигом.

Зачем. Провайдеры меняют модели раз в несколько недель, а ID у нас записаны
в config.json. Groq снял `qwen/qwen3-32b` (узнали 12.08.2026 по молчанию бота)
и `qwen/qwen3.6-27b` (снят 14.09, узнали 28.09 на ручном разборе логов). Оба
раза преемник уже лежал в каталоге провайдера: `qwen/qwen3.6-27b`, потом
`qwen/qwen3.8-27b`. Параллельно у Gemini вышли 3.7 и 3.8 flash, а мы об этом
не знали.

Здесь три вещи:
  • `list_groq()` / `list_gemini()` — что провайдер отдаёт прямо сейчас;
  • `family()` — «семейство + версия» из ID (qwen/qwen3.8-27b → qwen/qwen*-27b, 3.8);
  • `review()` — для каждой нашей модели: жива ли, и есть ли новее в семействе.

Решение, что делать с находкой, принимает вызывающий: снятую резервную
модель можно заменить преемником сразу, основную — только сообщить.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
# Groq отдаёт 403 на User-Agent по умолчанию у urllib; ставим свой везде.
_UA = {"User-Agent": "redmond-hub/model-catalog"}

# ID → (семейство, версия, превью?). Версия — кортеж чисел для сравнения.
_FAMILY_RX = [
    # qwen/qwen3.8-27b → ("qwen/qwen*-27b", (3, 8))
    re.compile(r"^(?P<pre>qwen/qwen)(?P<ver>\d+(?:\.\d+)*)(?P<post>-[\w.-]+)$"),
    # gemini-3.8-flash, gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-3.1-pro-preview
    re.compile(r"^(?P<pre>gemini-)(?P<ver>\d+(?:\.\d+)*)(?P<post>-(?:flash-lite|flash|pro))"
               r"(?P<preview>-preview[\w-]*)?$"),
    # meta-llama/llama-4-scout-17b-… и подобные: версия после «llama-»
    re.compile(r"^(?P<pre>(?:meta-llama/)?llama-)(?P<ver>\d+(?:\.\d+)*)(?P<post>-[\w.-]+)$"),
]


def family(model: str) -> Optional[Tuple[str, Tuple[int, ...], bool]]:
    """(семейство, версия, превью) или None, если ID не версионирован."""
    for rx in _FAMILY_RX:
        m = rx.match(model or "")
        if m:
            ver = tuple(int(x) for x in m.group("ver").split("."))
            preview = bool(m.groupdict().get("preview"))
            return f"{m.group('pre')}*{m.group('post')}", ver, preview
    return None


def successor(model: str, live: Iterable[str]) -> Optional[str]:
    """Самая новая стабильная (не preview) модель того же семейства новее `model`."""
    fam = family(model)
    if fam is None:
        return None
    best: Optional[Tuple[Tuple[int, ...], str]] = None
    for cand in live:
        cf = family(cand)
        if cf is None or cf[0] != fam[0] or cf[2] or cf[1] <= fam[1]:
            continue
        if best is None or cf[1] > best[0]:
            best = (cf[1], cand)
    return best[1] if best else None


def list_groq(api_key: str) -> Optional[List[str]]:
    """ID моделей Groq или None, если каталог недоступен."""
    if not api_key:
        return None
    try:
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers=dict(_UA, Authorization=f"Bearer {api_key}"), timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("Каталог Groq недоступен: HTTP %s %s", r.status_code, r.text[:200])
            return None
        return [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except Exception as e:  # noqa: BLE001
        logger.warning("Каталог Groq недоступен: %s", e)
        return None


def list_gemini(api_key: str) -> Optional[List[str]]:
    """ID моделей Gemini с generateContent или None, если каталог недоступен."""
    if not api_key:
        return None
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"pageSize": 200},
                         headers=dict(_UA, **{"x-goog-api-key": api_key}), timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.warning("Каталог Gemini недоступен: HTTP %s %s", r.status_code, r.text[:200])
            return None
        return [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    except Exception as e:  # noqa: BLE001
        logger.warning("Каталог Gemini недоступен: %s", e)
        return None


@dataclass
class Finding:
    role: str            # где модель используется: «groq_fallback_model», «роутер» …
    provider: str
    model: str
    gone: bool           # снята провайдером
    newer: Optional[str]  # есть стабильная новее в том же семействе


def review(configured: Dict[str, Tuple[str, str]],
           live: Dict[str, Optional[List[str]]]) -> List[Finding]:
    """configured: роль → (провайдер, модель); live: провайдер → каталог|None.

    Модели провайдера с недоступным каталогом не оцениваются вообще: «каталог
    не ответил» не означает «модель снята».
    """
    out: List[Finding] = []
    for role, (provider, model) in configured.items():
        catalog = live.get(provider)
        if catalog is None or not model:
            continue
        gone = model not in catalog
        newer = successor(model, catalog)
        if gone or newer:
            out.append(Finding(role, provider, model, gone, newer))
    return out
