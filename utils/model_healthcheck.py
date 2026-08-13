"""Проверка моделей на старте: жива ли каждая модель из конфига.

Зачем. `qwen/qwen3-32b` Groq снёс где-то между июнем и августом 2026. Узнали мы
об этом 12.08 — по молчанию бота в чате, через два месяца после того, как модель
умерла. Fallback-модель по определению используется редко, поэтому её смерть
незаметна ровно до того момента, когда она нужна.

Один запрос на модель при запуске (max_tokens=1) закрывает этот класс: снятая
модель видна в логе сразу, а не в момент отказа основной.

Старт НЕ блокируем: провайдер может лежать временно, бот всё равно должен
подняться и работать на том, что живо.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

OK = "ok"
GONE = "gone"        # снята провайдером — чинится только правкой конфига
UNAVAILABLE = "unavailable"  # 429/сеть/ключ — транзиентно, чинится само

_TIMEOUT_SEC = 15.0


def _classify(err: str) -> str:
    low = (err or "").lower()
    if "model_not_found" in low or "does not exist" in low or "not found" in low:
        return GONE
    return UNAVAILABLE


def _check_groq(model: str, api_key: str) -> Tuple[str, str]:
    """(статус, деталь) для одной модели Groq."""
    if not api_key:
        return UNAVAILABLE, "нет ключа"
    try:
        from groq import Groq
    except ImportError:
        return UNAVAILABLE, "groq SDK не установлен"
    try:
        client = Groq(api_key=api_key, timeout=_TIMEOUT_SEC, max_retries=0)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return OK, ""
    except Exception as e:
        return _classify(str(e)), str(e)[:160]


def _check_gemini(model: str) -> Tuple[str, str]:
    """(статус, деталь) для одной модели Gemini.

    Ходим через наш же generate(), а не голым запросом: заодно проверяется, что
    thinkingConfig для этого семейства подобран верно (на 3.x с thinkingBudget=0
    прилетает 400 — ровно этим ловится «поменяли модель, забыли параметр»).
    """
    try:
        from utils import gemini
    except ImportError as e:
        return UNAVAILABLE, str(e)[:160]
    if not gemini.api_key_from_env():
        return UNAVAILABLE, "нет ключа"
    data = gemini.generate([{"text": "ping"}], model=model, max_tokens=1, timeout=_TIMEOUT_SEC)
    # generate() гасит ошибку внутри и отдаёт None; тела ошибки тут нет,
    # поэтому статус общий — деталь ищем в предыдущей строке лога от gemini.
    return (OK, "") if data is not None else (UNAVAILABLE, "нет ответа (детали строкой выше)")


def check_models(config: Any) -> List[Tuple[str, str, str, str]]:
    """Проверяет все модели из конфига. Возвращает [(провайдер, модель, статус, деталь)]."""
    from logic import agent_router
    from utils import gemini

    groq_key = getattr(config, "groq_api_key", "")
    groq_models = [m for m in (getattr(config, "groq_model", ""),
                               getattr(config, "groq_fallback_model", "")) if m]
    gemini_models = [m for m in (getattr(config, "gemini_model", "") or gemini.DEFAULT_MODEL,
                                 gemini.GROUNDING_MODEL,
                                 agent_router._ROUTER_MODEL) if m]

    results: List[Tuple[str, str, str, str]] = []
    for m in dict.fromkeys(groq_models):
        status, detail = _check_groq(m, groq_key)
        results.append(("groq", m, status, detail))
    for m in dict.fromkeys(gemini_models):
        status, detail = _check_gemini(m)
        results.append(("gemini", m, status, detail))
    return results


def run_and_log(config: Any) -> List[Tuple[str, str, str, str]]:
    """Прогнать проверку и написать итог в лог. Никогда не бросает."""
    try:
        results = check_models(config)
    except Exception:
        logger.warning("Healthcheck моделей упал — пропускаем", exc_info=True)
        return []

    gone = [r for r in results if r[2] == GONE]
    unavailable = [r for r in results if r[2] == UNAVAILABLE]

    alive = ", ".join(f"{m}" for _, m, s, _ in results if s == OK)
    logger.info("Healthcheck моделей: живых %d из %d (%s)",
                len(results) - len(gone) - len(unavailable), len(results), alive or "—")
    for provider, model, _, detail in unavailable:
        logger.warning("Healthcheck: %s/%s недоступна — %s", provider, model, detail)
    for provider, model, _, detail in gone:
        logger.error(
            "Healthcheck: %s/%s СНЯТА провайдером — правь config.json, "
            "иначе она молча не сработает в момент отказа основной. %s",
            provider, model, detail,
        )
    return results
