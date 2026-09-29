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
from typing import Any, Dict, List, Tuple

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
    from utils import groq
    _completion, err = groq.chat(model, [{"role": "user", "content": "ping"}],
                                 max_tokens=1, api_key=api_key, timeout=_TIMEOUT_SEC)
    return (OK, "") if not err else (_classify(err), err[:160])


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
    """Проверяет все модели всех пулов. Возвращает [(провайдер, модель, статус, деталь)].

    Модель, которую шлюз держит заблокированной по лимиту, не пингуется: она
    жива, у неё кончилась квота, а пинг потратил бы запрос и объявил бы её
    недоступной (дневная квота Gemini — 20 запросов).

    Gemini генерацией не пингуем вовсе: каждая проверка на старте съедала по
    запросу из 20 суточных у каждой модели пулов, а рестартов бывает несколько
    в день. Хватает каталога (бесплатно) и того, что видел шлюз: каталог
    показывает и снятые модели (gemini-2.5-flash-lite в нём есть, а отвечает
    404 «no longer available»), поэтому 404 из шлюза важнее каталога. Пинг —
    только если каталог недоступен. Уровень размышлений модели шлюз узнаёт
    сам по первому ответу (utils/gemini._raise_level_floor)."""
    from utils import gemini, llm_gate, model_catalog

    groq_key = getattr(config, "groq_api_key", "")
    catalog = None
    results: List[Tuple[str, str, str, str]] = []
    for provider, m in all_models(config):
        if llm_gate.gone(m):
            results.append((provider, m, GONE, "; ".join(llm_gate.describe([m]))))
            continue
        if llm_gate.blocked(m):
            results.append((provider, m, OK, "; ".join(llm_gate.describe([m])) or "в лимите"))
            continue
        if provider == "groq":
            status, detail = _check_groq(m, groq_key)
        else:
            if catalog is None:
                catalog = model_catalog.list_gemini(
                    getattr(config, "gemini_api_key", "") or gemini.api_key_from_env()) or []
            if catalog:
                status, detail = (OK, "в каталоге") if m in catalog else (GONE, "нет в каталоге")
            else:
                status, detail = _check_gemini(m)
        results.append((provider, m, status, detail))
    return results


def all_models(config: Any) -> List[Tuple[str, str]]:
    """(провайдер, модель) для всех моделей, на которых стоит хаб, без повторов.

    Пулы берутся как есть (их настраивает ResponseGenerator при старте): здесь
    их не перечитываем из конфига — это откатило бы замену снятой модели
    преемником, сделанную review_catalog минутой раньше."""
    from utils import llm_gate
    if not llm_gate.is_configured():
        # Проверка на старте идёт раньше, чем ResponseGenerator настроит шлюз:
        # без этого она видела пулы по умолчанию, а не config (29.09.2026).
        llm_gate.configure_from(config)
    lead = [getattr(config, f, "") for f in ("groq_model", "groq_fallback_model", "gemini_model")]
    seen = dict.fromkeys(m for m in lead + [m for models in llm_gate.pools().values()
                                             for m in models] if m)
    return [(llm_gate.provider_of(m), m) for m in seen]


def check_cipher_auth() -> Tuple[str, str]:
    """(статус, деталь) авторизации Claude Code CLI.

    Дешёвая проверка по файлу credentials — без запроса к API, значит без
    траты общего с десктопом лимита Pro. Ловит ровно тот случай, который
    случился 13–15.08.2026: авторизация исчезла, Cipher молча умер, и узнали
    об этом только когда полезли проверять руками.
    """
    try:
        from core.cipher_wrapper import auth_status
        status = auth_status()
    except Exception as e:  # noqa: BLE001
        return UNAVAILABLE, f"проверка не отработала: {e}"
    if not status["ok"]:
        return GONE, status["reason"]
    return (OK, status["reason"]) if status["reason"] else (OK, "")


def configured_models(config: Any) -> Dict[str, Tuple[str, str]]:
    """Роль → (провайдер, модель) для всех моделей, на которых стоит хаб.

    Основные и резервная модели чата — отдельными ролями (их замена — разное
    решение), остальные — по пулам задач. До 29.09.2026 роутер и поиск
    значились константами из кода, а vision не значился вовсе: снятую Groq
    llama-4-scout проверка так и не увидела."""
    from utils import gemini, llm_gate
    roles = {
        "groq_model": ("groq", getattr(config, "groq_model", "")),
        "groq_fallback_model": ("groq", getattr(config, "groq_fallback_model", "")),
        "gemini_model": ("gemini", getattr(config, "gemini_model", "") or gemini.DEFAULT_MODEL),
    }
    named = {m for _p, m in roles.values()}
    for provider, m in all_models(config):
        if m not in named:
            tasks = [t for t, models in llm_gate.pools().items() if m in models]
            roles[f"пул {', '.join(tasks)}: {m}"] = (provider, m)
    return roles


# Какие роли можно менять на преемника автоматически. Резервная модель
# Groq: пока она снята, резерва нет вообще, и любой вариант лучше. Основные
# модели сами не меняем — смена основной меняет поведение всех агентов, это
# решение владельца.
_AUTO_REPLACE_ROLES = {"groq_fallback_model"}


def review_catalog(config: Any) -> List[Any]:
    """Сверить наши модели с живыми каталогами; снятый резерв заменить преемником.

    Возвращает находки (utils.model_catalog.Finding). Замена действует до
    рестарта процесса и пишется в лог ERROR-ом: правка config.json остаётся
    за человеком, но хаб не стоит без резерва до этой правки.
    """
    from utils import gemini, model_catalog
    live = {
        "groq": model_catalog.list_groq(getattr(config, "groq_api_key", "")),
        "gemini": model_catalog.list_gemini(
            getattr(config, "gemini_api_key", "") or gemini.api_key_from_env()),
    }
    findings = model_catalog.review(configured_models(config), live)
    from utils import llm_gate
    for f in findings:
        if f.gone and f.newer and (f.role in _AUTO_REPLACE_ROLES or f.role.startswith("пул ")):
            if f.role in _AUTO_REPLACE_ROLES:
                setattr(config, f.role, f.newer)
            llm_gate.replace_model(f.model, f.newer)
            logger.error("Модель %s снята провайдером — %s временно переключена на "
                         "преемника %s. Поправь config.json.", f.model, f.role, f.newer)
        elif f.gone:
            logger.error("Модель %s (%s) снята провайдером%s", f.model, f.role,
                         f", преемник: {f.newer}" if f.newer else ", преемника в каталоге нет")
        else:
            logger.info("Для %s (%s) есть новее: %s", f.role, f.model, f.newer)
    return findings


def describe_findings(findings: List[Any]) -> str:
    """Текст для владельца. Пусто — сообщать нечего."""
    lines = []
    for f in findings:
        if f.gone and f.newer and (f.role in _AUTO_REPLACE_ROLES or f.role.startswith("пул ")):
            lines.append(f"• {f.model} снята {f.provider} — резерв сам переключился на "
                         f"{f.newer}. Надо закрепить в config.json.")
        elif f.gone:
            lines.append(f"• {f.model} ({f.role}) снята {f.provider}"
                         + (f" — есть преемник {f.newer}." if f.newer else " — преемника нет."))
        else:
            lines.append(f"• {f.role}: у нас {f.model}, вышла {f.newer}. Проверить и перейти?")
    if not lines:
        return ""
    return "🔧 Модели изменились у провайдеров:\n" + "\n".join(lines)


def run_and_log(config: Any) -> List[Tuple[str, str, str, str]]:
    """Прогнать проверку и написать итог в лог. Никогда не бросает."""
    try:
        review_catalog(config)
    except Exception:  # noqa: BLE001
        logger.warning("Сверка с каталогом моделей упала — пропускаем", exc_info=True)
    auth, detail = check_cipher_auth()
    if auth == GONE:
        logger.error("Cipher НЕ АВТОРИЗОВАН: %s. Пока это так, он не отвечает "
                     "вообще — Влад узнает об этом, только обратившись к нему.", detail)
    elif detail:
        logger.warning("Cipher: %s", detail)
    else:
        logger.info("Cipher: авторизация в порядке")

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
