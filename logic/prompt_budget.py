"""Бюджет промпта: измерение размера и ТЕНЕВОЙ отбор инструментов.

Зачем. 12.08.2026 Redmond словил 429 на запросе «Почему гемини упал?».
Разбор промпта, который не пролез в 8000 TPM:

    ~4086 ток  схемы 32 инструментов   ← 58%, уходят на КАЖДОМ хопе
    ~1500 ток  выдача web_search
     ~800 ток  системный промпт
     ~600 ток  история чата
      ~40 ток  собственно вопрос        ← 0.5%

То есть лимит выбило не тяжёлым вопросом, а накладными расходами. Схемы
инструментов снимаются только на последнем хопе (там compose без tools) —
а падало на первом же хопе после поиска.

Этот модуль пока НИЧЕГО НЕ МЕНЯЕТ в поведении. Он:
  1. считает и пишет в лог реальный размер промпта (раньше его не было видно
     вообще — поэтому проблему и не замечали);
  2. в тени прогоняет отбор инструментов и логирует, сколько бы сэкономили
     и — главное — не отрезали ли мы инструмент, который модель реально
     запросила. Вот это и есть метрика, по которой решаем, включать ли отбор.

Включение — отдельным решением, после недели наблюдений.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence, Set

logger = logging.getLogger(__name__)

# Free-tier Groq на gpt-oss-120b. Нужен только для порога предупреждения —
# на поведение не влияет. Проверено по заголовкам x-ratelimit 13.08.2026.
GROQ_TPM_LIMIT = 8000
WARN_RATIO = 0.7

# Инструменты, которые кладём в набор всегда: дешёвые, общие и запрашиваются
# без предсказуемых ключевых слов. Лучше переплатить сотню токенов, чем
# оставить модель без базовой возможности.
CORE_TOOLS = frozenset({
    "get_current_time",
    "web_search",
    "read_dossier_section",
})

MAX_SELECTED = 10

_WORD_RX = re.compile(r"\w{4,}", re.UNICODE)


def estimate_tokens(obj: Any) -> int:
    """Грубая оценка: ~4 символа на токен.

    Точный счётчик потребовал бы токенизатора модели — ради порога в логе это
    лишняя зависимость на VM с 954 MB RAM. Для «схемы съели половину бюджета»
    точности хватает с запасом.
    """
    if isinstance(obj, str):
        return len(obj) // 4
    try:
        return len(json.dumps(obj, ensure_ascii=False)) // 4
    except (TypeError, ValueError):
        return len(str(obj)) // 4


def describe(messages: Sequence[dict], tools: Any) -> Dict[str, int]:
    """Разложение промпта по статьям расходов (в примерных токенах)."""
    parts = {"system": 0, "user": 0, "assistant": 0, "tool_results": 0}
    for m in messages:
        role = m.get("role", "")
        size = estimate_tokens(m.get("content") or "") + estimate_tokens(m.get("tool_calls") or "")
        if role == "tool":
            parts["tool_results"] += size
        elif role in parts:
            parts[role] += size
    parts["tool_schemas"] = estimate_tokens(tools) if tools else 0
    parts["total"] = sum(parts.values())
    return parts


def _tool_name(schema: dict) -> str:
    return (schema.get("function") or {}).get("name", "")


def select_tools(user_text: str, tools: Sequence[dict],
                 max_selected: int = MAX_SELECTED) -> List[dict]:
    """Отбор инструментов по релевантности сообщению.

    Намеренно тупой: пересечение слов запроса с именем и описанием инструмента
    плюс постоянное ядро. Умнее пока не надо — сначала теневой прогон покажет,
    хватает ли вообще такого отбора. Если он будет отрезать нужное, вкладываться
    в классификатор; если нет — незачем платить лишним LLM-вызовом.
    """
    if not tools:
        return []
    words = {w.lower() for w in _WORD_RX.findall(user_text or "")}

    scored = []
    for t in tools:
        name = _tool_name(t)
        if name in CORE_TOOLS:
            continue
        fn = t.get("function") or {}
        haystack = f"{name} {fn.get('description', '')}".lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, name, t))
    scored.sort(key=lambda x: (-x[0], x[1]))

    core = [t for t in tools if _tool_name(t) in CORE_TOOLS]
    picked = core + [t for _, _, t in scored[:max(0, max_selected - len(core))]]
    return picked


def log_shadow(agent_name: str, user_text: str, messages: Sequence[dict],
               tools: Sequence[dict]) -> Set[str]:
    """Записать в лог размер промпта и что дал бы отбор. Возвращает имена
    отобранных инструментов — вызывающий сверяет с тем, что модель запросила."""
    parts = describe(messages, tools)
    total = parts["total"]
    line = (f"Prompt [{agent_name}]: ~{total} ток "
            f"(схемы {parts['tool_schemas']}, system {parts['system']}, "
            f"user {parts['user']}, tools-out {parts['tool_results']})")
    if total > GROQ_TPM_LIMIT * WARN_RATIO:
        logger.warning("%s — больше %d%% лимита TPM (%d)",
                       line, int(WARN_RATIO * 100), GROQ_TPM_LIMIT)
    else:
        logger.info(line)

    selected = select_tools(user_text, tools)
    names = {_tool_name(t) for t in selected}
    if tools:
        saved = parts["tool_schemas"] - estimate_tokens(selected)
        logger.info("Shadow-отбор [%s]: %d из %d инструментов, сэкономили бы ~%d ток",
                    agent_name, len(selected), len(tools), max(0, saved))
    return names


def log_selection_miss(agent_name: str, requested: str, selected: Set[str]) -> None:
    """Модель запросила инструмент, которого теневой отбор бы не дал.

    Это ЕДИНСТВЕННАЯ метрика, по которой решается включение отбора: пока такие
    строки появляются, включать нельзя — модель осталась бы без инструмента.

    Пустой `selected` = теневой прогон не считался (не тот хоп, tools сняты) —
    промахом это не является.
    """
    if not selected:
        return
    if requested and requested not in selected:
        logger.warning("Shadow-промах [%s]: модель запросила %s, отбор его не включил",
                       agent_name, requested)
