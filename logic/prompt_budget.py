"""Бюджет промпта: сколько весит запрос, по частям, — в лог.

Зачем. 12.08.2026 Redmond словил 429 на запросе «Почему гемини упал?».
Разбор промпта, который не пролез в 8000 TPM:

    ~4086 ток  схемы 32 инструментов   ← 58%, уходят на КАЖДОМ хопе
    ~1500 ток  выдача web_search
     ~800 ток  системный промпт
     ~600 ток  история чата
      ~40 ток  собственно вопрос        ← 0.5%

То есть лимит выбило не тяжёлым вопросом, а накладными расходами. Размер
каждого запроса теперь виден в логе («Prompt [Iris]: ~N ток (схемы …)»).
Отбор инструментов, который это лечит, живёт в logic/tool_select (теневой
отбор, что был здесь, снят 01.10.2026: его заменил настоящий).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)

# Free-tier Groq на gpt-oss-120b. Нужен только для порога предупреждения —
# на поведение не влияет. Проверено по заголовкам x-ratelimit 13.08.2026.
GROQ_TPM_LIMIT = 8000
WARN_RATIO = 0.7



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


def log_size(agent_name: str, messages: Sequence[dict], tools: Sequence[dict]) -> int:
    """Записать в лог размер промпта по статьям. Возвращает оценку в токенах."""
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
    return total


