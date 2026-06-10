"""
Приоритеты владельца — детерминированный расчёт «что сейчас важно».

LLM не может рассуждать о том, чего нет в контексте: блок TOP PRIORITIES
собирается чистым Python из реальных дедлайнов (data/coach/deadlines.json)
+ слотов сегодняшнего дня и вставляется в системный промпт Iris при каждой
генерации. Ноль LLM-вызовов. Тот же расчёт питает crunch-пинг тикера.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from logic import coach_storage
from logic.week_schedule import STUDY_TIMETABLE, get_shift
from utils.time import now_local

_IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}
_DAY_NAMES = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _pending_deadlines() -> List[Dict[str, Any]]:
    out = []
    for d in coach_storage.list_deadlines():
        if d.get("status") != "pending":
            continue
        try:
            due = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        out.append({**d, "_due": due})
    return out


def top_priorities(max_items: int = 4) -> List[Dict[str, Any]]:
    """Pending-дедлайны по срочности (дата, потом важность).
    Просроченные не скрываются ещё неделю — наверху, с пометкой."""
    today = now_local().date()
    items = [d for d in _pending_deadlines() if d["_due"] >= today - timedelta(days=7)]
    items.sort(key=lambda d: (d["_due"], _IMPORTANCE_RANK.get(d.get("importance"), 1)))
    return items[:max_items]


def crunch_deadline(days: int = 3) -> Optional[Dict[str, Any]]:
    """Ближайший HIGH-дедлайн в пределах N дней (включая просроченный) —
    триггер crunch-пинга в тикере."""
    today = now_local().date()
    for d in top_priorities():
        if d.get("importance") == "high" and d["_due"] <= today + timedelta(days=days):
            return d
    return None


def build_priorities_block() -> str:
    """Блок для системного промпта Iris. Пустая строка если показывать нечего."""
    now = now_local()
    today = now.date()
    lines: List[str] = []

    prios = top_priorities()
    if prios:
        lines.append("TOP PRIORITIES (computed from real deadlines — this is the truth):")
        for i, d in enumerate(prios, 1):
            left = (d["_due"] - today).days
            when = f"{_DAY_NAMES[d['_due'].weekday()]} {d['_due'].strftime('%d.%m')}"
            if left < 0:
                tail = f"ПРОСРОЧЕН на {-left} дн"
            elif left == 0:
                tail = "СЕГОДНЯ"
            else:
                tail = f"осталось {left} дн"
            lines.append(
                f"  {i}. {when} — {d['title']} [{d.get('importance', 'medium')}] — {tail}"
            )

    day_parts: List[str] = []
    shift = get_shift(today)
    if shift:
        day_parts.append(f"смена {shift['start']}–{shift['end']}")
    for start, end, what in STUDY_TIMETABLE.get(today.weekday(), []):
        day_parts.append(f"{start}–{end} {what}")
    if day_parts:
        label = f"{_DAY_NAMES[today.weekday()]} {today.strftime('%d.%m')}"
        lines.append(f"TODAY ({label}): " + "; ".join(day_parts))

    return "\n".join(lines)
