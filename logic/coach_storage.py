"""
Coach storage — JSON-файлы для целей/дедлайнов/дневника.

Структура:
  data/coach/goals.json    — список целей с прогрессом
  data/coach/deadlines.json — дедлайны с датой
  data/coach/diary.json    — записи дневника

Простой JSON, без БД — на старте достаточно. Если разрастётся — мигрируем в SQLite.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.time import now_local

logger = logging.getLogger(__name__)


COACH_DIR_CANDIDATES = [
    Path("data/coach"),
    Path(__file__).parent.parent / "data" / "coach",
]


def _coach_dir() -> Path:
    """Возвращает существующую или создаёт первую кандидатную директорию."""
    for p in COACH_DIR_CANDIDATES:
        if p.parent.exists():
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = COACH_DIR_CANDIDATES[0]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json(name: str, default: Any) -> Any:
    p = _coach_dir() / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read %s: %s — returning default", p, e)
        return default


def _save_json(name: str, data: Any) -> None:
    p = _coach_dir() / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id(items: List[Dict[str, Any]]) -> int:
    return (max((i.get("id", 0) for i in items), default=0) + 1)


# ============================================================================
# Goals
# ============================================================================

def list_goals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    goals = _load_json("goals.json", [])
    if status:
        goals = [g for g in goals if g.get("status") == status]
    return goals


def add_goal(title: str, why: str = "", target_date: Optional[str] = None) -> Dict[str, Any]:
    goals = _load_json("goals.json", [])
    goal = {
        "id": _next_id(goals),
        "title": title.strip(),
        "why": why.strip(),
        "target_date": target_date,
        "status": "active",
        "created": now_local().strftime("%Y-%m-%d"),
        "progress_log": [],
    }
    goals.append(goal)
    _save_json("goals.json", goals)
    return goal


def mark_goal_done(goal_id: int, note: str = "") -> Optional[Dict[str, Any]]:
    goals = _load_json("goals.json", [])
    for g in goals:
        if g.get("id") == goal_id:
            g["status"] = "done"
            g["closed"] = now_local().strftime("%Y-%m-%d")
            if note:
                g.setdefault("progress_log", []).append({
                    "date": now_local().strftime("%Y-%m-%d"),
                    "note": note,
                })
            _save_json("goals.json", goals)
            return g
    return None


# ============================================================================
# Deadlines
# ============================================================================

def list_deadlines(upcoming_days: Optional[int] = None) -> List[Dict[str, Any]]:
    deadlines = _load_json("deadlines.json", [])
    if upcoming_days is not None:
        from datetime import timedelta
        cutoff = now_local().date() + timedelta(days=upcoming_days)
        result = []
        for d in deadlines:
            try:
                dt = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
                if now_local().date() <= dt <= cutoff:
                    result.append(d)
            except ValueError:
                continue
        return result
    return deadlines


def add_deadline(title: str, due: str, importance: str = "medium") -> Dict[str, Any]:
    """due: YYYY-MM-DD. importance: low/medium/high."""
    deadlines = _load_json("deadlines.json", [])
    deadline = {
        "id": _next_id(deadlines),
        "title": title.strip(),
        "due": due,
        "importance": importance,
        "status": "pending",
        "created": now_local().strftime("%Y-%m-%d"),
    }
    deadlines.append(deadline)
    _save_json("deadlines.json", deadlines)
    return deadline


# ============================================================================
# Diary
# ============================================================================

def add_diary_entry(text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    diary = _load_json("diary.json", [])
    entry = {
        "id": _next_id(diary),
        "timestamp": now_local().isoformat(timespec="minutes"),
        "text": text.strip(),
        "tags": tags or [],
    }
    diary.append(entry)
    _save_json("diary.json", diary)
    return entry


def read_diary(last_n: int = 10, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    diary = _load_json("diary.json", [])
    if tag:
        diary = [d for d in diary if tag in (d.get("tags") or [])]
    return diary[-last_n:]


# ============================================================================
# Presence — фиксация «проснулся» по первому сообщению дня
# ============================================================================

# Окно пробуждения: первое сообщение Влада в этом интервале считается подъёмом.
# Сообщения 00:00–05:00 — ночные посиделки, не подъём.
_WAKE_WINDOW = (5, 14)  # часы, [from, to)


def log_wake_if_first() -> Optional[Dict[str, Any]]:
    """
    Вызывается на каждом сообщении владельца. Если это первое сообщение
    сегодня в окне пробуждения — пишет запись в дневник (тег «сон»)
    и возвращает её; иначе None. Идемпотентно по дате (presence.json).
    Sync без await — в одном event loop гонки между 4 ботами нет.
    """
    now = now_local()
    if not (_WAKE_WINDOW[0] <= now.hour < _WAKE_WINDOW[1]):
        return None

    presence = _load_json("presence.json", {})
    today = now.strftime("%Y-%m-%d")
    if presence.get("last_wake_date") == today:
        return None

    presence["last_wake_date"] = today
    presence["wake_time"] = now.strftime("%H:%M")
    _save_json("presence.json", presence)
    return add_diary_entry(
        f"Проснулся — первое сообщение в {presence['wake_time']}",
        tags=["сон"],
    )


def woke_today() -> bool:
    presence = _load_json("presence.json", {})
    return presence.get("last_wake_date") == now_local().strftime("%Y-%m-%d")


# ============================================================================
# Day state — анти-спам для проактивных пингов (тикер)
# ============================================================================

def get_day_state() -> Dict[str, Any]:
    """State за сегодня: какие пинги отправлены, snooze. Авто-сброс на новой дате."""
    state = _load_json("day_state.json", {})
    today = now_local().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "pings": {}, "snooze_until": None}
    return state


def save_day_state(state: Dict[str, Any]) -> None:
    _save_json("day_state.json", state)


def mark_ping(ping_id: str) -> None:
    state = get_day_state()
    state["pings"][ping_id] = now_local().strftime("%H:%M")
    save_day_state(state)


def set_snooze(hours: float) -> str:
    """Тишина пингов до (сейчас + hours). Возвращает «HH:MM» до которого молчим."""
    from datetime import timedelta
    hours = max(0.5, min(hours, 12.0))
    until = now_local() + timedelta(hours=hours)
    state = get_day_state()
    state["snooze_until"] = until.isoformat(timespec="minutes")
    save_day_state(state)
    return until.strftime("%H:%M")


def today_tags() -> set:
    """Все теги дневника за сегодня — тикер проверяет закрыт ли слот (питание/спорт/…)."""
    today = now_local().strftime("%Y-%m-%d")
    tags: set = set()
    for e in _load_json("diary.json", []):
        if str(e.get("timestamp", "")).startswith(today):
            tags.update(e.get("tags") or [])
    return tags
