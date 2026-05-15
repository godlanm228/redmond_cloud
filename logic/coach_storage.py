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
        "created": datetime.now().strftime("%Y-%m-%d"),
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
            g["closed"] = datetime.now().strftime("%Y-%m-%d")
            if note:
                g.setdefault("progress_log", []).append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
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
        cutoff = datetime.now().date() + timedelta(days=upcoming_days)
        result = []
        for d in deadlines:
            try:
                dt = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
                if datetime.now().date() <= dt <= cutoff:
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
        "created": datetime.now().strftime("%Y-%m-%d"),
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
        "timestamp": datetime.now().isoformat(timespec="minutes"),
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
