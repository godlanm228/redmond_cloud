"""
Недельное расписание Влада: рабочие смены (из скрина графика, парсинг в logic/vision.py)
+ статичное учебное расписание (SoSe 2026, HRW Campus Bottrop).

Смены: data/coach/shifts.json — {"YYYY-MM-DD": {"start": "HH:MM", "end": "HH:MM", ...}}.
Метаданные опциональны: status/source/confidence/last_confirmed_at/updated.
Старый формат со start/end остаётся валидным.
Учёба: константа в коде (меняется раз в семестр — проще править тут, чем JSON).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from logic.coach_storage import _load_json, _save_json  # тот же data/coach каталог
from utils.time import now_local

logger = logging.getLogger(__name__)

_DAY_NAMES = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

# Дорога (мин): работа — самокат от дома; универ — дом → Essen Hbf → SB16 (~34 мин) → кампус.
WORK_COMMUTE_MIN = 20
UNI_COMMUTE_MIN = 60

# Учебное расписание: weekday (0=пн) → [(start, end, что)]. Ботроп = дорога UNI_COMMUTE_MIN.
# Среда: туториум 13:15-14:50 Влад скипает осознанно — вместо него домашний блок.
STUDY_TIMETABLE: Dict[int, List[Tuple[str, str, str]]] = {
    0: [("14:05", "15:45", "Лекция Grundlagen der Ingenieurmathematik (Ботроп)")],
    1: [
        ("12:20", "14:00", "Лекция Ingenieurmathematik (Ботроп)"),
        ("14:05", "15:45", "Практика Ingenieurmathematik (Ботроп)"),
    ],
    2: [("13:15", "14:50", "Домашняя учёба (туториум скипается в пользу дома)")],
    4: [
        ("08:00", "09:35", "Лекция Projektmanagement (Ботроп)"),
        ("11:30", "13:05", "Практика Projektmanagement (Ботроп)"),
    ],
}

# Дни без пар и обычно без смен — кандидаты на «день на себя» (CS/Dota/друзья).
HOME_STUDY_WEEKDAYS = (2, 3)  # ср (вместо туториума), чт (пар нет)


# ---------- смены: storage ----------

ACTIVE_SHIFT_STATUSES = {"planned", "confirmed", "uncertain", "moved"}


def _is_active_shift(shift: Optional[Dict[str, Any]]) -> bool:
    return bool(shift and shift.get("start") and shift.get("end")
                and shift.get("status", "planned") in ACTIVE_SHIFT_STATUSES)


def get_shift_record(d: date) -> Optional[Dict[str, Any]]:
    """Raw record for a date, including cancelled/uncertain metadata."""
    shifts = _load_json("shifts.json", {})
    item = shifts.get(d.strftime("%Y-%m-%d"))
    return item if isinstance(item, dict) else None


def get_shift(d: date) -> Optional[Dict[str, Any]]:
    """Active shift only. Cancelled records are visible via get_shift_record()."""
    shift = get_shift_record(d)
    return shift if _is_active_shift(shift) else None


def save_shifts(items: List[Dict[str, Any]]) -> int:
    """Merge смен в shifts.json. Возвращает сколько сохранено.

    Backward-compatible: callers may still pass only date/start/end. Optional
    metadata lets text corrections and photo imports carry status/source/confidence.
    """
    shifts = _load_json("shifts.json", {})
    n = 0
    updated = now_local().isoformat(timespec="minutes")
    for it in items:
        d, start, end = it.get("date"), it.get("start"), it.get("end")
        if not d:
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        prev = shifts.get(d, {}) if isinstance(shifts.get(d), dict) else {}
        status = str(it.get("status") or prev.get("status") or "planned").strip().lower()
        if status == "cancelled":
            record = {
                **prev,
                "status": "cancelled",
                "source": it.get("source") or prev.get("source") or "text",
                "confidence": it.get("confidence") or prev.get("confidence") or "high",
                "updated": updated,
            }
            if it.get("note"):
                record["note"] = str(it["note"]).strip()
        else:
            if not (start and end):
                continue
            record = {
                **prev,
                "start": start,
                "end": end,
                "status": status,
                "source": it.get("source") or prev.get("source") or "unknown",
                "confidence": it.get("confidence") or prev.get("confidence") or "medium",
                "updated": updated,
            }
            if status == "confirmed":
                record["last_confirmed_at"] = updated
            if it.get("note"):
                record["note"] = str(it["note"]).strip()
        shifts[d] = record
        n += 1
    if n:
        _save_json("shifts.json", shifts)
    return n


def describe_saved_shifts(items: List[Dict[str, str]], n: int) -> str:
    """Человекочитаемое подтверждение сохранённых смен для чата."""
    lines = []
    for it in sorted(items, key=lambda x: x.get("date", "")):
        try:
            d = datetime.strptime(it["date"], "%Y-%m-%d").date()
            lines.append(f"• {_DAY_NAMES[d.weekday()]} {d.strftime('%d.%m')}: {it['start']}–{it['end']}")
        except (KeyError, ValueError):
            continue
    return (
        f"Принял график, смен сохранено: {n}\n\n" + "\n".join(lines) +
        "\n\nЕсли что-то распознал криво — поправь текстом."
    )


def format_week(days: int = 8) -> str:
    """Человекочитаемое расписание на N дней вперёд: смены + пары. Для tool/пингов."""
    out: List[str] = []
    today = now_local().date()
    for i in range(days):
        d = today + timedelta(days=i)
        parts: List[str] = []
        shift_record = get_shift_record(d)
        shift = get_shift(d)
        if shift:
            parts.append(f"смена {shift['start']}–{shift['end']} (дорога ~{WORK_COMMUTE_MIN} мин)")
        elif shift_record and shift_record.get("status") == "cancelled":
            parts.append("смена отменена")
        for start, end, what in STUDY_TIMETABLE.get(d.weekday(), []):
            parts.append(f"{start}–{end} {what}")
        label = f"{_DAY_NAMES[d.weekday()]} {d.strftime('%d.%m')}"
        out.append(f"{label}: " + ("; ".join(parts) if parts else "свободен"))
    return "\n".join(out)
