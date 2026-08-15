"""
Недельное расписание Влада: рабочие смены (из скрина графика, парсинг в logic/vision.py)
+ статичное учебное расписание (SoSe 2026, HRW Campus Bottrop).

Смены: таблица shifts (одна строка на дату) + append-only журнал shift_events.
Метаданные: status/source/confidence/last_confirmed_at/updated/note.
Форма словаря сохранена как в старом shifts.json — вызывающие не менялись.
Учёба: константа в коде (меняется раз в семестр — проще править тут, чем JSON).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)

# Приоритет источников (B1). Машинная догадка НЕ перетирает то, что сказал
# человек: 12.08.2026 фото графика с двумя сотрудниками записало чужие смены,
# а бот при этом сам предлагал «поправь текстом» — и следующее фото стёрло бы
# правку обратно. Равный или больший приоритет побеждает, меньший — отклоняется
# и попадает в журнал с причиной.
SOURCE_PRIORITY = {"manual": 3, "text": 2, "photo": 1, "scheduler": 1, "unknown": 0}


def _priority(source: str) -> int:
    return SOURCE_PRIORITY.get(str(source or "unknown").lower(), 0)

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


def _shift_row(r) -> Dict[str, Any]:
    """Строка таблицы → тот же словарь, что раньше лежал в shifts.json.
    Пустые поля опускаем: у вызывающих есть проверки на их отсутствие."""
    out: Dict[str, Any] = {
        "start": r["start"], "end": r["end"], "status": r["status"],
        "source": r["source"], "confidence": r["confidence"],
        "updated": r["updated"],
    }
    if r["last_confirmed_at"]:
        out["last_confirmed_at"] = r["last_confirmed_at"]
    if r["note"]:
        out["note"] = r["note"]
    return out


def get_shift_record(d: date) -> Optional[Dict[str, Any]]:
    """Raw record for a date, including cancelled/uncertain metadata."""
    row = db.query_one("SELECT * FROM shifts WHERE date=?", (d.strftime("%Y-%m-%d"),))
    return _shift_row(row) if row is not None else None


def log_shift_event(date_str: str, action: str, source: str,
                    payload: Optional[Dict[str, Any]] = None,
                    reason: str = "") -> None:
    """Запись в append-only журнал смен (B2).

    Отвечает на вопрос «откуда тут взялась эта смена», который 12.08.2026
    пришлось выяснять вручную по логам и переписке.
    """
    import json
    db.execute(
        "INSERT INTO shift_events(ts, date, action, source, payload, reason)"
        " VALUES(?,?,?,?,?,?)",
        (now_local().isoformat(timespec="minutes"), date_str, action,
         str(source or "unknown"),
         json.dumps(payload or {}, ensure_ascii=False), reason or None),
    )


def shift_history(date_str: str) -> List[Dict[str, Any]]:
    """История изменений по дате — для ответа «почему тут эта смена»."""
    return [
        {"ts": r["ts"], "action": r["action"], "source": r["source"],
         "reason": r["reason"]}
        for r in db.query(
            "SELECT * FROM shift_events WHERE date=? ORDER BY id", (date_str,))
    ]


def get_shift(d: date) -> Optional[Dict[str, Any]]:
    """Active shift only. Cancelled records are visible via get_shift_record()."""
    shift = get_shift_record(d)
    return shift if _is_active_shift(shift) else None


def save_shifts(items: List[Dict[str, Any]]) -> int:
    """Merge смен. Возвращает сколько РЕАЛЬНО сохранено.

    Backward-compatible: callers may still pass only date/start/end. Optional
    metadata lets text corrections and photo imports carry status/source/confidence.

    С 15.08.2026 действует приоритет источников: запись с меньшим приоритетом
    не перетирает существующую (фото не стирает правку текстом). Отклонённые
    попытки не молчат — уходят в журнал с причиной и видны в shift_history().
    """
    n = 0
    updated = now_local().isoformat(timespec="minutes")
    with db.transaction() as conn:
        for it in items:
            d, start, end = it.get("date"), it.get("start"), it.get("end")
            if not d or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
                continue

            row = conn.execute("SELECT * FROM shifts WHERE date=?", (d,)).fetchone()
            prev = _shift_row(row) if row is not None else {}
            status = str(it.get("status") or prev.get("status") or "planned").strip().lower()
            source = str(it.get("source") or prev.get("source")
                         or ("text" if status == "cancelled" else "unknown"))

            # Приоритет: равный или выше — пишем; ниже — отклоняем с записью
            # в журнал. Одинаковые значения не считаем конфликтом: фото,
            # подтверждающее уже известную смену, ничего не портит.
            if prev:
                same = (prev.get("start") == start and prev.get("end") == end
                        and prev.get("status") == status)
                if not same and _priority(source) < _priority(prev.get("source", "")):
                    log_shift_event(
                        d, "reject", source, dict(it),
                        reason=(f"приоритет ниже: {source} не перетирает "
                                f"{prev.get('source')} ({prev.get('start')}–{prev.get('end')})"),
                    )
                    logger.info("Смена %s: отклонён %s поверх %s", d, source,
                                prev.get("source"))
                    continue

            if status == "cancelled":
                record = {
                    **prev,
                    "status": "cancelled",
                    "source": source,
                    "confidence": it.get("confidence") or prev.get("confidence") or "high",
                    "updated": updated,
                }
            else:
                if not (start and end):
                    continue
                record = {
                    **prev,
                    "start": start,
                    "end": end,
                    "status": status,
                    "source": source,
                    "confidence": it.get("confidence") or prev.get("confidence") or "medium",
                    "updated": updated,
                }
                if status == "confirmed":
                    record["last_confirmed_at"] = updated
            if it.get("note"):
                record["note"] = str(it["note"]).strip()

            conn.execute(
                "INSERT INTO shifts(date, start, end, status, source, confidence,"
                " updated, last_confirmed_at, note) VALUES(?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(date) DO UPDATE SET start=excluded.start,"
                " end=excluded.end, status=excluded.status, source=excluded.source,"
                " confidence=excluded.confidence, updated=excluded.updated,"
                " last_confirmed_at=excluded.last_confirmed_at, note=excluded.note",
                (d, record.get("start"), record.get("end"), record["status"],
                 record["source"], record["confidence"], record["updated"],
                 record.get("last_confirmed_at"), record.get("note")),
            )
            conn.execute(
                "INSERT INTO shift_events(ts, date, action, source, payload, reason)"
                " VALUES(?,?,?,?,?,?)",
                (updated, d, "cancel" if status == "cancelled" else "set",
                 record["source"], _dumps(record), None),
            )
            n += 1
    return n


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


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
