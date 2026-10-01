"""
Недельное расписание владельца: рабочие смены (скрин графика, календарь,
текст) + учёба и прочие события (пары, спорт, встречи).

Смены: таблица shifts (несколько смен в день, v9) + append-only журнал
shift_events. «Та же смена» — та, что пересекается по времени: план 17:00 и
факт 17:14 — одна смена, утренняя и вечерняя — две.
Метаданные: status/source/confidence/last_confirmed_at/updated/note.

Учёба и события: таблица timetable — еженедельные строки со сроком действия
(с 21.08.2026; до этого расписание было константой в коде и не отменялось
ничем — см. комментарий у _SEED_TIMETABLE) и разовые строки на дату (v9:
вузовский календарь выгружает конкретные даты). Читать только через
study_slots() / day_events() / is_home_study_day().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)

# Приоритет источников (B1). Машинная догадка НЕ перетирает то, что сказал
# человек: 12.08.2026 фото графика с двумя сотрудниками записало чужие смены,
# а бот при этом сам предлагал «поправь текстом» — и следующее фото стёрло бы
# правку обратно. Равный или больший приоритет побеждает.
# Календарь (.ics) — выгрузка из системы, где смены и ведут: точнее фото,
# поэтому на уровне слов владельца.
SOURCE_PRIORITY = {"manual": 3, "text": 2, "calendar": 2, "photo": 1, "scheduler": 1,
                   "unknown": 0}

# Что делать, когда фото противоречит уже сказанному текстом.
#   ask        — спросить Влада (дефолт): молча отклонить мало, он об этом
#                просто не узнает и будет думать, что график обновился;
#   keep_mine  — всегда оставлять свою правку, не переспрашивая;
#   photo_wins — всегда доверять свежему фото.
# Хранится в kv, ставится через tool по фразам вроде «всегда бери с фото».
CONFLICT_POLICY_KEY = "shift_conflict_policy"
PENDING_CONFLICTS_KEY = "pending_shift_conflicts"
POLICIES = ("ask", "keep_mine", "photo_wins")


def _priority(source: str) -> int:
    return SOURCE_PRIORITY.get(str(source or "unknown").lower(), 0)


def get_conflict_policy() -> str:
    value = str(db.kv_get(CONFLICT_POLICY_KEY, "ask") or "ask").strip().lower()
    return value if value in POLICIES else "ask"


def set_conflict_policy(policy: str) -> str:
    policy = str(policy or "").strip().lower()
    if policy not in POLICIES:
        policy = "ask"
    db.kv_set(CONFLICT_POLICY_KEY, policy)
    return policy


@dataclass
class ShiftConflict:
    """Фото противоречит тому, что Влад сказал текстом."""
    date: str
    incoming: Dict[str, Any]
    existing: Dict[str, Any]

    def describe(self) -> str:
        d = datetime.strptime(self.date, "%Y-%m-%d").date()
        label = f"{_DAY_NAMES[d.weekday()]} {d.strftime('%d.%m')}"
        return (f"{label}: в графике {self.incoming.get('start')}–{self.incoming.get('end')}, "
                f"а с твоих слов {self.existing.get('start')}–{self.existing.get('end')}")


@dataclass
class ShiftApplyResult:
    saved: int = 0
    conflicts: List[ShiftConflict] = field(default_factory=list)
    # Что реально легло в таблицу — для квитанции. Пересказывать разобранное,
    # а не записанное, нельзя: 01.10 бот отчитался «смен сохранено: 4», а в
    # базе от этих четырёх осталась одна.
    applied: List[Dict[str, Any]] = field(default_factory=list)

    def __int__(self) -> int:
        return self.saved

    def __bool__(self) -> bool:
        return bool(self.saved or self.conflicts)

_DAY_NAMES = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]

# Дорога (мин): работа — самокат от дома; универ — дом → Essen Hbf → SB16 (~34 мин) → кампус.
WORK_COMMUTE_MIN = 20
UNI_COMMUTE_MIN = 60

# Учебное расписание живёт в таблице `timetable`, НЕ в коде.
#
# До 21.08.2026 здесь стояла константа STUDY_TIMETABLE («SoSe 2026»), и именно
# поэтому её нельзя было отменить: 17.08 владелец сказал, что у него
# семестрфериен, запись легла в дневник — а бот 17, 18 и 19 августа продолжал
# звать на пары, потому что источник утверждения лежал вне данных. В промпт
# при этом попадали оба факта разом, и модель добросовестно пересказывала оба.
#
# Ниже — только СИД для разовой миграции. Живые данные читаются через
# study_slots() / is_home_study_day(), которые уважают срок действия.
_SEED_TIMETABLE: Dict[int, List[Tuple[str, str, str, str]]] = {
    0: [("14:05", "15:45", "Лекция Grundlagen der Ingenieurmathematik (Ботроп)", "lecture")],
    1: [
        ("12:20", "14:00", "Лекция Ingenieurmathematik (Ботроп)", "lecture"),
        ("14:05", "15:45", "Практика Ingenieurmathematik (Ботроп)", "lecture"),
    ],
    2: [("13:15", "14:50", "Домашняя учёба (туториум скипается в пользу дома)", "home_study")],
    3: [("", "", "Домашняя учёба (пар нет)", "home_study")],
    4: [
        ("08:00", "09:35", "Лекция Projektmanagement (Ботроп)", "lecture"),
        ("11:30", "13:05", "Практика Projektmanagement (Ботроп)", "lecture"),
    ],
}


def _iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")


# Виды строк timetable. Учебные — те, что зовут «на пары» и глушат пинги.
EVENT_KINDS = ("lecture", "home_study", "sport", "work", "rest", "other")
STUDY_KINDS = ("lecture", "home_study")
KIND_LABELS = {"lecture": "учёба", "home_study": "дом. учёба", "sport": "спорт",
               "work": "работа", "rest": "отдых", "other": "дело"}


def _rows_for(d: date, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Строки расписания, ДЕЙСТВУЮЩИЕ на дату d: еженедельные в своём сроке
    действия и разовые на эту дату. Вне срока — не существуют.

    День отдыха на весь день (каникулы, праздник: разовая строка kind=rest без
    часов) снимает в этот день еженедельные пары — серия «каждый понедельник»
    не идёт через Рождество."""
    day = _iso(d)
    sql = ("SELECT * FROM timetable WHERE ((date IS NULL AND weekday=? AND valid_from<=?"
           " AND (valid_to IS NULL OR valid_to>=?)) OR date=?)")
    rows = [dict(r) for r in db.query(sql + " ORDER BY start, id",
                                      (d.weekday(), day, day, day))]
    if any(r["date"] and r["kind"] == "rest" and not r["start"] for r in rows):
        rows = [r for r in rows if r["date"] or r["kind"] not in STUDY_KINDS]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return rows


def study_slots(d: date) -> List[Tuple[str, str, str]]:
    """Учебные занятия с часами на дату: [(start, end, что)]. Пусто вне срока."""
    return [(r["start"], r["end"], r["title"])
            for r in _rows_for(d) if r["start"] and r["end"] and r["kind"] in STUDY_KINDS]


def day_events(d: date) -> List[Dict[str, Any]]:
    """Все строки расписания на дату (учёба, спорт, встречи…), по времени."""
    return _rows_for(d)


def _check_event(title: str, kind: str, start: str, end: str) -> str:
    """Причина отказа или ''."""
    if not str(title or "").strip():
        return "нет названия"
    if kind not in EVENT_KINDS:
        return f"неизвестный вид «{kind}» (есть: {', '.join(EVENT_KINDS)})"
    if bool(start) != bool(end):
        return "нужны и начало, и конец (или ни того, ни другого — на весь день)"
    for hm in (start, end):
        if hm and not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", hm):
            return f"время «{hm}» не в формате HH:MM"
    return ""


def add_event(title: str, kind: str, start: str = "", end: str = "", *,
              on: Optional[date] = None, weekly_from: Optional[date] = None,
              until: Optional[date] = None, location: str = "",
              source: str = "manual") -> int:
    """Одна строка расписания: разовая (on=дата) или еженедельная (weekly_from —
    первое занятие, until — последнее включительно). Возвращает id.
    ValueError с причиной — если данные неполные."""
    why = _check_event(title, kind, start, end)
    if not why and (on is None) == (weekly_from is None):
        why = "нужна либо дата, либо начало еженедельной серии"
    if why:
        raise ValueError(why)
    created = now_local().isoformat(timespec="minutes")
    first = on or weekly_from
    cur = db.execute(
        "INSERT INTO timetable(weekday, start, end, title, kind, valid_from, valid_to,"
        " source, created, date, location) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (first.weekday(), start or "", end or "", str(title).strip()[:200], kind,
         _iso(first), _iso(until) if until else None, source, created,
         _iso(on) if on else None, str(location or "").strip()[:200]),
    )
    return int(cur.lastrowid)


def remove_event(row_id: int, on: date) -> str:
    """Убрать строку: разовую — удалить; еженедельную — закончить серию днём
    раньше `on` (история «что действовало» остаётся, как у expire_timetable).
    Возвращает, что сделано."""
    row = db.query_one("SELECT * FROM timetable WHERE id=?", (int(row_id),))
    if row is None:
        return ""
    if row["date"]:
        db.execute("DELETE FROM timetable WHERE id=?", (int(row_id),))
        return "удалено"
    last = _iso(on - timedelta(days=1))
    if last < row["valid_from"]:
        db.execute("DELETE FROM timetable WHERE id=?", (int(row_id),))
        return "серия удалена целиком"
    db.execute("UPDATE timetable SET valid_to=? WHERE id=?", (last, int(row_id)))
    return f"серия закончена {last}"


EXTEND_PREFIX = "extend|"


def _reference_week(rows: List[Dict[str, Any]]) -> Optional[date]:
    """Понедельник недели-образца: из последних трёх недель с разовыми
    событиями — самая полная (неполная последняя неделя не станет шаблоном)."""
    by_week: Dict[date, int] = {}
    for r in rows:
        d = datetime.strptime(r["date"], "%Y-%m-%d").date()
        monday = d - timedelta(days=d.weekday())
        by_week[monday] = by_week.get(monday, 0) + 1
    if not by_week:
        return None
    recent = sorted(by_week)[-3:]
    return max(recent, key=lambda m: (by_week[m], m))


@dataclass
class ExtendResult:
    reference: Optional[date] = None
    first: Optional[date] = None
    until: Optional[date] = None
    series: List[Dict[str, Any]] = field(default_factory=list)
    breaks: List[Tuple[date, date]] = field(default_factory=list)
    error: str = ""


def extend_weekly(until: Optional[date], breaks: List[Tuple[date, date]] = (),
                  reference: Optional[date] = None, source: str = "text") -> ExtendResult:
    """Продлить расписание: неделя-образец повторяется каждую неделю после
    последней недели с разовыми событиями — до `until` включительно или без
    конца (until=None: пока он не скажет «стоп», см. stop_extension). Каникулы
    — дни отдыха, они снимают пары (см. _rows_for). Повторный вызов заменяет
    прошлое продление того же календаря, а не добавляет второе.

    Образец — разовые события (из календаря/скринов), не еженедельные строки:
    теннис и так каждую неделю."""
    result = ExtendResult(until=until, breaks=list(breaks))
    dated = [dict(r) for r in db.query(
        "SELECT * FROM timetable WHERE date IS NOT NULL AND kind<>'rest'"
        " AND origin NOT LIKE ? ORDER BY date, start", (EXTEND_PREFIX + "%",))]
    if reference:
        monday = reference - timedelta(days=reference.weekday())
    else:
        monday = _reference_week(dated)
    if monday is None:
        result.error = "в расписании нет разовых событий — продлевать нечего"
        return result
    week = [r for r in dated
            if monday <= datetime.strptime(r["date"], "%Y-%m-%d").date() <= monday + timedelta(days=6)]
    if not week:
        result.error = f"на неделе с {monday:%d.%m} событий нет — не из чего делать образец"
        return result
    last_dated = max(datetime.strptime(r["date"], "%Y-%m-%d").date() for r in dated)
    first_monday = last_dated - timedelta(days=last_dated.weekday()) + timedelta(days=7)
    if until is not None and until < first_monday:
        result.error = (f"расписание и так есть до {last_dated:%d.%m}, а продлить просили "
                        f"до {until:%d.%m}")
        return result
    origin = EXTEND_PREFIX + (week[0].get("origin") or "manual")
    created = now_local().isoformat(timespec="minutes")
    result.reference, result.first = monday, first_monday
    with db.transaction() as conn:
        conn.execute("DELETE FROM timetable WHERE origin=?", (origin,))
        for r in week:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            first = first_monday + timedelta(days=d.weekday())
            if until is not None and first > until:
                continue
            conn.execute(
                "INSERT INTO timetable(weekday, start, end, title, kind, valid_from, valid_to,"
                " source, created, location, origin) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (d.weekday(), r["start"], r["end"], r["title"], r["kind"], _iso(first),
                 _iso(until) if until else None, source, created, r.get("location") or "",
                 origin))
            result.series.append({**r, "first": first})
        for a, b in breaks:
            day = a
            while day <= b and (day - a).days <= 62:
                conn.execute(
                    "INSERT INTO timetable(weekday, start, end, title, kind, valid_from,"
                    " source, created, date, origin) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (day.weekday(), "", "", "Каникулы", "rest", _iso(day), source, created,
                     _iso(day), origin))
                day += timedelta(days=1)
    return result


def coverage() -> str:
    """До какой даты в расписании есть пары — одной строкой, из данных.

    01.10.2026: данные удалили, а реплика Iris «продлила до 12.02» осталась в
    истории, и на «продли» она ответила «уже продлено». Что лежит в расписании
    сейчас, агент берёт отсюда, а не из старых реплик."""
    last = db.query_one(
        "SELECT MAX(date) d FROM timetable WHERE date IS NOT NULL AND kind IN ('lecture','home_study')"
        " AND origin NOT LIKE ?", (EXTEND_PREFIX + "%",))
    ext = db.query(
        "SELECT valid_to FROM timetable WHERE origin LIKE ? AND date IS NULL",
        (EXTEND_PREFIX + "%",))
    breaks = [r["date"] for r in db.query(
        "SELECT date FROM timetable WHERE origin LIKE ? AND kind='rest' ORDER BY date",
        (EXTEND_PREFIX + "%",))]
    if not (last and last["d"]) and not ext:
        return ""
    fmt = lambda s: datetime.strptime(s, "%Y-%m-%d").strftime("%d.%m.%Y")  # noqa: E731
    parts = [f"пары по датам из календаря — по {fmt(last['d'])}" if last and last["d"]
             else "пар по датам нет"]
    if ext:
        ends = [r["valid_to"] for r in ext]
        parts.append("продлено без конца" if None in ends
                     else f"продлено еженедельно до {fmt(max(ends))}")
        if breaks:
            parts.append(f"без пар {fmt(breaks[0])}–{fmt(breaks[-1])}")
    else:
        parts.append("не продлено")
    return "Расписание пар (из данных): " + "; ".join(parts) + "."


def stop_extension(last_day: date) -> int:
    """«Стоп, дальше без пар»: продлённые серии заканчиваются last_day включительно.
    Возвращает, сколько серий закончено."""
    cur = db.execute(
        "UPDATE timetable SET valid_to=? WHERE origin LIKE ? AND date IS NULL"
        " AND (valid_to IS NULL OR valid_to>?)",
        (_iso(last_day), EXTEND_PREFIX + "%", _iso(last_day)))
    db.execute("DELETE FROM timetable WHERE origin LIKE ? AND date>?",
               (EXTEND_PREFIX + "%", _iso(last_day)))
    return cur.rowcount or 0


def remove_origin(origin: str) -> int:
    """Убрать всё, что записано из одного источника, вместе с его продлением."""
    cur = db.execute("DELETE FROM timetable WHERE origin=? OR origin=?",
                     (origin, EXTEND_PREFIX + origin))
    return cur.rowcount or 0


@dataclass
class EventImportResult:
    added: int = 0
    updated: int = 0
    removed: int = 0  # были в прошлой выгрузке этого календаря, в новой нет


def import_events(events: List[Dict[str, Any]], origin: str, source: str,
                  date_from: date, date_to: date) -> EventImportResult:
    """Разовые события одного календаря за [date_from, date_to].

    Повторная выгрузка того же календаря — снимок, а не добавка: события с тем
    же uid обновляются, а исчезнувшие из выгрузки (отменённая пара) удаляются —
    но только в пределах периода этой выгрузки и только этого календаря
    (origin), чтобы файл с работы не стёр пары. events: date (YYYY-MM-DD),
    start, end, title, kind, location, uid.
    """
    result = EventImportResult()
    created = now_local().isoformat(timespec="minutes")
    uids = [str(e["uid"]) for e in events if e.get("uid")]
    with db.transaction() as conn:
        placeholders = ",".join("?" * len(uids)) or "''"
        cur = conn.execute(
            f"DELETE FROM timetable WHERE origin=? AND date BETWEEN ? AND ?"
            f" AND (uid IS NULL OR uid NOT IN ({placeholders}))",
            (origin, _iso(date_from), _iso(date_to), *uids),
        )
        result.removed = cur.rowcount or 0
        for e in events:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
            values = (d.weekday(), e.get("start") or "", e.get("end") or "",
                      str(e["title"]).strip()[:200], e["kind"], e["date"], source,
                      e["date"], str(e.get("location") or "").strip()[:200])
            row = conn.execute("SELECT id FROM timetable WHERE origin=? AND uid=?",
                               (origin, e.get("uid"))).fetchone() if e.get("uid") else None
            if row is not None:
                conn.execute(
                    "UPDATE timetable SET weekday=?, start=?, end=?, title=?, kind=?,"
                    " valid_from=?, source=?, date=?, location=? WHERE id=?",
                    (*values, row["id"]))
                result.updated += 1
            else:
                conn.execute(
                    "INSERT INTO timetable(weekday, start, end, title, kind, valid_from,"
                    " source, date, location, created, uid, origin)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*values, created, e.get("uid"), origin))
                result.added += 1
    return result


def is_home_study_day(d: date) -> bool:
    """День домашней учёбы — по данным, а не по вшитому кортежу weekday'ев."""
    return bool(_rows_for(d, kind="home_study"))


def set_timetable(entries: List[Tuple[int, str, str, str, str]], valid_from: str,
                  valid_to: Optional[str] = None, source: str = "manual") -> int:
    """Записать расписание на интервал. entries: (weekday, start, end, title, kind).

    Старое НЕ удаляем: у него свой интервал, и история «что действовало весной»
    должна оставаться отвечаемой — ровно как у смен в shift_events.
    """
    created = now_local().isoformat(timespec="minutes")
    n = 0
    with db.transaction() as conn:
        for weekday, start, end, title, kind in entries:
            conn.execute(
                "INSERT INTO timetable(weekday, start, end, title, kind,"
                " valid_from, valid_to, source, created) VALUES(?,?,?,?,?,?,?,?,?)",
                (int(weekday), start or "", end or "", title,
                 kind or "lecture", valid_from, valid_to, source, created),
            )
            n += 1
    return n


def expire_timetable(valid_to: str, source: str = "manual") -> int:
    """Закрыть всё бессрочно действующее расписание датой valid_to.

    Это «у меня каникулы» на языке данных: строки остаются, но перестают
    действовать со следующего дня.
    """
    cur = db.execute(
        "UPDATE timetable SET valid_to=?, source=? WHERE valid_to IS NULL AND date IS NULL",
        (valid_to, source),
    )
    return cur.rowcount or 0


def timetable_rows() -> List[Dict[str, Any]]:
    """Всё расписание целиком (для диагностики и ответа «что у меня записано»)."""
    return [dict(r) for r in db.query(
        "SELECT * FROM timetable ORDER BY valid_from, weekday, start")]


# Границы семестра, с которыми переносится старая константа. Обе — допущения,
# и обе правятся одним UPDATE, потому что теперь это данные:
#   SEED_VALID_FROM — типичное начало SoSe, точная дата не сохранилась нигде;
#   SEED_VALID_TO   — день перед тем, как владелец сказал «семестрфериен»
#                     (17.08.2026 12:01, запись в дневнике с тегом «учёба»).
SEED_VALID_FROM = "2026-04-01"
SEED_VALID_TO = "2026-08-16"


def seed_timetable_if_empty() -> int:
    """Разовый перенос расписания из кода в данные. Идемпотентен.

    Переносим СРАЗУ закрытым: семестр кончился, владелец в каникулах. Иначе
    первый же запуск после этой правки снова начал бы звать на пары.
    """
    row = db.query_one("SELECT COUNT(*) AS c FROM timetable")
    if row and int(row["c"]) > 0:
        return 0
    entries = [(weekday, start, end, title, kind)
               for weekday, slots in _SEED_TIMETABLE.items()
               for start, end, title, kind in slots]
    n = set_timetable(entries, valid_from=SEED_VALID_FROM,
                      valid_to=SEED_VALID_TO, source="seed")
    logger.info(
        "Расписание перенесено из кода в таблицу timetable: %d строк, "
        "срок действия %s — %s (закрыто, владелец в каникулах). "
        "Границы — допущения, правятся через set_timetable/expire_timetable.",
        n, SEED_VALID_FROM, SEED_VALID_TO,
    )
    return n


# ---------- смены: storage ----------

ACTIVE_SHIFT_STATUSES = {"planned", "confirmed", "uncertain", "moved"}


def _is_active_shift(shift: Optional[Dict[str, Any]]) -> bool:
    return bool(shift and shift.get("start") and shift.get("end")
                and shift.get("status", "planned") in ACTIVE_SHIFT_STATUSES)


def _shift_row(r) -> Dict[str, Any]:
    """Строка таблицы → словарь в форме старого shifts.json (+ id строки).
    Пустые поля опускаем: у вызывающих есть проверки на их отсутствие."""
    out: Dict[str, Any] = {
        "id": r["id"], "date": r["date"],
        "start": r["start"], "end": r["end"], "status": r["status"],
        "source": r["source"], "confidence": r["confidence"],
        "updated": r["updated"],
    }
    if r["last_confirmed_at"]:
        out["last_confirmed_at"] = r["last_confirmed_at"]
    if r["note"]:
        out["note"] = r["note"]
    return out


def _minutes(hm: Any) -> Optional[int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", str(hm or "").strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _interval(start: Any, end: Any) -> Optional[Tuple[int, int]]:
    """Минуты от начала дня; конец не позже начала = через полночь (16:00–00:00)."""
    s, e = _minutes(start), _minutes(end)
    if s is None or e is None:
        return None
    return (s, e + 24 * 60 if e <= s else e)


def _overlap(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> int:
    if a is None or b is None:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def get_shift_records(d: date) -> List[Dict[str, Any]]:
    """Все записи смен на дату (вкл. отменённые), по времени начала."""
    return [_shift_row(r) for r in db.query(
        "SELECT * FROM shifts WHERE date=? ORDER BY start, id", (_iso(d),))]


def get_shifts(d: date) -> List[Dict[str, Any]]:
    """Действующие смены на дату, по времени начала."""
    return [s for s in get_shift_records(d) if _is_active_shift(s)]


def _relevant(records: List[Dict[str, Any]], at: Optional[datetime]) -> Optional[Dict[str, Any]]:
    """Из нескольких смен дня — та, что важна сейчас: идущая, иначе ближайшая
    впереди, иначе последняя. Без момента времени — первая по началу."""
    if not records:
        return None
    if at is None:
        return records[0]
    now_min = at.hour * 60 + at.minute
    upcoming = None
    for r in records:
        iv = _interval(r.get("start"), r.get("end"))
        if iv is None:
            continue
        if iv[0] <= now_min < iv[1]:
            return r
        if iv[0] > now_min and upcoming is None:
            upcoming = r
    return upcoming or records[-1]


def get_shift_record(d: date, at: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Запись смены дня, важная в момент `at` (см. _relevant), вкл. отменённые.
    Все смены дня — get_shift_records()."""
    return _relevant(get_shift_records(d), at)


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


def get_shift(d: date, at: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """Действующая смена дня, важная в момент `at`. Все смены дня — get_shifts()."""
    return _relevant(get_shifts(d), at)


def save_shifts(items: List[Dict[str, Any]]) -> int:
    """Совместимая обёртка: только количество сохранённого.
    Конфликты видит apply_shifts — используй его, если надо о них сообщить."""
    return apply_shifts(items).saved


def apply_shifts(items: List[Dict[str, Any]]) -> ShiftApplyResult:
    """Merge смен с разбором конфликтов.

    Backward-compatible: callers may still pass only date/start/end. Optional
    metadata lets text corrections and photo imports carry status/source/confidence.

    Приоритет источников (с 15.08.2026): запись с меньшим приоритетом не
    перетирает существующую — фото не стирает правку текстом. Но и молча
    отклонять нельзя: Влад об этом не узнает и будет думать, что график
    обновился. Поэтому по умолчанию (policy='ask') такие случаи возвращаются
    как конфликты, и вызывающий спрашивает. Заранее заданная политика
    ('keep_mine'/'photo_wins') снимает вопрос.

    Какую запись правит пункт (с v9, несколько смен в день): ту, что
    пересекается с ним по времени (план 17:00 → факт 17:14 — та же смена).
    Не пересекается ни с одной — это ещё одна смена в этот день, а не замена:
    до 01.10.2026 любая запись на дату затирала предыдущую. Перенос на
    непересекающиеся часы — пункт с replaces=True (сказано «перенесли»).
    Пункт без часов (только статус: «сегодня не иду») относится ко всем
    действующим сменам дня.
    """
    result = ShiftApplyResult()
    policy = get_conflict_policy()
    updated = now_local().isoformat(timespec="minutes")
    with db.transaction() as conn:
        for it in items:
            d, start, end = it.get("date"), it.get("start"), it.get("end")
            if not d or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
                continue
            rows = [_shift_row(r) for r in conn.execute(
                "SELECT * FROM shifts WHERE date=? ORDER BY start, id", (d,))]
            for prev in _targets(rows, it) or [{}]:
                _apply_one(conn, it, prev, policy, updated, result)

    # Незакрытые конфликты держим до ответа Влада: без этого его «бери с фото»
    # применять не к чему — разобранные смены к тому моменту уже забыты.
    if result.conflicts:
        db.kv_set(PENDING_CONFLICTS_KEY, {
            "asked_at": updated,
            "items": [c.incoming for c in result.conflicts],
        })
    return result


def _targets(rows: List[Dict[str, Any]], it: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Записи дня, которые правит пункт. [] — новая смена."""
    iv = _interval(it.get("start"), it.get("end"))
    active = [r for r in rows if _is_active_shift(r)]
    if iv is None:
        # Только статус, без часов: относится к действующим сменам дня; нет
        # их — к отметке «отменена» без часов, если она уже есть.
        return active or [r for r in rows if not (r.get("start") and r.get("end"))][:1]
    best = max(rows, key=lambda r: _overlap(iv, _interval(r.get("start"), r.get("end"))),
               default=None)
    if best is not None and _overlap(iv, _interval(best.get("start"), best.get("end"))) > 0:
        return [best]
    if it.get("replaces") and len(active) == 1:
        return active
    # Отметка «смена отменена» без часов — дневной факт: смена с часами его заменяет.
    return [r for r in rows if not (r.get("start") and r.get("end"))][:1]


def _apply_one(conn, it: Dict[str, Any], prev: Dict[str, Any], policy: str,
               updated: str, result: ShiftApplyResult) -> None:
    """Один пункт против одной записи (prev={} — новая смена)."""
    d = it["date"]
    start = it.get("start") or prev.get("start")
    end = it.get("end") or prev.get("end")
    status = str(it.get("status") or prev.get("status") or "planned").strip().lower()
    source = str(it.get("source") or prev.get("source")
                 or ("text" if status == "cancelled" else "unknown"))

    # Одинаковые значения конфликтом не считаем: фото, подтверждающее
    # уже известную смену, ничего не портит и вопроса не стоит.
    if prev and _priority(source) < _priority(prev.get("source", "")):
        same = (prev.get("start") == start and prev.get("end") == end
                and prev.get("status") == status)
        if not same and policy != "photo_wins":
            conn.execute(
                "INSERT INTO shift_events(ts, date, action, source, payload, reason)"
                " VALUES(?,?,?,?,?,?)",
                (updated, d, "conflict" if policy == "ask" else "reject", source,
                 _dumps(dict(it)),
                 (f"{source} расходится с {prev.get('source')} "
                  f"({prev.get('start')}–{prev.get('end')}), политика: {policy}")),
            )
            logger.info("Смена %s: %s расходится с %s (политика %s)",
                        d, source, prev.get("source"), policy)
            if policy == "ask":
                result.conflicts.append(
                    ShiftConflict(date=d, incoming=dict(it), existing=dict(prev)))
            return

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
            return
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

    values = (record.get("start"), record.get("end"), record["status"], record["source"],
              record["confidence"], record["updated"], record.get("last_confirmed_at"),
              record.get("note"))
    if prev.get("id"):
        conn.execute(
            "UPDATE shifts SET start=?, end=?, status=?, source=?, confidence=?,"
            " updated=?, last_confirmed_at=?, note=? WHERE id=?", (*values, prev["id"]))
    else:
        conn.execute(
            "INSERT INTO shifts(start, end, status, source, confidence, updated,"
            " last_confirmed_at, note, date) VALUES(?,?,?,?,?,?,?,?,?)", (*values, d))
    record.pop("id", None)
    record.pop("date", None)
    conn.execute(
        "INSERT INTO shift_events(ts, date, action, source, payload, reason)"
        " VALUES(?,?,?,?,?,?)",
        (updated, d, "cancel" if status == "cancelled" else "set",
         record["source"], _dumps(record), None),
    )
    result.saved += 1
    result.applied.append({"date": d, "start": record.get("start"),
                           "end": record.get("end"), "status": record["status"]})


def pending_conflicts() -> List[Dict[str, Any]]:
    """Смены, по которым задан вопрос и ответа ещё не было."""
    data = db.kv_get(PENDING_CONFLICTS_KEY, {}) or {}
    return list(data.get("items") or [])


def resolve_pending_conflicts(decision: str, remember: bool = False) -> int:
    """Ответ Влада на вопрос о расхождении.

    decision: 'photo' — принять то, что в графике; 'mine' — оставить свою правку.
    remember=True запоминает решение политикой, и дальше вопрос не задаётся.
    Возвращает количество применённых смен.
    """
    decision = str(decision or "").strip().lower()
    items = pending_conflicts()
    db.kv_set(PENDING_CONFLICTS_KEY, {})
    if remember:
        set_conflict_policy("photo_wins" if decision == "photo" else "keep_mine")
    if decision != "photo" or not items:
        for it in items:
            log_shift_event(it.get("date", ""), "reject", it.get("source", "photo"),
                            it, reason="Влад оставил свою версию")
        return 0
    # Явное решение владельца — это уже не догадка машины, поэтому применяем
    # с источником manual: следующее фото его тоже не перетрёт.
    return apply_shifts([{**it, "source": "manual"} for it in items]).saved


def describe_conflicts(conflicts: List[ShiftConflict]) -> str:
    """Вопрос для чата: что из этого верно."""
    lines = [c.describe() for c in conflicts]
    tail = ("Что верно? Скажи «бери с фото» или «оставь как есть» — "
            "и добавь «всегда», если решать так же и дальше.")
    return "Тут расхождение с тем, что ты говорил:\n• " + "\n• ".join(lines) + "\n\n" + tail


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def describe_saved_shifts(items: List[Dict[str, str]], n: int) -> str:
    """Человекочитаемое подтверждение сохранённых смен для чата.
    items — ShiftApplyResult.applied: то, что легло в базу, а не то, что разобрали."""
    lines = []
    for it in sorted(items, key=lambda x: (x.get("date", ""), x.get("start") or "")):
        try:
            d = datetime.strptime(it["date"], "%Y-%m-%d").date()
            lines.append(f"• {_DAY_NAMES[d.weekday()]} {d.strftime('%d.%m')}: {it['start']}–{it['end']}")
        except (KeyError, ValueError):
            continue
    return (
        f"Принял график, смен сохранено: {n}\n\n" + "\n".join(lines) +
        "\n\nЕсли что-то распознал криво — поправь текстом."
    )


def describe_shift(s: Dict[str, Any]) -> str:
    if not (s.get("start") and s.get("end")):
        return "смена отменена"
    tail = " отменена" if s.get("status") == "cancelled" else ""
    return f"смена {s['start']}–{s['end']}{tail}"


def describe_event(r: Dict[str, Any], with_id: bool = False) -> str:
    """«14:05–15:45 Diskrete Mathematik» / «20:00–21:30 Tennis (спорт, еженедельно)»."""
    when = f"{r['start']}–{r['end']}" if r.get("start") and r.get("end") else "весь день"
    notes = [] if r.get("kind") == "lecture" else [KIND_LABELS.get(r.get("kind"), r.get("kind"))]
    if not r.get("date"):
        notes.append("еженедельно")
    tail = f" ({', '.join(notes)})" if notes else ""
    ref = f" #{r['id']}" if with_id and r.get("id") else ""
    return f"{when} {r['title']}{tail}{ref}"


def format_week(days: int = 8) -> str:
    """Расписание на N дней вперёд: все смены дня + пары, спорт, встречи.
    #id у событий — чтобы инструмент мог убрать именно это."""
    out: List[str] = []
    today = now_local().date()
    for i in range(days):
        d = today + timedelta(days=i)
        parts: List[str] = []
        for s in get_shift_records(d):
            text = describe_shift(s)
            if _is_active_shift(s):
                text += f" (дорога ~{WORK_COMMUTE_MIN} мин)"
            parts.append(text)
        parts += [describe_event(r, with_id=True) for r in day_events(d)]
        label = f"{_DAY_NAMES[d.weekday()]} {d.strftime('%d.%m')}"
        out.append(f"{label}: " + ("; ".join(parts) if parts else "свободен"))
    return "\n".join(out)
