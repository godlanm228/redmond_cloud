"""
Coach storage — цели, дедлайны, дневник, запас, состояние дня.

Хранилище — SQLite (`utils/db.py`), с 15.08.2026. До этого были JSON-файлы в
data/coach/, и разбор 12–13.08 нашёл там не отдельные баги, а свойства формата:
  • «прочитал → поменял → записал» без транзакции терял параллельные записи;
  • запись не атомарна — падение посреди write оставляло обрубок;
  • битый файл молча превращался в пустой (дневник из 84 записей → одна);
  • истории изменений не было: «откуда взялась эта смена» не ответить.

Публичный API не менялся: вызывающие (tools, pings, scheduler, week_schedule)
работают как раньше. Старые JSON остаются на диске замороженной копией —
перенос делает utils/migrate_json_to_db.py и ничего не удаляет.

Формат значений сохранён как в JSON: id сквозные, timestamp — ISO-строка,
tags — список. Это позволяет прогнать старые тесты как проверку на дрейф.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)


def _json_load(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


# id больше НЕ вычисляется чтением max+1: два потока успевали получить одно и
# то же число и второй падал с UNIQUE constraint failed. Вставляем с id=NULL —
# SQLite присваивает rowid сам, атомарно, и семантика та же (max+1).


# ============================================================================
# Goals
# ============================================================================

def _goal_row(r) -> Dict[str, Any]:
    return {
        "id": r["id"], "title": r["title"], "why": r["why"],
        "target_date": r["target_date"], "status": r["status"],
        "created": r["created"], "closed": r["closed"],
        "progress_log": _json_load(r["progress_log"], []),
    }


def list_goals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        rows = db.query("SELECT * FROM goals WHERE status=? ORDER BY id", (status,))
    else:
        rows = db.query("SELECT * FROM goals ORDER BY id")
    return [_goal_row(r) for r in rows]


def add_goal(title: str, why: str = "", target_date: Optional[str] = None) -> Dict[str, Any]:
    goal = {
        "id": None,
        "title": title.strip(),
        "why": why.strip(),
        "target_date": target_date,
        "status": "active",
        "created": now_local().strftime("%Y-%m-%d"),
        "progress_log": [],
    }
    cur = db.execute(
        "INSERT INTO goals(id, title, why, target_date, status, created, progress_log)"
        " VALUES(NULL,?,?,?,?,?,?)",
        (goal["title"], goal["why"], goal["target_date"],
         goal["status"], goal["created"], "[]"),
    )
    goal["id"] = cur.lastrowid
    return goal


def mark_goal_done(goal_id: int, note: str = "") -> Optional[Dict[str, Any]]:
    row = db.query_one("SELECT * FROM goals WHERE id=?", (goal_id,))
    if row is None:
        return None
    goal = _goal_row(row)
    goal["status"] = "done"
    goal["closed"] = now_local().strftime("%Y-%m-%d")
    if note:
        goal["progress_log"].append(
            {"date": now_local().strftime("%Y-%m-%d"), "note": note})
    db.execute(
        "UPDATE goals SET status=?, closed=?, progress_log=? WHERE id=?",
        (goal["status"], goal["closed"],
         json.dumps(goal["progress_log"], ensure_ascii=False), goal_id),
    )
    return goal


# ============================================================================
# Deadlines
# ============================================================================

def _deadline_row(r) -> Dict[str, Any]:
    out = {
        "id": r["id"], "title": r["title"], "due": r["due"],
        "importance": r["importance"], "status": r["status"],
        "created": r["created"],
    }
    # closed/note появляются только когда заполнены — как было в JSON,
    # иначе промпт Iris получает мусорные "closed": null у всех дедлайнов.
    if r["closed"]:
        out["closed"] = r["closed"]
    if r["note"]:
        out["note"] = r["note"]
    return out


def list_deadlines(upcoming_days: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = [_deadline_row(r) for r in db.query("SELECT * FROM deadlines ORDER BY id")]
    if upcoming_days is None:
        return rows
    today = now_local().date()
    cutoff = today + timedelta(days=upcoming_days)
    result = []
    for d in rows:
        try:
            due = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= due <= cutoff:
            result.append(d)
    return result


def add_deadline(title: str, due: str, importance: str = "medium") -> Dict[str, Any]:
    """due: YYYY-MM-DD. importance: low/medium/high."""
    deadline = {
        "id": None,
        "title": title.strip(),
        "due": due,
        "importance": importance,
        "status": "pending",
        "created": now_local().strftime("%Y-%m-%d"),
    }
    cur = db.execute(
        "INSERT INTO deadlines(id, title, due, importance, status, created)"
        " VALUES(NULL,?,?,?,?,?)",
        (deadline["title"], deadline["due"],
         deadline["importance"], deadline["status"], deadline["created"]),
    )
    deadline["id"] = cur.lastrowid
    return deadline


def mark_deadline_done(deadline_id: int) -> Optional[Dict[str, Any]]:
    """Закрыть дедлайн (сдал/прошло). Без этого сданный тест вечно висит
    в TOP PRIORITIES и Iris продолжает пушить."""
    row = db.query_one("SELECT * FROM deadlines WHERE id=?", (deadline_id,))
    if row is None:
        return None
    closed = now_local().strftime("%Y-%m-%d")
    db.execute("UPDATE deadlines SET status='done', closed=? WHERE id=?",
               (closed, deadline_id))
    out = _deadline_row(row)
    out["status"] = "done"
    out["closed"] = closed
    return out


def delete_deadline(deadline_id: int) -> Optional[Dict[str, Any]]:
    """Удалить дедлайн НАСОВСЕМ («удали/убери — не нужен»). Не путать с
    mark_deadline_done: done = «сдал/прошло», остаётся в истории; delete —
    ошибочный/неактуальный исчезает и статистику не портит (кейс 02.08:
    «Удали его» превратился в done, потому что удалять было нечем)."""
    row = db.query_one("SELECT * FROM deadlines WHERE id=?", (deadline_id,))
    if row is None:
        return None
    db.execute("DELETE FROM deadlines WHERE id=?", (deadline_id,))
    return _deadline_row(row)


def update_deadline(
    deadline_id: int,
    due: Optional[str] = None,
    title: Optional[str] = None,
    importance: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Правка существующего дедлайна (перенос даты и т.п.). «Перенесём на неделю»
    = ЭТО, а не add_deadline: новый рядом со старым pending = дубль, и Iris
    долбит по обоим (кейс #3/#4/#5 «Матан», июль 2026)."""
    row = db.query_one("SELECT * FROM deadlines WHERE id=?", (deadline_id,))
    if row is None:
        return None
    out = _deadline_row(row)
    if due:
        out["due"] = due
    if title:
        out["title"] = title.strip()
    if importance:
        out["importance"] = importance
    db.execute("UPDATE deadlines SET due=?, title=?, importance=? WHERE id=?",
               (out["due"], out["title"], out["importance"], deadline_id))
    return out


# ============================================================================
# Diary
# ============================================================================

def _diary_row(r) -> Dict[str, Any]:
    out = {"id": r["id"], "timestamp": r["ts"], "text": r["text"],
           "tags": _json_load(r["tags"], [])}
    data = _json_load(r["data"], None)
    if data:
        out["data"] = data
    return out


def add_diary_entry(
    text: str,
    tags: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Запись в дневник. Возвращает None если писать нечего (пустышка) или это
    точный дубль последней записи — анти-шум (Iris логировала мета/повторы).

    data — опциональная структурная нагрузка (напр. еда: dish/kcal/protein/place).
    Хранится в самой записи, чтобы аналитический слой (Этап 3) агрегировал тренды
    без отдельного meals.json. Тег [питание] при этом сохраняется для today_tags()."""
    text = (text or "").strip()
    if len(text) < 3:
        return None
    last = db.query_one("SELECT * FROM diary ORDER BY id DESC LIMIT 1")
    if last is not None and str(last["text"]).strip().lower() == text.lower():
        return _diary_row(last)
    entry = {
        "id": None,
        "timestamp": now_local().isoformat(timespec="minutes"),
        "text": text,
        "tags": tags or [],
    }
    if data:
        entry["data"] = data
    cur = db.execute(
        "INSERT INTO diary(id, ts, text, tags, data) VALUES(NULL,?,?,?,?)",
        (entry["timestamp"], entry["text"],
         json.dumps(entry["tags"], ensure_ascii=False),
         json.dumps(data, ensure_ascii=False) if data else None),
    )
    entry["id"] = cur.lastrowid
    return entry


def read_diary(last_n: int = 10, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [_diary_row(r) for r in db.query("SELECT * FROM diary ORDER BY id")]
    if tag:
        rows = [d for d in rows if tag in (d.get("tags") or [])]
    return rows[-last_n:] if last_n else rows


def delete_diary_entries(ids: List[int]) -> List[int]:
    """Удалить записи дневника по id. Возвращает РЕАЛЬНО удалённые id
    (чтобы Iris отчитывалась только о фактически снесённом, а не врала)."""
    id_set = {int(i) for i in ids}
    if not id_set:
        return []
    marks = ",".join("?" * len(id_set))
    existing = [r["id"] for r in
                db.query(f"SELECT id FROM diary WHERE id IN ({marks})", tuple(id_set))]
    if existing:
        db.execute(f"DELETE FROM diary WHERE id IN ({marks})", tuple(id_set))
    return existing


def last_entry_per_tag(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """Самая свежая запись дневника на каждый из тегов (любой давности).
    Чтобы Iris не отвечала «нет записей» о спорте/еде, когда они есть —
    последнее по теме инжектится в её STATE-блок."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in db.query("SELECT * FROM diary ORDER BY id"):
        entry = _diary_row(r)
        for t in entry.get("tags") or []:
            if t in tags:
                out[t] = entry
    return out


def today_tags() -> set:
    """Все теги дневника за сегодня — тикер проверяет закрыт ли слот (питание/спорт/…)."""
    today = now_local().strftime("%Y-%m-%d")
    tags: set = set()
    for r in db.query("SELECT tags FROM diary WHERE ts LIKE ?", (f"{today}%",)):
        tags.update(_json_load(r["tags"], []))
    return tags


def entries_today() -> int:
    """Сколько записей дневника за сегодня. Вечерний итог при 0 записей и
    отсутствовавшем Владе молчит — не спамит «день получился спокойный»."""
    today = now_local().strftime("%Y-%m-%d")
    row = db.query_one("SELECT COUNT(*) AS c FROM diary WHERE ts LIKE ?", (f"{today}%",))
    return int(row["c"])


# ============================================================================
# Pantry — запас продуктов (инкрементальный, НЕ снапшот-перезапись и НЕ граммы).
# Items = строки. Обновляется при покупке/готовке. updated → мягкая ресинхронизация.
# ============================================================================

def _norm_item(s: Any) -> str:
    return " ".join(str(s).lower().split())


def get_pantry() -> Dict[str, Any]:
    rows = db.query("SELECT item, added FROM pantry ORDER BY rowid")
    if not rows:
        return {"items": [], "updated": None}
    return {"items": [r["item"] for r in rows],
            "updated": max((r["added"] for r in rows if r["added"]), default=None)}


def pantry_update(add: Optional[List[str]] = None,
                  remove: Optional[List[str]] = None) -> Dict[str, Any]:
    """Инкрементально: добавить купленное / убрать потраченное. Дедуп по
    нормализованному имени, порядок добавления сохраняется."""
    today = now_local().strftime("%Y-%m-%d")
    with db.transaction() as conn:
        have = {_norm_item(r["item"]): r["item"]
                for r in conn.execute("SELECT item FROM pantry")}
        for it in (add or []):
            it = str(it).strip()
            if it and _norm_item(it) not in have:
                conn.execute("INSERT INTO pantry(item, added) VALUES(?,?)", (it, today))
                have[_norm_item(it)] = it
        for r in (remove or []):
            original = have.pop(_norm_item(r), None)
            if original is not None:
                conn.execute("DELETE FROM pantry WHERE item=?", (original,))
        # updated = дата последней операции, включая удаление: «запас трогали
        # сегодня» — это и про израсходованное тоже.
        conn.execute("UPDATE pantry SET added=?", (today,))
    return get_pantry()


def pantry_age_days() -> Optional[int]:
    """Сколько дней назад обновляли запас (для мягкой ресинхронизации). None — пусто."""
    upd = get_pantry().get("updated")
    if not upd:
        return None
    try:
        d = datetime.strptime(upd, "%Y-%m-%d").date()
        return (now_local().date() - d).days
    except (ValueError, TypeError):
        return None


# ============================================================================
# Week plan — текст плана недели от Iris (составляется при загрузке смен
# или по запросу «составь план недели», правится словами через чат)
# ============================================================================

def get_week_plan() -> Dict[str, Any]:
    return db.kv_get("week_plan", {}) or {}


def save_week_plan(text: str) -> Dict[str, Any]:
    plan = {
        "updated": now_local().isoformat(timespec="minutes"),
        "text": text.strip(),
    }
    db.kv_set("week_plan", plan)
    return plan


# ============================================================================
# Presence — фиксация «проснулся» по первому сообщению дня
# ============================================================================

# Окно пробуждения: первое сообщение Влада в этом интервале считается подъёмом.
# Сообщения 00:00–05:00 — ночные посиделки, не подъём.
_WAKE_WINDOW = (5, 15)  # часы, [from, to). До 15: ловим поздние подъёмы (бар → встаёт поздно)


def log_wake_if_first() -> Optional[Dict[str, Any]]:
    """
    Вызывается на каждом сообщении владельца. Если это первое сообщение
    сегодня в окне пробуждения — пишет запись в дневник (тег «сон»)
    и возвращает её; иначе None. Идемпотентно по дате.
    """
    now = now_local()
    if not (_WAKE_WINDOW[0] <= now.hour < _WAKE_WINDOW[1]):
        return None

    today = now.strftime("%Y-%m-%d")
    presence = db.kv_get("presence", {}) or {}
    if presence.get("last_wake_date") == today:
        return None

    presence["last_wake_date"] = today
    presence["wake_time"] = now.strftime("%H:%M")
    db.kv_set("presence", presence)
    return add_diary_entry(
        f"Проснулся — первое сообщение в {presence['wake_time']}",
        tags=["сон"],
    )


def woke_today() -> bool:
    presence = db.kv_get("presence", {}) or {}
    return presence.get("last_wake_date") == now_local().strftime("%Y-%m-%d")


def wake_time_today() -> Optional[str]:
    """«HH:MM» пробуждения, если зафиксировано сегодня."""
    presence = db.kv_get("presence", {}) or {}
    if presence.get("last_wake_date") == now_local().strftime("%Y-%m-%d"):
        return presence.get("wake_time")
    return None


# ============================================================================
# Day state — анти-спам для проактивных пингов (тикер)
# ============================================================================

def get_day_state() -> Dict[str, Any]:
    """State за сегодня: какие пинги отправлены. Авто-сброс на новой дате.
    (Тишина-по-запросу живёт отдельно в mute — она кросс-день.)"""
    state = db.kv_get("day_state", {}) or {}
    today = now_local().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "pings": {}}
    state.setdefault("pings", {})
    return state


def save_day_state(state: Dict[str, Any]) -> None:
    db.kv_set("day_state", state)


def mark_owner_seen() -> None:
    """Влад написал в HUB — фиксируем «на связи» (per-day, day_state
    сбрасывается на новой дате). last_seen — ПЕРВОЕ сообщение дня (питает
    cold-start тикера), last_msg — ПОСЛЕДНЕЕ (питает backoff: пинги после
    last_msg без ответа = «ему сейчас не до меня», тикер отступает)."""
    state = get_day_state()
    now_hm = now_local().strftime("%H:%M")
    if not state.get("last_seen"):
        state["last_seen"] = now_hm
    state["last_msg"] = now_hm
    save_day_state(state)


def owner_seen_today() -> bool:
    return bool(get_day_state().get("last_seen"))


def mark_ping(ping_id: str) -> None:
    state = get_day_state()
    state["pings"][ping_id] = now_local().strftime("%H:%M")
    save_day_state(state)


# ============================================================================
# Mute — «стоп» от Влада. Два уровня (с 03.08.2026):
#   scope='pings' (дефолт) — молчит только дневной тикер (checkin/еда/спорт/
#     учёба); утренний дайджест, напоминания о дедлайнах и вечерний итог
#     ОСТАЮТСЯ — это информация, а не «дёрганье».
#   scope='all' — полная тишина всего проактивного (только по явной просьбе).
# Ответы на его собственные сообщения не блокируются никогда.
# Кросс-день, в отличие от per-day day_state.
# ============================================================================

def set_mute(mode: str = "today", hours: float = 0, scope: str = "pings") -> str:
    """Выключить проактивные сообщения. mode: 'today' (до конца дня, дефолт) /
    'forever' (пока явно не снимут) / часы через hours>0. scope: 'pings'/'all'.
    Возвращает человекочитаемое «до когда» для подтверждения."""
    now = now_local()
    data: Dict[str, Any] = {
        "set": now.isoformat(timespec="minutes"),
        "scope": "all" if str(scope).strip().lower() == "all" else "pings",
    }
    if mode == "forever":
        data["until"] = "forever"
        db.kv_set("mute", data)
        return "пока не скажешь «пиши»"
    if mode != "today" and hours and hours > 0:
        until = now + timedelta(hours=max(0.5, min(float(hours), 168.0)))
        data["until"] = until.isoformat(timespec="minutes")
        db.kv_set("mute", data)
        return ("до " + until.strftime("%H:%M")
                if until.date() == now.date()
                else "до " + until.strftime("%H:%M %d.%m"))
    until = now.replace(hour=23, minute=59, second=59, microsecond=0)
    data["until"] = until.isoformat(timespec="minutes")
    db.kv_set("mute", data)
    return "до конца дня"


def unmute() -> None:
    db.kv_set("mute", {})


def _mute_record_active() -> Optional[Dict[str, Any]]:
    data = db.kv_get("mute", {}) or {}
    until = data.get("until")
    if not until:
        return None
    if until == "forever":
        return data
    try:
        # Пишем сюда только aware-ISO из now_local() — сравнение корректно.
        return data if now_local() < datetime.fromisoformat(until) else None
    except (ValueError, TypeError):
        return None


def muted_now() -> bool:
    """Активен ли ЛЮБОЙ mute — гасит проактивные пинги дневного тикера."""
    return _mute_record_active() is not None


def hard_muted_now() -> bool:
    """Полная тишина (scope='all') — гасит и дайджесты/вечерний итог.
    Записи без scope считаем 'all': они ставились, когда mute был единственным
    и полным — смысл уже действующей просьбы Влада не меняем."""
    rec = _mute_record_active()
    return bool(rec) and str(rec.get("scope") or "all") == "all"


# ============================================================================
# Ротация стилей пингов, радар дедлайнов, счётчик Gemini
# ============================================================================

def next_style_index(n: int) -> int:
    """Ротация стилевых вариантов пингов (persist кросс-день) — чтобы Iris не
    открывала сообщения одинаково два раза подряд (жалоба 03.08: «формат
    крайне одинаковый и надоедает»)."""
    with db.transaction() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key='ping_style'").fetchone()
        data = _json_load(row["value"] if row else None, {})
        idx = (int(data.get("idx", -1)) + 1) % max(1, n)
        conn.execute(
            "INSERT INTO kv(key, value, updated) VALUES('ping_style',?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (json.dumps({"idx": idx}), now_local().isoformat(timespec="minutes")),
        )
    return idx


def radar_pinged(deadline_id: Any) -> bool:
    """Уже делали ранний «радар»-пинг по этому дедлайну? Persistent."""
    return str(deadline_id) in (db.kv_get("radar", {}) or {})


def mark_radar(deadline_id: Any) -> None:
    with db.transaction() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key='radar'").fetchone()
        data = _json_load(row["value"] if row else None, {})
        data[str(deadline_id)] = now_local().strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO kv(key, value, updated) VALUES('radar',?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (json.dumps(data, ensure_ascii=False),
             now_local().isoformat(timespec="minutes")),
        )


# Gemini RPD-гард — счётчик запросов за день (free-tier; пул общий
# с vision/поиском/дайджестом/Iris-петлёй). Авто-сброс на новой дате.
_GEMINI_RPD_WARN = (1200, 1450)  # пороги для одноразового warning в лог


def gemini_bump() -> int:
    """Инкремент под транзакцией: на JSON два параллельных вызова читали одно
    и то же значение и счётчик отставал от реальности."""
    today = now_local().strftime("%Y-%m-%d")
    with db.transaction() as conn:
        row = conn.execute("SELECT value FROM kv WHERE key='gemini_usage'").fetchone()
        data = _json_load(row["value"] if row else None, {})
        if data.get("date") != today:
            data = {"date": today, "count": 0}
        data["count"] = int(data.get("count", 0)) + 1
        conn.execute(
            "INSERT INTO kv(key, value, updated) VALUES('gemini_usage',?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (json.dumps(data), now_local().isoformat(timespec="minutes")),
        )
    if data["count"] in _GEMINI_RPD_WARN:
        logger.warning("Gemini RPD: %d запросов сегодня (free-tier лимит ~1500)",
                       data["count"])
    return data["count"]


def gemini_count_today() -> int:
    data = db.kv_get("gemini_usage", {}) or {}
    return int(data.get("count", 0)) if data.get("date") == now_local().strftime("%Y-%m-%d") else 0
