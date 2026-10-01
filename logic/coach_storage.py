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
import re
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
    # Под транзакцией: без неё два параллельных вызова читали один и тот же
    # progress_log и второй затирал заметку первого.
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        if row is None:
            return None
        goal = _goal_row(row)
        goal["status"] = "done"
        goal["closed"] = now_local().strftime("%Y-%m-%d")
        if note:
            goal["progress_log"].append(
                {"date": now_local().strftime("%Y-%m-%d"), "note": note})
        conn.execute(
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
    """Дедлайны. `upcoming_days` — окно вперёд; просроченное из него НЕ выпадает.

    Раньше фильтр был `today <= due <= cutoff`, то есть окно отрезало всё, что
    раньше сегодня. Этой функцией отвечает инструмент `list_deadlines` — и на
    вопрос «какие у меня дедлайны» владелец не видел именно тот, который
    пропустил. Скедулер при этом звал версию без окна и просрочку видел:
    система знала, а показать не могла.

    Кривая дата больше не проглатывается молча: запись остаётся видимой и
    сообщает о себе в лог. Прежде она лежала в базе и не показывалась нигде
    и никогда.

    Порядок: сначала просроченное (самое срочное), потом ближайшее.
    """
    rows = [_deadline_row(r) for r in db.query("SELECT * FROM deadlines ORDER BY id")]
    if upcoming_days is None:
        return rows

    today = now_local().date()
    cutoff = today + timedelta(days=upcoming_days)
    dated: List[Any] = []
    undated: List[Dict[str, Any]] = []
    for d in rows:
        if str(d.get("status") or "pending") != "pending":
            continue
        try:
            due = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            logger.warning(
                "Дедлайн #%s «%s»: дата «%s» не читается — показываю без срока, "
                "иначе он исчезнет из выдачи насовсем",
                d.get("id"), d.get("title"), d.get("due"))
            undated.append(d)
            continue
        if due <= cutoff:            # просроченное тоже проходит: due < today
            dated.append((due, d))
    dated.sort(key=lambda pair: pair[0])
    return [d for _, d in dated] + undated


def _same_task(a: str, b: str) -> bool:
    norm = lambda s: " ".join(re.findall(r"\w+", (s or "").lower()))  # noqa: E731
    return norm(a) == norm(b)


def find_open_duplicate(title: str, due: str) -> Optional[Dict[str, Any]]:
    """Открытый дедлайн с тем же названием и сроком ±1 день — это тот же.
    Июль 2026: «Матан: 6 открытых тестов» лёг трижды (#3, #4, #5)."""
    try:
        target = datetime.strptime(due, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    for d in list_deadlines():
        if d.get("status") != "pending" or not _same_task(d.get("title", ""), title):
            continue
        try:
            if abs((datetime.strptime(d["due"], "%Y-%m-%d").date() - target).days) <= 1:
                return d
        except (ValueError, TypeError):
            continue
    return None


def add_deadline(title: str, due: str, importance: str = "medium") -> Dict[str, Any]:
    """due: YYYY-MM-DD. importance: low/medium/high.
    Тот же открытый дедлайн (find_open_duplicate) не дублируется: возвращается
    существующий с пометкой duplicate=True."""
    existing = find_open_duplicate(title, due)
    if existing:
        return {**existing, "duplicate": True}
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
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM deadlines WHERE id=?",
                           (deadline_id,)).fetchone()
        if row is None:
            return None
        closed = now_local().strftime("%Y-%m-%d")
        conn.execute("UPDATE deadlines SET status='done', closed=? WHERE id=?",
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
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM deadlines WHERE id=?",
                           (deadline_id,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM deadlines WHERE id=?", (deadline_id,))
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
    # Под транзакцией: функция пишет ВСЕ три поля, поэтому без неё
    # параллельная правка заголовка и срока теряла одну из двух.
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM deadlines WHERE id=?",
                           (deadline_id,)).fetchone()
        if row is None:
            return None
        out = _deadline_row(row)
        if due:
            out["due"] = due
        if title:
            out["title"] = title.strip()
        if importance:
            out["importance"] = importance
        conn.execute(
            "UPDATE deadlines SET due=?, title=?, importance=? WHERE id=?",
            (out["due"], out["title"], out["importance"], deadline_id))
    return out


# ============================================================================
# Diary
# ============================================================================

# Пометки на данных (01.10.2026, правило владельца: данные не удаляются, а
# помечаются — иначе непонятно, актуальны они или нет). У записи дневника в
# data: source — кто записал (owner — его словами, кодом из сообщения; agent —
# агент инструментом; code — код сам, напр. «проснулся»); status — retracted,
# если запись оказалась ошибкой: она остаётся в истории с причиной, но не
# участвует ни в итогах, ни в «последнем по теме», ни в слотах пингов.
RETRACTED = "retracted"


def _diary_row(r) -> Dict[str, Any]:
    out = {"id": r["id"], "timestamp": r["ts"], "text": r["text"],
           "tags": _json_load(r["tags"], [])}
    data = _json_load(r["data"], None)
    if data:
        out["data"] = data
    return out


def is_retracted(entry: Dict[str, Any]) -> bool:
    return (entry.get("data") or {}).get("status") == RETRACTED


def retract_diary_entry(entry_id: int, reason: str) -> Optional[Dict[str, Any]]:
    """Пометить запись ошибочной, не удаляя: текст и время остаются, рядом —
    причина и когда отозвана."""
    with db.transaction() as conn:
        row = conn.execute("SELECT * FROM diary WHERE id=?", (int(entry_id),)).fetchone()
        if row is None:
            return None
        entry = _diary_row(row)
        data = dict(entry.get("data") or {})
        data.update(status=RETRACTED, reason=str(reason or "").strip()[:300],
                    retracted=now_local().isoformat(timespec="minutes"))
        conn.execute("UPDATE diary SET data=? WHERE id=?",
                     (json.dumps(data, ensure_ascii=False), int(entry_id)))
    entry["data"] = data
    return entry


def add_diary_entry(
    text: str,
    tags: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None,
    source: str = "",
) -> Optional[Dict[str, Any]]:
    """Запись в дневник. Возвращает None если писать нечего (пустышка) или это
    точный дубль последней записи — анти-шум (Iris логировала мета/повторы).

    data — опциональная структурная нагрузка (напр. еда: dish/kcal/protein/place).
    Хранится в самой записи, чтобы аналитический слой (Этап 3) агрегировал тренды
    без отдельного meals.json. Тег [питание] при этом сохраняется для today_tags()."""
    text = (text or "").strip()
    if len(text) < 3:
        return None
    if source:
        data = {**(data or {}), "source": source}
    entry = {
        "id": None,
        "timestamp": now_local().isoformat(timespec="minutes"),
        "text": text,
        "tags": tags or [],
    }
    if data:
        entry["data"] = data
    # Проверка на дубль и вставка — под одной транзакцией. Без неё два
    # параллельных вызова видели одну и ту же «последнюю запись», антидубль
    # не срабатывал ни у одного, и в дневник ложился двойник.
    with db.transaction() as conn:
        last = conn.execute(
            "SELECT * FROM diary ORDER BY id DESC LIMIT 1").fetchone()
        if last is not None and str(last["text"]).strip().lower() == text.lower():
            return _diary_row(last)
        cur = conn.execute(
            "INSERT INTO diary(id, ts, text, tags, data) VALUES(NULL,?,?,?,?)",
            (entry["timestamp"], entry["text"],
             json.dumps(entry["tags"], ensure_ascii=False),
             json.dumps(data, ensure_ascii=False) if data else None),
        )
        entry["id"] = cur.lastrowid
    return entry


def read_diary(last_n: int = 10, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Записи вместе с отозванными — они помечены (is_retracted), чтобы на
    «что я записывал» ответ был полным, но отозванное не выдавалось за факт."""
    rows = [_diary_row(r) for r in db.query("SELECT * FROM diary ORDER BY id")]
    if tag:
        rows = [d for d in rows if tag in (d.get("tags") or [])]
    return rows[-last_n:] if last_n else rows


def delete_diary_entries(ids: List[int]) -> List[Dict[str, Any]]:
    """Удалить записи дневника по id. Возвращает УДАЛЁННЫЕ ЗАПИСИ целиком.

    Раньше возвращались только номера, и Iris отчитывалась «Удалила записи:
    #3» — правду о неверном действии. 17.08.2026 так погибла непричастная
    запись месячной давности, а ошибочная осталась на месте, и заметить это
    было нечем. Инвариант И2: изменяющее действие сообщает, ЧТО стало, а не
    «ок» — тогда подмена видна сразу, в том же сообщении.

    Чтение и удаление под одной транзакцией: между SELECT и DELETE запись
    могла исчезнуть, и отчёт разошёлся бы с фактом.
    """
    id_set = {int(i) for i in ids}
    if not id_set:
        return []
    marks = ",".join("?" * len(id_set))
    with db.transaction() as conn:
        rows = [_diary_row(r) for r in
                conn.execute(f"SELECT * FROM diary WHERE id IN ({marks})", tuple(id_set))]
        if rows:
            conn.execute(f"DELETE FROM diary WHERE id IN ({marks})", tuple(id_set))
    return rows


def last_entry_per_tag(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """Самая свежая запись дневника на каждый из тегов (любой давности).
    Чтобы Iris не отвечала «нет записей» о спорте/еде, когда они есть —
    последнее по теме инжектится в её STATE-блок."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in db.query("SELECT * FROM diary ORDER BY id"):
        entry = _diary_row(r)
        if is_retracted(entry):
            continue
        for t in entry.get("tags") or []:
            if t in tags:
                out[t] = entry
    return out


def today_tags() -> set:
    """Все теги дневника за сегодня — тикер проверяет закрыт ли слот (питание/спорт/…)."""
    today = now_local().strftime("%Y-%m-%d")
    tags: set = set()
    for r in db.query("SELECT tags, data FROM diary WHERE ts LIKE ?", (f"{today}%",)):
        if (_json_load(r["data"], None) or {}).get("status") == RETRACTED:
            continue
        tags.update(_json_load(r["tags"], []))
    return tags


def entries_today() -> int:
    """Сколько записей дневника за сегодня. Вечерний итог при 0 записей и
    отсутствовавшем Владе молчит — не спамит «день получился спокойный»."""
    today = now_local().strftime("%Y-%m-%d")
    return sum(1 for r in db.query("SELECT data FROM diary WHERE ts LIKE ?", (f"{today}%",))
               if (_json_load(r["data"], None) or {}).get("status") != RETRACTED)


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

def _monday(d) -> str:
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def get_week_plan() -> Dict[str, Any]:
    """План + пометка актуальности: current — составлен на эту неделю.

    01.10.2026 инструмент отдавал план на 13–19 августа без единого слова о
    том, что он протух, и Iris подтягивала его в ответы. Неделя плана — та,
    на которую он составлен (week_of); у старых записей — неделя сохранения."""
    plan = db.kv_get("week_plan", {}) or {}
    if not plan:
        return {}
    week_of = plan.get("week_of")
    if not week_of and plan.get("updated"):
        try:
            week_of = _monday(datetime.fromisoformat(plan["updated"]).date())
        except ValueError:
            week_of = None
    return {**plan, "week_of": week_of,
            "current": week_of == _monday(now_local().date())}


def save_week_plan(text: str) -> Dict[str, Any]:
    plan = {
        "updated": now_local().isoformat(timespec="minutes"),
        "week_of": _monday(now_local().date()),
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


# Сколько дней запас и просроченные дедлайны считаются актуальными без
# подтверждения. Дальше — пометка «устарел» и один вопрос владельцу.
PANTRY_STALE_DAYS = 14
DEADLINE_STALE_DAYS = 14


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
        source="code",
    )


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
    # Через дни, не только сегодня: «его не было N дней» и «ответил ли на пинг».
    db.kv_set("last_owner_at", now_local().isoformat(timespec="minutes"))
    _resolve_ping_answers(answered=True)


def owner_seen_today() -> bool:
    return bool(get_day_state().get("last_seen"))


def mark_ping(ping_id: str) -> None:
    state = get_day_state()
    state["pings"][ping_id] = now_local().strftime("%H:%M")
    save_day_state(state)
    log = db.kv_get(PING_LOG_KEY, []) or []
    log.append({"type": ping_id.split(":")[0], "at": now_local().isoformat(timespec="minutes"),
                "answered": None})
    cutoff = (now_local() - timedelta(days=PING_LOG_DAYS)).isoformat()
    db.kv_set(PING_LOG_KEY, [p for p in log if p["at"] >= cutoff])


# Журнал пингов: тип, когда, ответил ли он в течение часа. По нему тикер
# отключает пинги, на которые он не отвечает (10.06–01.10.2026: «как ты, какие
# планы» — 75 пингов, ответ на четверть, 46 дней бот писал в пустоту).
PING_LOG_KEY = "ping_log"
PING_LOG_DAYS = 45
PING_ANSWER_MIN = 60


def _resolve_ping_answers(answered: bool) -> None:
    """Его сообщение отвечает на пинги последнего часа; более старые без ответа —
    не отвечены."""
    log = db.kv_get(PING_LOG_KEY, []) or []
    if not log:
        return
    now = now_local()
    changed = False
    for p in log:
        if p.get("answered") is not None:
            continue
        try:
            age = (now - datetime.fromisoformat(p["at"])).total_seconds() / 60
        except (KeyError, ValueError):
            continue
        if age <= PING_ANSWER_MIN and answered:
            p["answered"], changed = True, True
        elif age > PING_ANSWER_MIN:
            p["answered"], changed = False, True
    if changed:
        db.kv_set(PING_LOG_KEY, log)


def ping_reply_rate(ping_type: str, days: int = 30) -> tuple:
    """(сколько пингов этого типа за `days` дней, на сколько он ответил за час)."""
    _resolve_ping_answers(answered=False)
    cutoff = (now_local() - timedelta(days=days)).isoformat()
    rows = [p for p in db.kv_get(PING_LOG_KEY, []) or []
            if p.get("type") == ping_type and p.get("at", "") >= cutoff]
    return len(rows), sum(1 for p in rows if p.get("answered"))


def last_owner_at() -> Optional[datetime]:
    raw = db.kv_get("last_owner_at", "")
    try:
        return datetime.fromisoformat(raw) if raw else None
    except ValueError:
        return None


# ============================================================================
# Mute — «стоп» от Влада. Два уровня (с 03.08.2026):
#   scope='pings' (дефолт) — молчит только дневной тикер (выходить/еда/спорт/
#     учёба); утренний дайджест, напоминания о дедлайнах и вечерний итог
#     ОСТАЮТСЯ — это информация, а не «дёрганье».
#   scope='all' — полная тишина всего проактивного (только по явной просьбе).
# Ответы на его собственные сообщения не блокируются никогда.
# Кросс-день, в отличие от per-day day_state.
# ============================================================================

_WEEKDAYS_SHORT = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


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
        return "без срока — пока не скажешь «пиши»"
    if mode != "today" and hours and hours > 0:
        # Потолок 30 дней: раньше было 168 ч, и «на 10 дней» молча резалось до 7.
        until = now + timedelta(hours=max(0.5, min(float(hours), 720.0)))
        data["until"] = until.isoformat(timespec="minutes")
        db.kv_set("mute", data)
        if until.date() == now.date():
            return "до " + until.strftime("%H:%M")
        return f"до {_WEEKDAYS_SHORT[until.weekday()]} {until.strftime('%d.%m %H:%M')}"
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


def mute_info() -> Optional[Dict[str, Any]]:
    """Действующая тишина ({scope, until, set}) или None."""
    rec = _mute_record_active()
    return dict(rec) if rec else None


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
# Ротация стилей пингов, радар дедлайнов
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
