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
from datetime import datetime, timedelta
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


def mark_deadline_done(deadline_id: int) -> Optional[Dict[str, Any]]:
    """Закрыть дедлайн (сдал/прошло). Без этого сданный тест вечно висит
    в TOP PRIORITIES и Iris продолжает пушить."""
    deadlines = _load_json("deadlines.json", [])
    for d in deadlines:
        if d.get("id") == deadline_id:
            d["status"] = "done"
            d["closed"] = now_local().strftime("%Y-%m-%d")
            _save_json("deadlines.json", deadlines)
            return d
    return None


def delete_deadline(deadline_id: int) -> Optional[Dict[str, Any]]:
    """Удалить дедлайн НАСОВСЕМ («удали/убери — не нужен»). Не путать с
    mark_deadline_done: done = «сдал/прошло», остаётся в истории; delete —
    ошибочный/неактуальный исчезает и статистику не портит (кейс 02.08:
    «Удали его» превратился в done, потому что удалять было нечем)."""
    deadlines = _load_json("deadlines.json", [])
    for i, d in enumerate(deadlines):
        if d.get("id") == deadline_id:
            deadlines.pop(i)
            _save_json("deadlines.json", deadlines)
            return d
    return None


def update_deadline(
    deadline_id: int,
    due: Optional[str] = None,
    title: Optional[str] = None,
    importance: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Правка существующего дедлайна (перенос даты и т.п.). «Перенесём на неделю»
    = ЭТО, а не add_deadline: новый рядом со старым pending = дубль, и Iris
    долбит по обоим (кейс #3/#4/#5 «Матан», июль 2026)."""
    deadlines = _load_json("deadlines.json", [])
    for d in deadlines:
        if d.get("id") == deadline_id:
            if due:
                d["due"] = due
            if title:
                d["title"] = title.strip()
            if importance:
                d["importance"] = importance
            _save_json("deadlines.json", deadlines)
            return d
    return None


# ============================================================================
# Diary
# ============================================================================

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
    diary = _load_json("diary.json", [])
    if diary and str(diary[-1].get("text", "")).strip().lower() == text.lower():
        return diary[-1]
    entry = {
        "id": _next_id(diary),
        "timestamp": now_local().isoformat(timespec="minutes"),
        "text": text,
        "tags": tags or [],
    }
    if data:
        entry["data"] = data
    diary.append(entry)
    _save_json("diary.json", diary)
    return entry


def read_diary(last_n: int = 10, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    diary = _load_json("diary.json", [])
    if tag:
        diary = [d for d in diary if tag in (d.get("tags") or [])]
    return diary[-last_n:]


def delete_diary_entries(ids: List[int]) -> List[int]:
    """Удалить записи дневника по id. Возвращает РЕАЛЬНО удалённые id
    (чтобы Iris отчитывалась только о фактически снесённом, а не врала)."""
    id_set = {int(i) for i in ids}
    diary = _load_json("diary.json", [])
    removed = [e["id"] for e in diary if e.get("id") in id_set]
    if removed:
        diary = [e for e in diary if e.get("id") not in id_set]
        _save_json("diary.json", diary)
    return removed


def last_entry_per_tag(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """Самая свежая запись дневника на каждый из тегов (любой давности).
    Чтобы Iris не отвечала «нет записей» о спорте/еде, когда они есть —
    последнее по теме инжектится в её STATE-блок."""
    out: Dict[str, Dict[str, Any]] = {}
    for e in _load_json("diary.json", []):
        for t in (e.get("tags") or []):
            if t in tags:
                out[t] = e
    return out


# ============================================================================
# Pantry — запас продуктов (инкрементальный, НЕ снапшот-перезапись и НЕ граммы).
# Items = строки. Обновляется при покупке/готовке. updated → мягкая ресинхронизация.
# ============================================================================

def _norm_item(s: Any) -> str:
    return " ".join(str(s).lower().split())


def get_pantry() -> Dict[str, Any]:
    return _load_json("pantry.json", {"items": [], "updated": None})


def pantry_update(add: Optional[List[str]] = None,
                  remove: Optional[List[str]] = None) -> Dict[str, Any]:
    """Инкрементально: добавить купленное / убрать потраченное. Дедуп по
    нормализованному имени, порядок добавления сохраняется."""
    data = get_pantry()
    items: List[str] = list(data.get("items") or [])
    have = {_norm_item(i) for i in items}
    for it in (add or []):
        it = str(it).strip()
        if it and _norm_item(it) not in have:
            items.append(it)
            have.add(_norm_item(it))
    if remove:
        rm = {_norm_item(r) for r in remove}
        items = [i for i in items if _norm_item(i) not in rm]
    data = {"items": items, "updated": now_local().strftime("%Y-%m-%d")}
    _save_json("pantry.json", data)
    return data


def pantry_age_days() -> Optional[int]:
    """Сколько дней назад обновляли запас (для мягкой ресинхронизации). None — пусто."""
    upd = get_pantry().get("updated")
    if not upd:
        return None
    try:
        d = datetime.strptime(upd, "%Y-%m-%d").date()
        return (now_local().date() - d).days
    except ValueError:
        return None


# ============================================================================
# Week plan — текст плана недели от Iris (составляется при загрузке смен
# или по запросу «составь план недели», правится словами через чат)
# ============================================================================

def get_week_plan() -> Dict[str, Any]:
    return _load_json("week_plan.json", {})


def save_week_plan(text: str) -> Dict[str, Any]:
    plan = {
        "updated": now_local().isoformat(timespec="minutes"),
        "text": text.strip(),
    }
    _save_json("week_plan.json", plan)
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


def wake_time_today() -> Optional[str]:
    """«HH:MM» пробуждения, если зафиксировано сегодня."""
    presence = _load_json("presence.json", {})
    if presence.get("last_wake_date") == now_local().strftime("%Y-%m-%d"):
        return presence.get("wake_time")
    return None


# ============================================================================
# Day state — анти-спам для проактивных пингов (тикер)
# ============================================================================

def get_day_state() -> Dict[str, Any]:
    """State за сегодня: какие пинги отправлены. Авто-сброс на новой дате.
    (Тишина-по-запросу живёт отдельно в mute.json — она кросс-день.)"""
    state = _load_json("day_state.json", {})
    today = now_local().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "pings": {}}
    return state


def save_day_state(state: Dict[str, Any]) -> None:
    _save_json("day_state.json", state)


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
# Кросс-день (mute.json), в отличие от per-day day_state.
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
        _save_json("mute.json", data)
        return "пока не скажешь «пиши»"
    if mode != "today" and hours and hours > 0:
        until = now + timedelta(hours=max(0.5, min(float(hours), 168.0)))
        data["until"] = until.isoformat(timespec="minutes")
        _save_json("mute.json", data)
        return ("до " + until.strftime("%H:%M")
                if until.date() == now.date()
                else "до " + until.strftime("%H:%M %d.%m"))
    until = now.replace(hour=23, minute=59, second=59, microsecond=0)
    data["until"] = until.isoformat(timespec="minutes")
    _save_json("mute.json", data)
    return "до конца дня"


def unmute() -> None:
    _save_json("mute.json", {})


def _mute_record_active() -> Optional[Dict[str, Any]]:
    data = _load_json("mute.json", {})
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


def today_tags() -> set:
    """Все теги дневника за сегодня — тикер проверяет закрыт ли слот (питание/спорт/…)."""
    today = now_local().strftime("%Y-%m-%d")
    tags: set = set()
    for e in _load_json("diary.json", []):
        if str(e.get("timestamp", "")).startswith(today):
            tags.update(e.get("tags") or [])
    return tags


def entries_today() -> int:
    """Сколько записей дневника за сегодня. Вечерний итог при 0 записей и
    отсутствовавшем Владе молчит — не спамит «день получился спокойный»."""
    today = now_local().strftime("%Y-%m-%d")
    return sum(
        1 for e in _load_json("diary.json", [])
        if str(e.get("timestamp", "")).startswith(today)
    )


def next_style_index(n: int) -> int:
    """Ротация стилевых вариантов пингов (persist кросс-день) — чтобы Iris не
    открывала сообщения одинаково два раза подряд (жалоба 03.08: «формат
    крайне одинаковый и надоедает»)."""
    data = _load_json("ping_style.json", {})
    idx = (int(data.get("idx", -1)) + 1) % max(1, n)
    _save_json("ping_style.json", {"idx": idx})
    return idx


# ============================================================================
# Radar — одноразовый ранний пинг по дедлайну (кросс-день, чтобы не нудеть)
# ============================================================================

def radar_pinged(deadline_id: Any) -> bool:
    """Уже делали ранний «радар»-пинг по этому дедлайну? Persistent (radar.json)."""
    return str(deadline_id) in _load_json("radar.json", {})


def mark_radar(deadline_id: Any) -> None:
    data = _load_json("radar.json", {})
    data[str(deadline_id)] = now_local().strftime("%Y-%m-%d")
    _save_json("radar.json", data)


# ============================================================================
# Gemini RPD-гард — счётчик запросов за день (free-tier 1500/проект; пул общий
# с vision/поиском/дайджестом/Iris-петлёй). Авто-сброс на новой дате.
# ============================================================================

_GEMINI_RPD_WARN = (1200, 1450)  # пороги для одноразового warning в лог


def gemini_bump() -> int:
    data = _load_json("gemini_usage.json", {})
    today = now_local().strftime("%Y-%m-%d")
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    data["count"] = int(data.get("count", 0)) + 1
    _save_json("gemini_usage.json", data)
    if data["count"] in _GEMINI_RPD_WARN:
        logger.warning("Gemini RPD: %d запросов сегодня (free-tier лимит ~1500)", data["count"])
    return data["count"]


def gemini_count_today() -> int:
    data = _load_json("gemini_usage.json", {})
    return int(data.get("count", 0)) if data.get("date") == now_local().strftime("%Y-%m-%d") else 0
