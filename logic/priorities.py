"""
Приоритеты владельца — детерминированный расчёт «что сейчас важно».

LLM не может рассуждать о том, чего нет в контексте: блок TOP PRIORITIES
собирается чистым Python из реальных дедлайнов (таблица deadlines)
+ слотов сегодняшнего дня и вставляется в системный промпт Iris при каждой
генерации. Ноль LLM-вызовов. Тот же расчёт питает crunch-пинг тикера.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from logic import coach_storage
from logic.week_schedule import day_events, describe_event, get_shifts
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


def stale_deadlines() -> List[Dict[str, Any]]:
    """Открытые, но просроченные больше DEADLINE_STALE_DAYS дней: скорее всего
    сданы или потеряли смысл, а закрыть забыли. Не приоритет — вопрос."""
    today = now_local().date()
    return [d for d in _pending_deadlines()
            if (today - d["_due"]).days > coach_storage.DEADLINE_STALE_DAYS]


def top_priorities(max_items: int = 4) -> List[Dict[str, Any]]:
    """Pending-дедлайны по срочности (дата, потом важность), просроченные сверху.

    Отсечки по давности здесь больше нет. Раньше стояло
    `d["_due"] >= today - timedelta(days=7)`: дедлайн, просроченный больше чем
    на неделю, исчезал из промпта совсем — то есть чем дольше владелец тянул,
    тем реже коуч напоминал, а через семь дней замолкал вовсе.
    """
    stale = {d["id"] for d in stale_deadlines()}
    items = [d for d in _pending_deadlines() if d["id"] not in stale]
    items.sort(key=lambda d: (d["_due"], _IMPORTANCE_RANK.get(d.get("importance"), 1)))
    return items[:max_items]


def pending_count() -> int:
    """Сколько актуальных pending-дедлайнов — чтобы блок промпта не обрезал молча."""
    return len(_pending_deadlines()) - len(stale_deadlines())


def crunch_deadline(days: int = 3) -> Optional[Dict[str, Any]]:
    """Ближайший HIGH-дедлайн в пределах N дней (включая просроченный) —
    триггер crunch-пинга в тикере."""
    today = now_local().date()
    for d in top_priorities():
        if d.get("importance") == "high" and d["_due"] <= today + timedelta(days=days):
            return d
    return None


def crunch_tonight() -> Optional[Dict[str, Any]]:
    """HIGH-дедлайн сегодня/завтра. Инжектится в промпт Iris как флаг, что
    поздний вечер перестаёт быть запретным: если других окон нет, учёба ночью
    НОРМАЛЬНА. Снимает жёсткость 22:30 ровно тогда, когда коучинг важнее всего."""
    today = now_local().date()
    for d in top_priorities():
        if d.get("importance") == "high" and d["_due"] <= today + timedelta(days=1):
            return d
    return None


def radar_deadline(lo: int = 4, hi: int = 7) -> Optional[Dict[str, Any]]:
    """Pending-дедлайн в окне [lo,hi] дней, про который ещё НЕ предупреждали
    проактивно — раннее мягкое «на радаре, начал?». Importance не важен: средний
    экзамен (матан) должен всплыть заранее, а не только за 3 дня (crunch)."""
    today = now_local().date()
    for d in _pending_deadlines():
        left = (d["_due"] - today).days
        if lo <= left <= hi and not coach_storage.radar_pinged(d.get("id")):
            return d
    return None


def build_day_context() -> str:
    """DAY CONTEXT для Iris: когда проснулся, что было за день (дневник),
    что завтра. Лечит класс проблем «план в прошлое»: модель видит реальный
    день владельца, а не сочиняет идеальный с 09:00."""
    now = now_local()
    today = now.strftime("%Y-%m-%d")
    lines = ["DAY CONTEXT (computed from real data):"]
    lines.append(f"  Сейчас: {_DAY_NAMES[now.weekday()]} {now.strftime('%d.%m %H:%M')}")

    wake = coach_storage.wake_time_today()
    if wake:
        lines.append(f"  Проснулся: {wake}")

    entries = [e for e in coach_storage.read_diary(last_n=30)
               if str(e.get("timestamp", "")).startswith(today)]
    if entries:
        lines.append("  Сегодня в дневнике:")
        for e in entries[-8:]:
            t = str(e.get("timestamp", ""))[11:16]
            tags = ",".join(e.get("tags") or [])
            text = (e.get("text") or "").replace("\n", " ")[:60]
            lines.append(f"    {t} [{tags}] {text}")

    recent = coach_storage.last_entry_per_tag(["спорт", "питание", "учёба", "учеба", "сон"])
    if "учеба" in recent and "учёба" not in recent:
        recent["учёба"] = recent.pop("учеба")
    recent_lines = []
    for tag in ("спорт", "питание", "учёба", "сон"):
        e = recent.get(tag)
        if e:
            d = str(e.get("timestamp", ""))[:10]
            txt = (e.get("text") or "").replace("\n", " ")[:50]
            recent_lines.append(f"    {tag}: {d} — {txt}")
    if recent_lines:
        lines.append("  Недавнее (последняя запись по теме — не говори «нет записей» вслепую):")
        lines += recent_lines

    from logic.week_schedule import coverage
    covered = coverage()
    if covered:
        lines.append(f"  {covered}")
    from logic.memory_review import last_week_line, yesterday_line
    for line in (yesterday_line(now.date()), last_week_line(now.date())):
        if line:
            lines.append(f"  {line}")

    tomorrow = get_shifts(now.date() + timedelta(days=1))
    if tomorrow:
        lines.append("  Завтра: " + "; ".join(
            f"смена {s['start']}–{s['end']}" for s in tomorrow))

    lines.append("  ПРАВИЛО: план дня — только вперёд от «Сейчас», прошедшие часы не планировать.")
    return "\n".join(lines)


def _stale_question_due() -> bool:
    """Вопрос про устаревшие дедлайны — не чаще раза в неделю."""
    from utils import db
    last = db.kv_get("stale_deadlines_asked", "")
    today = now_local().date()
    try:
        if last and (today - datetime.strptime(last, "%Y-%m-%d").date()).days < 7:
            return False
    except ValueError:
        pass  # битая дата — считаем, что не спрашивали
    db.kv_set("stale_deadlines_asked", today.strftime("%Y-%m-%d"))
    return True


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
        hidden = pending_count() - len(prios)
        if hidden > 0:
            # Молчаливое обрезание читается как «это всё» — а это не всё.
            lines.append(f"  …и ещё {hidden} незакрытых — спроси list_deadlines")

    stale = stale_deadlines()
    if stale and _stale_question_due():
        listing = "; ".join(f"#{d['id']} «{d['title']}» ({d['due']})" for d in stale[:5])
        lines.append(
            f"⚠ УСТАРЕВШИЕ ДЕДЛАЙНЫ (просрочены больше {coach_storage.DEADLINE_STALE_DAYS} дн., "
            f"не закрыты): {listing}. Спроси его ОДИН раз, закрыть ли их; не дави и не "
            f"выдавай за срочное.")

    crunch = crunch_tonight()
    if crunch:
        when_c = "сегодня" if crunch["_due"] == today else "завтра"
        lines.append(
            f"⚠ CRUNCH: «{crunch['title']}» — дедлайн {when_c}. Если раньше окон сегодня "
            f"нет, поздний вечер — НОРМАЛЬНЫЙ слот для подготовки: не запрещай его и не "
            f"читай нотаций про сон, помоги спланировать (что успеть, когда стоп)."
        )

    day_parts: List[str] = []
    for shift in get_shifts(today):
        day_parts.append(f"смена {shift['start']}–{shift['end']}")
    day_parts += [describe_event(r) for r in day_events(today)]
    if day_parts:
        label = f"{_DAY_NAMES[today.weekday()]} {today.strftime('%d.%m')}"
        lines.append(f"TODAY ({label}): " + "; ".join(day_parts))

    return "\n".join(lines)
