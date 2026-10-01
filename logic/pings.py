"""
Решение «пинговать или молчать» для дневного тикера. ЧИСТЫЙ Python — LLM
дёргается только когда решение «пинговать» уже принято.

Редизайн 01.10.2026 — по данным 10.06–01.10 (138 пингов):
  • «как ты, какие планы» (cold-start) — 75 пингов, ответ за час на 25%,
    в 46 из 85 дней бот писал, а владелец не написал ни разу; с августа бот
    писал ему столько же, сколько он боту. Этого пинга больше нет.
  • лучше всего работали пинги про конкретное: еда перед сменой/обед (45–50%),
    тренировка (46%), дедлайн (66%).
Отсюда правила:
  1. Каждый пинг — про конкретный факт: событие по расписанию, еду перед
     длинным блоком, дедлайн, его собственный план. Не «как ты».
  2. Время-критичные (выходить на смену/пару, «завтра рано») приходят, даже
     если он сегодня не писал. Остальные — только когда он на связи и не
     писал последний час (он в разговоре — пинг лишний).
  3. Тип пинга, на который он за 30 дней ответил реже чем в 20% случаев,
     отключается сам (journal: coach_storage.ping_log); ночной разбор об этом
     скажет. Через 30 дней без пингов этого типа он снова доступен.
  4. Не больше MAX_PINGS_PER_DAY в день, пауза MIN_GAP_MIN, два пинга подряд
     без ответа — тишина до его сообщения (кроме время-критичных).
  5. Долго не писал (ABSENCE_DAYS) — одно сообщение: что впереди, без упрёков.

Вариативность (03.08.2026, «формат крайне одинаковый и надоедает»): к фактам
пинга добавляется ротируемая стилевая инструкция (_style_hint).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from logic import coach_storage
from logic.priorities import crunch_deadline, radar_deadline
from logic.situation_engine import build_day_situation, parse_hm
from logic.week_schedule import (
    WORK_COMMUTE_MIN,
    day_events,
    get_shifts,
    is_home_study_day,
)
from utils.time import now_local

logger = logging.getLogger(__name__)

MAX_PINGS_PER_DAY = 3
MIN_GAP_MIN = 90
# Он только что писал — он в разговоре, пинг сверху лишний.
QUIET_AFTER_MESSAGE_MIN = 60
# Приветствие — только в первые полтора часа после пробуждения. Без верхней
# границы оно ждало конца тишины: 01.10.2026 тишина кончилась в 16:45, и в
# 17:00 Iris «поздоровалась» про «проснулся недавно (в 11:59)» после часа
# переписки, повторив то, что уже обсудили.
GREETING_WINDOW_MIN = 90
# Окно пинга «обед» в часах [с, до). Вечером про обед не спрашиваем.
MEAL_WINDOW = (14, 17)
# Дорога до места, минуты: на работу — самокат; в универ — дом → Essen Hbf →
# SB16 → кампус. Для остального дорога неизвестна — напоминания «выходить» нет.
COMMUTE_MIN = {"shift": WORK_COMMUTE_MIN, "lecture": 60}
# Напоминание «выходить» — в окне [выход−35, выход−5] минут (тикер раз в 30 мин).
LEAVE_WINDOW = (35, 5)
# «Завтра рано»: вечером, если первое событие завтра раньше EARLY_START_HOUR.
EVENING_BEFORE = (21, 23)
EARLY_START_HOUR = 10
# «Как прошло?» — через 1–3 часа после времени, которое он сам назвал.
FOLLOWUP_AFTER_MIN = (60, 180)
ABSENCE_DAYS = 2
ABSENCE_HOURS = (13, 20)
# Подстройка: тип с ответом реже ADAPT_MIN_RATE (при ≥ ADAPT_MIN_PINGS за окно) — выкл.
ADAPT_DAYS = 30
ADAPT_MIN_PINGS = 5
ADAPT_MIN_RATE = 0.2
# Время-критичные: идут без его сообщения сегодня, мимо паузы и backoff.
TIME_CRITICAL = {"leave", "tomorrow_early", "greeting"}

# Стилевые инструкции ротируются кросс-день (coach_storage.next_style_index) —
# Iris не должна открывать сообщения одинаково два раза подряд.
STYLE_VARIANTS = [
    "Стиль: начни с лёгкого прикола или неожиданного захода по теме — без кринжа "
    "и без смайлико-спама.",
    "Стиль: мягко и по-человечески, как друг, который просто рядом — без "
    "коуч-тона и без бодрячества.",
    "Стиль: коротко и по делу, одна-две фразы, без вступлений и приветствий.",
    "Стиль: начни с наблюдения о его дне (из STATE), а вопрос — вторым "
    "предложением.",
    "Стиль: задай вопрос нестандартно — не «как дела/поел?», а живой "
    "формулировкой, будто продолжаешь вчерашний разговор.",
]
TONE_GENTLE = (
    "Стиль: у него сегодня рабочий день/смена — тон бережный и короткий: "
    "поддержать, не грузить, ничего не требовать."
)
NO_REPEAT = (
    "Не повторяй формулировки прошлых пингов: шаблонные открытия («Привет! Как "
    "ты?», «Не забудь…») запрещены — придумай новое первое предложение."
)


def _style_hint(situation) -> str:
    """Ротируемая стилевая добавка к фактам пинга. Смена сегодня → всегда
    бережно; иначе — следующий вариант из пула (без повтора подряд)."""
    if situation.has_work_today or situation.shift.active:
        return f"{TONE_GENTLE} {NO_REPEAT}"
    idx = coach_storage.next_style_index(len(STYLE_VARIANTS))
    return f"{STYLE_VARIANTS[idx]} {NO_REPEAT}"


def decide_ping() -> Optional[Tuple[str, str]]:
    """
    Возвращает (ping_id, контекст для Iris-промпта) или None.
    Вызывающий обязан сразу mark_ping(ping_id) — защита от дублей.
    """
    now = now_local()
    situation = build_day_situation(now)
    decision = _slot_decision(situation, now)
    if decision is None:
        return None
    ping_id, context_text = decision
    return ping_id, f"{context_text} {_style_hint(situation)}"


# ---------------------------------------------------------------------------
# Факты для пингов
# ---------------------------------------------------------------------------

def _events(d: date, now: datetime) -> List[Dict[str, Any]]:
    """Смены и события дня с часами, по времени: {kind, start, title, place}."""
    out = [{"kind": "shift", "start": s["start"], "end": s["end"], "title": "смена",
            "place": ""} for s in get_shifts(d)]
    out += [{"kind": r["kind"], "start": r["start"], "end": r["end"], "title": r["title"],
             "place": r.get("location") or ""}
            for r in day_events(d) if r.get("start") and r.get("end")]
    return sorted(out, key=lambda e: e["start"])


def _at(d: date, hm: str, now: datetime) -> Optional[datetime]:
    t = parse_hm(hm, now)
    return t.replace(year=d.year, month=d.month, day=d.day) if t else None


def _class_hours(events: List[Dict[str, Any]]) -> float:
    total = 0.0
    for e in events:
        if e["kind"] == "lecture":
            s, f = parse_hm(e["start"], datetime(2000, 1, 1)), parse_hm(e["end"], datetime(2000, 1, 1))
            if s and f and f > s:
                total += (f - s).total_seconds() / 3600
    return total


_TIME_RX = re.compile(r"(?:\bв|\bum|\bat|\bк)\s*(\d{1,2})(?:[:.](\d{2}))?\b", re.IGNORECASE)


def _planned_today(now: datetime) -> List[Tuple[datetime, str]]:
    """Его собственные планы на сегодня со временем («собес в 15», «um 15»)."""
    today = now.date().isoformat()
    out = []
    for e in coach_storage.read_diary(last_n=30):
        if not str(e.get("timestamp", "")).startswith(today) or coach_storage.is_retracted(e):
            continue
        if "план" not in (e.get("tags") or []):
            continue
        if (e.get("data") or {}).get("source") not in (None, "owner"):
            continue
        m = _TIME_RX.search(e["text"])
        if not m or int(m.group(1)) > 23:
            continue
        at = now.replace(hour=int(m.group(1)), minute=int(m.group(2) or 0), second=0,
                         microsecond=0)
        out.append((at, e["text"]))
    return out


def _disabled(ping_type: str) -> bool:
    """Он почти не отвечает на этот тип — не пингуем (см. правило 3)."""
    n, answered = coach_storage.ping_reply_rate(ping_type, ADAPT_DAYS)
    if n >= ADAPT_MIN_PINGS and answered / n < ADAPT_MIN_RATE:
        logger.info("Пинг «%s» выключен: ответ %d из %d за %d дн.", ping_type, answered, n,
                    ADAPT_DAYS)
        return True
    return False


# ---------------------------------------------------------------------------
# Политика
# ---------------------------------------------------------------------------

def _slot_decision(situation, now) -> Optional[Tuple[str, str]]:
    """Какой пинг сейчас, если какой-то. Стиль добавляет caller."""
    pings = situation.pings
    if situation.muted or len(pings) >= MAX_PINGS_PER_DAY:
        return None

    for ping_id, text in _candidates(situation, now):
        kind = ping_id.split(":")[0]
        if ping_id in pings:
            continue
        if kind == "absence":
            # Его нет со вчера и дольше — «на связи сегодня» тут не условие.
            if _disabled(kind):
                continue
        elif kind not in TIME_CRITICAL:
            if not _regular_allowed(situation, now) or _disabled(kind):
                continue
        _on_send(kind, now)
        return ping_id, text
    return None


def _on_send(kind: str, now: datetime) -> None:
    """Пометки «уже сказали» — только когда пинг действительно выбран."""
    if kind == "radar":
        rad = radar_deadline()
        if rad is not None:
            coach_storage.mark_radar(rad["id"])
    elif kind == "absence":
        last = coach_storage.last_owner_at()
        if last is not None:
            _mark_absence_asked(last)


def _regular_allowed(situation, now) -> bool:
    """Обычный (не время-критичный) пинг: он на связи, не в разговоре, без
    двух проигнорированных подряд, после паузы, не на паре."""
    if situation.in_study_block or not situation.owner_seen:
        return False
    if situation.ignored_pings_streak() >= 2:
        return False
    last_msg = parse_hm(situation.last_msg, now) if situation.last_msg else None
    if last_msg and now - last_msg < timedelta(minutes=QUIET_AFTER_MESSAGE_MIN):
        return False
    last = situation.last_ping_at()
    return last is None or (now - last) >= timedelta(minutes=MIN_GAP_MIN)


def _candidates(situation, now):
    """Пинги-кандидаты по приоритету. Генератор: факты считаются по мере нужды."""
    today = now.date()
    tags = situation.tags
    shift = situation.shift.active_record
    shift_start = situation.shift.start_at
    events = _events(today, now)

    # --- выходить: ближайшее событие с известной дорогой ---
    for e in events:
        commute = COMMUTE_MIN.get(e["kind"])
        start = _at(today, e["start"], now)
        if not commute or not start or start <= now:
            continue
        leave = start - timedelta(minutes=commute)
        # Он ещё на предыдущем занятии (пары подряд в одном кампусе) — выходить
        # никуда не надо. Симуляция 05.10: «выходить в 13:05 на 14:05» посреди пары.
        busy = any((_at(today, o["end"], now) or start) > leave - timedelta(minutes=30)
                   for o in events if o is not e and o["start"] < e["start"])
        if busy:
            break
        if leave - timedelta(minutes=LEAVE_WINDOW[0]) <= now <= leave - timedelta(minutes=LEAVE_WINDOW[1]):
            where = f" ({e['place']})" if e["place"] else ""
            yield (f"leave:{e['start']}", (
                f"Сегодня {e['title']} в {e['start']}{where}, дорога ~{commute} мин — выходить "
                f"примерно в {leave:%H:%M}. Напомни одной строкой, без лишнего."))
        break

    # --- утреннее приветствие (+ вопрос ночного разбора) ---
    wake = situation.wake_time
    if wake and situation.owner_seen:
        ws = parse_hm(wake, now)
        if ws is not None and (ws + timedelta(minutes=15) <= now
                               <= ws + timedelta(minutes=GREETING_WINDOW_MIN)):
            from logic.memory_review import take_morning_question
            question = take_morning_question(now.date())
            ask = (f" Вместо вопроса про план задай вопрос из ночного разбора: «{question}» "
                   f"(своими словами, коротко)." if question
                   else " И один лёгкий вопрос про план.")
            yield ("greeting", (
                f"Влад проснулся недавно (в {wake}) и на связи. Поздоровайся тепло и "
                f"коротко, по-человечески, дай сводку дня из STATE (смена/лекции/горящие "
                f"дедлайны если есть).{ask} Без списка на полэкрана, без давления."
            ))

    # --- завтра рано ---
    if EVENING_BEFORE[0] <= now.hour < EVENING_BEFORE[1]:
        tomorrow = today + timedelta(days=1)
        first = next((e for e in _events(tomorrow, now) if e["start"] < f"{EARLY_START_HOUR:02d}:00"),
                     None)
        if first:
            commute = COMMUTE_MIN.get(first["kind"])
            starts = _at(tomorrow, first["start"], now)
            leave = (f", выходить около {starts - timedelta(minutes=commute):%H:%M}"
                     if commute and starts else "")
            yield ("tomorrow_early", (
                f"Завтра рано: {first['title']} в {first['start']}{leave}. Коротко напомни "
                f"про это и про сон/будильник — без нотаций."))

    # --- смена под вопросом ---
    if situation.shift.needs_confirmation(now) and not situation.has_work_today:
        yield ("shift_confirm", (
            f"По расписанию сегодня смена {shift['start']}–{shift['end']}. "
            "Аккуратно спроси, всё ли в силе или график поменялся. Одно сообщение, "
            "без давления и без дополнительных советов."
        ))

    # --- еда перед длинным блоком: смена или 3+ часа пар ---
    if "питание" not in tags:
        block = None
        if shift_start is not None:
            block = (shift_start, f"смена {shift['start']}–{shift['end']}")
        elif _class_hours([e for e in events if _at(today, e["start"], now) > now]) >= 3:
            first = next(e for e in events if e["kind"] == "lecture"
                         and _at(today, e["start"], now) > now)
            block = (_at(today, first["start"], now), f"пары с {first['start']} больше трёх часов")
        if block and block[0] - timedelta(hours=3) <= now <= block[0] - timedelta(minutes=40):
            yield ("meal", (
                f"Сегодня {block[1]}. За день нет ни одной записи о еде. Пингани коротко: "
                f"поесть нормально ДО этого, потом будет некогда."))
        elif (shift_start is None or shift_start.hour >= 19) and MEAL_WINDOW[0] <= now.hour < MEAL_WINDOW[1]:
            yield ("meal", (
                "Время к обеду, а записей о еде за день нет. Пингани коротко: поел ли, "
                "и если нет — пусть поест по-нормальному, не кофе единым."))

    # --- как прошло то, что он сам запланировал ---
    for at, text in _planned_today(now):
        if at + timedelta(minutes=FOLLOWUP_AFTER_MIN[0]) <= now <= at + timedelta(minutes=FOLLOWUP_AFTER_MIN[1]):
            yield ("followup", (
                f"Он сам писал сегодня: «{text}». Время прошло — спроси коротко, как прошло. "
                f"Если в дневнике уже есть итог — не спрашивай, а отреагируй на него."))
            break

    # --- горящий дедлайн ---
    crunch = crunch_deadline()
    if (crunch is not None and not ({"учёба", "учеба"} & tags) and now.hour >= 11
            and (shift_start is None or now < shift_start - timedelta(hours=2))):
        left = (crunch["_due"] - now.date()).days
        when = ("СЕГОДНЯ" if left == 0 else f"просрочен на {-left} дн" if left < 0
                else f"через {left} дн ({crunch['due']})")
        yield ("crunch", (
            f"Горящий дедлайн: «{crunch['title']}» — {when}. Записей про учёбу за день нет. "
            f"Спроси прямо: когда сегодня сядет за подготовку — и предложи конкретный слот "
            f"по расписанию дня. Один раз, жёстко, но без пиления."))

    # --- тренировка: если спорт не стоит в расписании и день не забит парами ---
    sport_planned = any(e["kind"] == "sport" for e in events)
    if ("спорт" not in tags and not situation.has_work_today and not sport_planned
            and _class_hours(events) < 5):
        if shift_start is None and 17 <= now.hour < 21:
            yield ("training", (
                "Сегодня смены нет, тренировки в расписании нет и записей о спорте нет. "
                "Спроси коротко: тренька сегодня будет? Без давления — если нет, принять."))
        elif shift_start is not None and shift_start.hour >= 18 and 11 <= now.hour < shift_start.hour - 3:
            yield ("training", (
                f"Смена сегодня только в {shift['start']} — до неё есть окно. Записей о спорте "
                f"нет. Мягко предложи короткую треньку до работы, если есть силы."))

    # --- день домашней учёбы ---
    if (is_home_study_day(today) and not ({"работа", "учёба", "учеба"} & tags) and now.hour >= 13
            and (shift_start is None or now < shift_start - timedelta(hours=2))):
        yield ("study", (
            "Сегодня день домашней учёбы, а записей про учёбу/работу нет. Спроси коротко, "
            "что сегодня по учёбе. Одно сообщение."))

    # --- радар: дедлайн через 4–7 дней ---
    rad = radar_deadline()
    if (rad is not None and not ({"учёба", "учеба"} & tags) and now.hour >= 11
            and (shift_start is None or now < shift_start - timedelta(hours=2))):
        left = (rad["_due"] - now.date()).days
        yield ("radar", (
            f"На радаре дедлайн «{rad['title']}» — через {left} дн ({rad['due']}). "
            f"Ещё не горит, но спроси мягко: начал ли, нужен ли слот в плане недели. "
            f"Один раз, без давления."))

    # --- долго не писал: одно сообщение без упрёков ---
    last = coach_storage.last_owner_at()
    if (last is not None and (now - last) >= timedelta(days=ABSENCE_DAYS)
            and ABSENCE_HOURS[0] <= now.hour < ABSENCE_HOURS[1]
            and not situation.owner_seen and not _absence_asked(last)):
        yield ("absence", (
            f"Он не писал {(now - last).days} дн. Одно короткое сообщение без упрёков и без "
            f"«как ты»: что у него впереди по расписанию и дедлайнам (из STATE) и что ты на "
            f"связи. Не повторяй, если не ответит."))


def _absence_asked(since: datetime) -> bool:
    from utils import db
    return db.kv_get("absence_ping_for", "") == since.isoformat(timespec="minutes")


def _mark_absence_asked(since: datetime) -> None:
    from utils import db
    db.kv_set("absence_ping_for", since.isoformat(timespec="minutes"))
