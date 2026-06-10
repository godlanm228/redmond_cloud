"""
Решение «пинговать или молчать» для дневного тикера. ЧИСТЫЙ Python — LLM
дёргается только когда решение «пинговать» уже принято (экономия токенов).

Анти-спам гарантии (в коде, не на совести LLM):
  • тишина пока Влад не проснулся (presence), при snooze, ночью (тикер 10–23);
  • один пинг на слот в день, максимум MAX_PINGS_PER_DAY, пауза ≥ MIN_GAP_MIN;
  • слот закрыт записью в дневнике с нужным тегом → пинга нет
    («поел» → питание; «сегодня без зала» → спорт; и т.д.).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from logic import coach_storage
from logic.week_schedule import (
    HOME_STUDY_WEEKDAYS,
    STUDY_TIMETABLE,
    WORK_COMMUTE_MIN,
    get_shift,
)
from utils.time import now_local

logger = logging.getLogger(__name__)

MAX_PINGS_PER_DAY = 5
MIN_GAP_MIN = 90


def _parse_hm(s: str, base: datetime) -> Optional[datetime]:
    try:
        h, m = map(int, s.split(":"))
        return base.replace(hour=h, minute=m, second=0, microsecond=0)
    except (ValueError, AttributeError):
        return None


def decide_ping() -> Optional[Tuple[str, str]]:
    """
    Возвращает (ping_id, контекст для Iris-промпта) или None.
    Вызывающий обязан сразу mark_ping(ping_id) — защита от дублей.
    """
    now = now_local()
    state = coach_storage.get_day_state()

    # --- глобальные предохранители ---
    if not coach_storage.woke_today():
        return None
    snooze = state.get("snooze_until")
    if snooze:
        try:
            if now < datetime.fromisoformat(snooze):
                return None
        except ValueError:
            pass
    pings = state.get("pings", {})
    if len(pings) >= MAX_PINGS_PER_DAY:
        return None
    if pings:
        last = max(_parse_hm(t, now) or now for t in pings.values())
        if (now - last) < timedelta(minutes=MIN_GAP_MIN):
            return None

    tags = coach_storage.today_tags()
    shift = get_shift(now.date())
    shift_start = _parse_hm(shift["start"], now) if shift else None

    # --- 1. Еда перед сменой: окно [старт-3ч, старт-40мин], выход ~за 20 мин ---
    if (
        shift_start is not None
        and "питание" not in tags
        and "meal" not in pings
        and shift_start - timedelta(hours=3) <= now <= shift_start - timedelta(minutes=40)
    ):
        leave = (shift_start - timedelta(minutes=WORK_COMMUTE_MIN)).strftime("%H:%M")
        return ("meal", (
            f"Сегодня смена {shift['start']}–{shift['end']}, выходить примерно в {leave}. "
            f"За день нет ни одной записи о еде. Пингани коротко: поесть нормально ДО смены, "
            f"на работе будет одна пицца."
        ))

    # --- 2. Обед в день без смены (или смена поздно вечером) ---
    if (
        (shift_start is None or shift_start.hour >= 19)
        and "питание" not in tags
        and "meal" not in pings
        and now.hour >= 14
    ):
        return ("meal", (
            "Время к обеду, а записей о еде за день нет. Пингани коротко: поел ли, "
            "и если нет — пусть поест по-нормальному, не кофе единым."
        ))

    # --- 3. Тренировка ---
    if "спорт" not in tags and "training" not in pings:
        if shift_start is None and now.hour >= 17:
            return ("training", (
                "Сегодня смены нет. Записей о спорте нет. Спроси коротко: тренька сегодня будет? "
                "Без давления — если скажет нет, принять и не пилить."
            ))
        if shift_start is not None and shift_start.hour >= 18 and 11 <= now.hour < shift_start.hour - 3:
            return ("training", (
                f"Смена сегодня только в {shift['start']} — до неё есть окно. Записей о спорте нет. "
                f"Мягко предложи короткую треньку до работы, если есть силы. Одно сообщение, без давления."
            ))

    # --- 4. Домашняя учёба (ср — вместо туториума, чт — пар нет) ---
    if (
        now.weekday() in HOME_STUDY_WEEKDAYS
        and "study" not in pings
        and not ({"работа", "учёба", "учеба"} & tags)
        and now.hour >= 13
        and (shift_start is None or now < shift_start - timedelta(hours=2))
    ):
        wed_note = (
            "Сегодня среда — туториум по матеше он скипает в пользу занятий дома. "
            if now.weekday() == 2 else ""
        )
        return ("study", (
            f"{wed_note}Записей про учёбу/работу за день нет. Спроси коротко: что сегодня по учёбе, "
            f"какой план? Одно сообщение."
        ))

    return None
