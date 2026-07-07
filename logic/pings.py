"""
Решение «пинговать или молчать» для дневного тикера. ЧИСТЫЙ Python — LLM
дёргается только когда решение «пинговать» уже принято (экономия токенов).

Анти-спам гарантии (в коде, не на совести LLM):
  • тишина при mute («стоп»/mute_notifications) и ночью (тикер 10–23);
    активити-пинги (еда/спорт/учёба) —
    только после того как Влад на связи сегодня;
  • cold-start (исключение): если день идёт, а Влада не слышно — Iris инициирует
    САМА один раз (мягкое «как ты, какие планы»), можно молча проигнорить;
  • один пинг на слот в день, максимум MAX_PINGS_PER_DAY, пауза ≥ MIN_GAP_MIN;
  • слот закрыт записью в дневнике с нужным тегом → пинга нет
    («поел» → питание; «сегодня без зала» → спорт; и т.д.).
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional, Tuple

from logic import coach_storage
from logic.priorities import crunch_deadline, radar_deadline
from logic.situation_engine import build_day_situation, parse_hm
from logic.week_schedule import (
    HOME_STUDY_WEEKDAYS,
    WORK_COMMUTE_MIN,
)
from utils.time import now_local

logger = logging.getLogger(__name__)

MAX_PINGS_PER_DAY = 5
MIN_GAP_MIN = 90


def decide_ping() -> Optional[Tuple[str, str]]:
    """
    Возвращает (ping_id, контекст для Iris-промпта) или None.
    Вызывающий обязан сразу mark_ping(ping_id) — защита от дублей.
    """
    now = now_local()
    situation = build_day_situation(now)
    pings = situation.pings

    # --- глобальные предохранители (действуют и для cold-start, и для активити) ---
    if not situation.proactive_allowed(MAX_PINGS_PER_DAY, MIN_GAP_MIN):
        return None

    shift = situation.shift.active_record
    shift_start = situation.shift.start_at

    # --- 0. ХОЛОДНЫЙ СТАРТ: Влад сегодня не на связи и записей за день нет, но
    #        день идёт — Iris инициирует САМА, не дожидаясь первого сообщения.
    #        Один раз, мягко, можно проигнорить. Единственный пинг, не требующий,
    #        чтобы Влад уже написал (остальные — после того как он на связи). ---
    if (
        not situation.owner_seen
        and "checkin" not in pings
        and now.hour >= 12
        and (shift_start is None or now < shift_start - timedelta(minutes=40))
    ):
        return ("checkin", (
            "Влад сегодня ещё не на связи, записей за день нет. Поздоровайся тепло, "
            "по-человечески спроси как он и какие планы на день — без давления, без "
            "списка дел. Не ответит — это нормально, не повторяй и не пили."
        ))

    # Активити-пинги (еда/спорт/учёба/дедлайны) — только когда Влад на связи:
    # пинговать про еду пока он молчит/спит бессмысленно.
    if not situation.owner_seen:
        return None

    tags = situation.tags
    has_work_today = situation.has_work_today

    # --- 0b. Утреннее приветствие: ~15 мин после пробуждения, один раз за день.
    #         Триггер — первое сообщение дня (реакция на дайджест и т.п.). Сводку
    #         дня (смена/лекции/дедлайны) Iris берёт из STATE-блока промпта. ---
    wake = situation.wake_time
    if wake and "greeting" not in pings:
        ws = parse_hm(wake, now)
        if ws is not None and now >= ws + timedelta(minutes=15):
            return ("greeting", (
                f"Влад проснулся недавно (в {wake}) и на связи. Поздоровайся тепло и "
                f"коротко, по-человечески, дай сводку дня из STATE (смена/лекции/горящие "
                f"дедлайны если есть) и один лёгкий вопрос про план. Без списка на "
                f"полэкрана, без давления."
            ))

    # --- 0c. Смена есть, но запись старая/неуверенная: один аккуратный чек.
    # Не каждый день: confirmed/high-confidence text shifts молчат; спрашиваем
    # только legacy/uncertain/stale записи и только если Влад уже на связи.
    if (
        situation.shift.needs_confirmation(now)
        and "shift_confirm" not in pings
        and not has_work_today
    ):
        return ("shift_confirm", (
            f"По расписанию сегодня смена {shift['start']}–{shift['end']}. "
            "Аккуратно спроси, всё ли в силе или график поменялся. Одно сообщение, "
            "без давления и без дополнительных советов."
        ))

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

    # --- 3. Crunch: HIGH-дедлайн ≤3 дней, а учёбы за день не было ---
    # Работает в ЛЮБОЙ день недели (горящий тест важнее правила «учёба ср/чт»).
    crunch = crunch_deadline()
    if (
        crunch is not None
        and "crunch" not in pings
        and "study" not in pings
        and not ({"учёба", "учеба"} & tags)
        and now.hour >= 11
        and (shift_start is None or now < shift_start - timedelta(hours=2))
    ):
        left = (crunch["_due"] - now.date()).days
        when = (
            "СЕГОДНЯ" if left == 0
            else f"просрочен на {-left} дн" if left < 0
            else f"через {left} дн ({crunch['due']})"
        )
        return ("crunch", (
            f"Горящий дедлайн: «{crunch['title']}» — {when}. Записей про учёбу за день нет. "
            f"Спроси прямо: когда сегодня сядет за подготовку — и предложи конкретный слот "
            f"по расписанию дня. Один раз, жёстко, но без пиления."
        ))

    # --- 4. Тренировка ---
    if "спорт" not in tags and "training" not in pings and not has_work_today:
        if shift_start is None and 17 <= now.hour < 21:
            return ("training", (
                "Сегодня смены нет. Записей о спорте нет. Спроси коротко: тренька сегодня будет? "
                "Без давления — если скажет нет, принять и не пилить."
            ))
        if shift_start is not None and shift_start.hour >= 18 and 11 <= now.hour < shift_start.hour - 3:
            return ("training", (
                f"Смена сегодня только в {shift['start']} — до неё есть окно. Записей о спорте нет. "
                f"Мягко предложи короткую треньку до работы, если есть силы. Одно сообщение, без давления."
            ))

    # --- 5. Домашняя учёба (ср — вместо туториума, чт — пар нет) ---
    if (
        now.weekday() in HOME_STUDY_WEEKDAYS
        and "study" not in pings
        and "crunch" not in pings  # crunch уже пинганул про учёбу — не дублируем
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

    # --- 6. Радар: дедлайн через 4-7 дней, ещё не предупреждали (одноразово, мягко) ---
    rad = radar_deadline()
    if (
        rad is not None
        and "radar" not in pings
        and not ({"учёба", "учеба"} & tags)
        and now.hour >= 11
        and (shift_start is None or now < shift_start - timedelta(hours=2))
    ):
        coach_storage.mark_radar(rad["id"])
        left = (rad["_due"] - now.date()).days
        return ("radar", (
            f"На радаре дедлайн «{rad['title']}» — через {left} дн ({rad['due']}). "
            f"Ещё не горит, но спроси мягко: начал ли, нужен ли слот в плане недели. "
            f"Один раз, без давления."
        ))

    return None
