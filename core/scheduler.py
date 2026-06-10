"""
Proactive scheduler — APScheduler-джобы поверх готового LLM-pipeline.

Джобы (все в Europe/Berlin, cron в коде — персистентность не нужна,
расписание статично):
  • 09:00 — Newser шлёт утренний дайджест (get_news_headlines all).
  • 09:03 — Iris: дедлайны на сегодня/завтра + просроченные.
            Если их нет — молчит (проверка в Python, не в LLM — без спама).
  • 22:30 — Iris: вечерний итог дня (дневник за сегодня, цели, сон).

Генерация — тот же ResponseGenerator/tool-loop, что и у чат-хендлеров:
scheduled-prompt уходит как user-сообщение агента, ответ — через
Coordinator от имени агента. Iris знает про «(scheduled …)» из промпта.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.coordinator import Coordinator
from core.dispatcher import Dispatcher
from logic.agents import IRIS, NEWSER, AgentConfig
from utils.time import OWNER_TZ, now_local

logger = logging.getLogger(__name__)


_JOB_TIMEOUT_SEC = 200.0


async def _generate_and_send(
    dispatcher: Dispatcher,
    coordinator: Coordinator,
    chat_id: int,
    agent: AgentConfig,
    prompt: str,
) -> None:
    """Общий путь scheduled-джобы: generate → respond_as. Ошибки/зависания не роняют scheduler."""
    try:
        intent = dispatcher.intent_recognizer.recognize(prompt)
        async with coordinator.typing(agent.name, chat_id):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    dispatcher.response_generator.generate,
                    intent, prompt, "owner", agent, chat_id,
                ),
                timeout=_JOB_TIMEOUT_SEC,
            )
        if not response:
            logger.warning("Scheduled job for %s: empty response, nothing sent", agent.name)
            return
        await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)
        logger.info("Scheduled job done: %s → chat %s (%d chars)", agent.name, chat_id, len(response))
    except asyncio.TimeoutError:
        logger.warning("Scheduled job for %s timed out (%ds)", agent.name, _JOB_TIMEOUT_SEC)
    except Exception:
        logger.exception("Scheduled job failed for %s", agent.name)


# ---------- джобы ----------

async def morning_digest(dispatcher: Dispatcher, coordinator: Coordinator, chat_id: int) -> None:
    prompt = (
        "(scheduled, утренний дайджест) Вызови get_news_headlines с category='all' "
        "и собери утреннюю сводку по секциям. Короткое приветствие одной строкой, без вопросов."
    )
    await _generate_and_send(dispatcher, coordinator, chat_id, NEWSER, prompt)


def _deadlines_needing_attention() -> list:
    """Просроченные pending + дедлайны на сегодня/завтра. Пусто = джоба молчит."""
    from logic.coach_storage import list_deadlines
    today = now_local().date()
    attention = []
    for d in list_deadlines():
        if d.get("status") != "pending":
            continue
        try:
            due = datetime.strptime(d.get("due", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        if due <= today + timedelta(days=1):
            attention.append(d)
    return attention


async def morning_deadlines(dispatcher: Dispatcher, coordinator: Coordinator, chat_id: int) -> None:
    attention = _deadlines_needing_attention()
    if not attention:
        logger.info("Morning deadlines: nothing urgent, staying silent")
        return
    listing = "; ".join(f"#{d['id']} «{d['title']}» → {d['due']}" for d in attention)
    prompt = (
        f"(scheduled, утро) Дедлайны, требующие внимания: {listing}. "
        "Напомни Владу о них коротко и по делу — что горит сегодня/завтра, что просрочено."
    )
    await _generate_and_send(dispatcher, coordinator, chat_id, IRIS, prompt)


async def day_ticker(dispatcher: Dispatcher, coordinator: Coordinator, chat_id: int) -> None:
    """
    Каждые 30 мин (10–23): чистый Python решает нужен ли пинг (еда/тренька/учёба
    с учётом смен и пар), LLM зовём только при решении «да». mark_ping ДО генерации —
    защита от дублей при медленном Groq.
    """
    from logic.coach_storage import mark_ping
    from logic.pings import decide_ping

    decision = decide_ping()
    if decision is None:
        return
    ping_id, context_text = decision
    mark_ping(ping_id)
    logger.info("Day ticker: ping «%s»", ping_id)
    prompt = f"(scheduled, пинг дня) {context_text} В твоём стиле, коротко."
    await _generate_and_send(dispatcher, coordinator, chat_id, IRIS, prompt)


async def evening_summary(dispatcher: Dispatcher, coordinator: Coordinator, chat_id: int) -> None:
    today = now_local().strftime("%Y-%m-%d")
    prompt = (
        f"(scheduled, 22:30, вечерний итог) Сегодня {today}. Прочитай дневник (read_diary, "
        "last_n=15) и возьми только записи за сегодня, плюс цели (list_goals, active). "
        "Итог дня в двух частях: (1) коротко факты — что зафиксировано "
        "(сон/питание/спорт/работа/учёба/решения), что с целями; отдельно отметь "
        "наблюдения от Redmond, если есть (записи с тегом «от Redmond»: "
        "обязательство/состояние/паттерн/факт) — что он за день передал тебе про Влада; "
        "(2) характер дня — одной-двумя строками: каким он получился "
        "(плотный/лёгкий/чиловый/продуктивный/рваный) и тёплая честная строка по факту: "
        "поработал/позанимался — признай прямо («день вышел плотный, поработал хорошо»), "
        "отдыхал — «чиловый день, имеешь право». Это ситуативное тепло по реальным "
        "записям, не pep-talk. Пустой день — одна строка без морали. "
        "Если есть активная цель про сон — закончи напоминанием про неё, в своём стиле."
    )
    await _generate_and_send(dispatcher, coordinator, chat_id, IRIS, prompt)


# ---------- сборка ----------

def setup_scheduler(
    dispatcher: Dispatcher,
    coordinator: Coordinator,
    chat_id: int,
) -> AsyncIOScheduler:
    """Создаёт scheduler с джобами. Запуск — sched.start() в работающем event loop."""
    sched = AsyncIOScheduler(timezone=OWNER_TZ or "Europe/Berlin")
    args = [dispatcher, coordinator, chat_id]

    # grace_time щедрый: даже если event loop был занят, джоба ВЫПОЛНИТСЯ
    # с опозданием, а не пропустится молча (так пропал вечерний итог 10.06).
    sched.add_job(morning_digest, CronTrigger(hour=9, minute=0), args=args,
                  id="morning_digest", coalesce=True, misfire_grace_time=3600)
    sched.add_job(morning_deadlines, CronTrigger(hour=9, minute=3), args=args,
                  id="morning_deadlines", coalesce=True, misfire_grace_time=3600)
    sched.add_job(evening_summary, CronTrigger(hour=22, minute=30), args=args,
                  id="evening_summary", coalesce=True, misfire_grace_time=3600)
    sched.add_job(day_ticker, CronTrigger(hour="10-22", minute="0,30"), args=args,
                  id="day_ticker", coalesce=True, misfire_grace_time=600)

    # Диагностика: явно логируем выполнение/пропуск/ошибку джоб. Раньше misfire
    # был невидим (apscheduler-логгер подавлен), причину сбоя нельзя было понять.
    def _on_job_event(event) -> None:
        if event.code == EVENT_JOB_MISSED:
            logger.warning("Scheduler: job %s MISSED (loop был занят дольше grace)", event.job_id)
        elif event.code == EVENT_JOB_ERROR:
            logger.error("Scheduler: job %s ERROR: %s", event.job_id, event.exception)
        else:
            logger.debug("Scheduler: job %s executed", event.job_id)

    sched.add_listener(_on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)

    return sched
