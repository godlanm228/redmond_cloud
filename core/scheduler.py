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

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from core.coordinator import Coordinator
from core.dispatcher import Dispatcher
from logic.agents import IRIS, NEWSER, AgentConfig
from utils.time import OWNER_TZ, now_local

logger = logging.getLogger(__name__)


async def _generate_and_send(
    dispatcher: Dispatcher,
    coordinator: Coordinator,
    chat_id: int,
    agent: AgentConfig,
    prompt: str,
) -> None:
    """Общий путь scheduled-джобы: generate → respond_as. Ошибки не роняют scheduler."""
    try:
        intent = dispatcher.intent_recognizer.recognize(prompt)
        async with coordinator.typing(agent.name, chat_id):
            response = await asyncio.to_thread(
                dispatcher.response_generator.generate,
                intent, prompt, "owner", agent, chat_id,
            )
        if not response:
            logger.warning("Scheduled job for %s: empty response, nothing sent", agent.name)
            return
        await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)
        logger.info("Scheduled job done: %s → chat %s (%d chars)", agent.name, chat_id, len(response))
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


async def evening_summary(dispatcher: Dispatcher, coordinator: Coordinator, chat_id: int) -> None:
    today = now_local().strftime("%Y-%m-%d")
    prompt = (
        f"(scheduled, 22:30, вечерний итог) Сегодня {today}. Прочитай дневник (read_diary, "
        "last_n=15) и возьми только записи за сегодня, плюс цели (list_goals, active). "
        "Короткий итог дня: что зафиксировано (сон/питание/спорт/работа/решения), что с целями. "
        "Если есть активная цель про сон — закончи напоминанием про неё, в своём стиле. "
        "Если за день записей нет — так и скажи одной строкой, без морали."
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

    sched.add_job(morning_digest, CronTrigger(hour=9, minute=0), args=args,
                  id="morning_digest", coalesce=True, misfire_grace_time=600)
    sched.add_job(morning_deadlines, CronTrigger(hour=9, minute=3), args=args,
                  id="morning_deadlines", coalesce=True, misfire_grace_time=600)
    sched.add_job(evening_summary, CronTrigger(hour=22, minute=30), args=args,
                  id="evening_summary", coalesce=True, misfire_grace_time=600)

    return sched
