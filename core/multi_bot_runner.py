"""
Multi-bot runner — собирает 4 Application и запускает их параллельно.

Каждый Application слушает свой токен. Все они делят:
  • Dispatcher (auth/safety/intent + response_generator) — один на всех
  • Coordinator (мап agent_name → Bot) — для cross-agent отправки
  • router_states (per-chat sticky) — общий dict, ключ chat_id

Запуск через asyncio.gather на 4 Application'ах.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from core.coordinator import Coordinator
from core.dispatcher import Dispatcher
from handlers.multi_bot import _gate, redmond_handler, redmond_photo_handler, slim_agent_handler
from logic.agents import AGENTS, AgentConfig

logger = logging.getLogger(__name__)


# Маппинг agent.name → env var, в котором лежит токен.
TOKEN_ENV_BY_AGENT: Dict[str, str] = {
    "Redmond": "TELEGRAM_BOT_TOKEN",
    "Iris": "IRIS_BOT_TOKEN",
    "Cipher": "CIPHER_BOT_TOKEN",
    "Newser": "NEWSER_BOT_TOKEN",
}


def _resolve_token(agent: AgentConfig) -> Optional[str]:
    env_name = TOKEN_ENV_BY_AGENT.get(agent.name)
    if not env_name:
        return None
    return os.getenv(env_name) or None


# ---------- common commands ----------

async def cmd_ping(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update, context):
        return
    if update.message:
        await update.message.reply_text("pong 🏓")


async def cmd_whoami(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update, context):
        return
    if not update.message:
        return
    user = update.effective_user
    chat = update.effective_chat
    chat_info = f"\nChat ID: `{chat.id}`\nChat type: {chat.type}" if chat else ""
    await update.message.reply_text(
        f"User ID: `{user.id}`\n"
        f"Username: @{user.username or '—'}\n"
        f"Name: {user.first_name}"
        f"{chat_info}",
        parse_mode="Markdown",
    )


# ---------- build single app ----------

def _build_app(
    agent: AgentConfig,
    token: str,
    dispatcher: Dispatcher,
    coordinator: Coordinator,
    shared_router_states: dict,
) -> Application:
    app = Application.builder().token(token).build()

    app.bot_data["agent"] = agent
    app.bot_data["dispatcher"] = dispatcher
    app.bot_data["coordinator"] = coordinator
    app.bot_data["router_states"] = shared_router_states

    # /ping и /whoami — только у Redmond, иначе все 4 бота отвечают на одну команду
    if agent.name == "Redmond":
        app.add_handler(CommandHandler("ping", cmd_ping))
        app.add_handler(CommandHandler("whoami", cmd_whoami))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, redmond_handler))
        # Скрин графика смен от Влада → vision-парсинг (только Redmond, чтобы 4 бота
        # не обрабатывали одно фото параллельно)
        app.add_handler(MessageHandler(filters.PHOTO, redmond_photo_handler))
    else:
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, slim_agent_handler))

    return app


# ---------- run all ----------

async def run_multi_bot(dispatcher: Dispatcher) -> None:
    """
    Стартует 4 параллельных polling-loop в одном процессе.
    Блокируется до получения сигнала остановки.
    """
    # 1. Резолвим токены
    apps_to_run: List[Tuple[AgentConfig, str]] = []
    missing: List[str] = []
    for agent in AGENTS:
        token = _resolve_token(agent)
        if not token:
            missing.append(f"{agent.name} ({TOKEN_ENV_BY_AGENT.get(agent.name, '?')})")
            continue
        apps_to_run.append((agent, token))

    if missing:
        logger.warning("No token for: %s — эти боты не запущены", ", ".join(missing))

    if not apps_to_run:
        raise RuntimeError("Нет ни одного токена. Проверь .env (TELEGRAM_BOT_TOKEN и др.)")

    # 2. Создаём Bot instances для Coordinator (использует те же токены)
    bots: Dict[str, Bot] = {agent.name: Bot(token=token) for agent, token in apps_to_run}
    coordinator = Coordinator(bots)

    # 3. Общий router_states между всеми Application
    shared_router_states: dict = {}

    # 4. Собираем Application'ы
    apps: List[Application] = []
    for agent, token in apps_to_run:
        app = _build_app(agent, token, dispatcher, coordinator, shared_router_states)
        apps.append(app)
        logger.info("Built Application for %s (@%s)", agent.name, agent.bot_username)

    # 5. Инициализация + старт
    for app in apps:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("Polling started: %s", app.bot_data["agent"].name)

    logger.info("All %d bots are listening", len(apps))

    # 6. Proactive scheduler (утренний дайджест, дедлайны, вечерний итог)
    scheduler = None
    main_chat_raw = os.getenv("MAIN_CHAT_ID")
    if main_chat_raw:
        try:
            from core.scheduler import setup_scheduler
            scheduler = setup_scheduler(dispatcher, coordinator, int(main_chat_raw))
            scheduler.start()
            logger.info(
                "Scheduler started: %s",
                ", ".join(j.id for j in scheduler.get_jobs()),
            )
        except Exception:
            scheduler = None
            logger.exception("Scheduler failed to start — боты работают без проактивных джоб")
    else:
        logger.warning("MAIN_CHAT_ID not set — scheduler disabled")

    # 7. Держим живыми до cancel
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received")
    finally:
        if scheduler is not None:
            try:
                scheduler.shutdown(wait=False)
            except Exception:
                logger.debug("Scheduler shutdown failed", exc_info=True)
        for app in reversed(apps):
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                logger.exception("Error during shutdown of %s", app.bot_data["agent"].name)
        logger.info("All bots stopped")
