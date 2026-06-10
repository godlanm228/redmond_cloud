"""
Handlers для multi-bot режима (4 параллельных Application'а).

Архитектура:
  • Все 4 бота слушают одну группу через polling (Privacy OFF у всех).
  • Каждый Application получает свой апдейт от TG (Iris-апдейт идёт в Iris-app и т.д.).
  • Redmond — главный router (если нет явного @-меншина, решает кому отвечать).
  • Iris/Cipher/Newser — slim handler: реагируют только на свой @-меншин
    (только от Влада; делегирование агент→агент идёт in-process).

Авторизация:
  • Обрабатываем только сообщения Влада (user_id in ALLOWED_USER_IDS) в MAIN_CHAT.
  • Всё остальное — игнор. Telegram НЕ доставляет ботам сообщения других ботов
    (ограничение Bot API), поэтому межагентное делегирование живёт in-process
    (_run_delegation), а @-меншен в чате — витрина для владельца.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional, Set

from telegram import Update
from telegram.ext import ContextTypes

from logic.agents import (
    AgentConfig,
    REDMOND,
    agent_by_name,
    find_by_trigger,
)
from logic.agent_router import RouterState, llm_research_flag, route
from logic.tools import DELEGATION_MARKER

logger = logging.getLogger(__name__)


# ---------- конфиг ----------

def _allowed_user_ids() -> set:
    raw = os.getenv("ALLOWED_USER_IDS") or ""
    return {int(x.strip()) for x in raw.split(",") if x.strip().isdigit()}


def _main_chat_id() -> Optional[int]:
    raw = os.getenv("MAIN_CHAT_ID")
    if not raw:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


# ---------- авторизация ----------

# Multilingual фраза при попытке использовать бота вне Redberry HUB.
# Отправляется ОДИН раз на chat_id, потом silent (anti-DDoS).
_OUTSIDE_MESSAGE = (
    "🔒 I work only in the Redberry HUB chat.\n"
    "🔒 Я работаю только в чате Redberry HUB.\n"
    "🔒 Я працюю лише в чаті Redberry HUB.\n"
    "🔒 Ich arbeite nur im Redberry-HUB-Chat."
)


def _is_from_owner(update: Update) -> bool:
    """True если сообщение лично от Влада (не от бота, известный ID)."""
    user = update.effective_user
    return user is not None and not user.is_bot and user.id in _allowed_user_ids()


async def _gate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """
    Главные ворота безопасности.
    Возвращает True если можно обрабатывать сообщение.

    Правила:
      1. Сообщение должно быть от Влада (боты сообщений других ботов
         не получают вообще — Bot API, проверять нечего).
      2. Чат должен быть MAIN_CHAT_ID (если задан в env).
      3. Если правила нарушены и это первое нарушение в этом чате —
         отправляем one-time multilingual фразу. Дальше silent.
    """
    user = update.effective_user
    chat = update.effective_chat
    if user is None or chat is None:
        return False

    if not _is_from_owner(update):
        # Не Влад — silent ignore всегда (даже без one-time message).
        # Если ответим — раскрываем что бот живой, плюс лишняя нагрузка.
        return False

    # Первое сообщение Влада за день (в окне 05-14) = «проснулся» → дневник.
    # Идемпотентно по дате, поэтому 4 параллельных _gate не плодят дублей.
    try:
        from logic.coach_storage import log_wake_if_first
        entry = log_wake_if_first()
        if entry:
            logger.info("Wake detected: %s", entry["text"])
    except Exception:
        logger.debug("wake detection failed", exc_info=True)

    main_chat = _main_chat_id()
    if main_chat is not None and chat.id != main_chat:
        # Влад пишет, но НЕ в Redberry HUB: one-time multilingual ответ → дальше silent.
        warned: Set[int] = context.application.bot_data.setdefault("warned_chats", set())
        if chat.id not in warned:
            warned.add(chat.id)
            try:
                await update.message.reply_text(_OUTSIDE_MESSAGE)
            except Exception:
                logger.debug("Failed to send outside-chat warning", exc_info=True)
            logger.info(
                "Outside-chat attempt: chat_id=%s user_id=%s — sent one-time warning",
                chat.id, user.id,
            )
        else:
            logger.debug("Outside-chat repeat (silent): chat_id=%s", chat.id)
        return False

    return True


# ---------- генерация ответа (общий код) ----------

async def _generate(
    agent: AgentConfig,
    user_text: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int = 0,
    status_cb=None,
    force_tool: Optional[str] = None,
) -> str:
    """
    Вызывает response_generator для указанного агента.
    chat_id → per-chat history (изоляция Iris/Newser/Redmond контекстов).
    status_cb — живой статус из tool-loop; force_tool — принудительное
    делегирование (классификатор решил «research»). Cipher идёт через subprocess.
    """
    if agent.executor == "cipher_subprocess":
        return await _generate_cipher(user_text, context)

    bot_data = context.application.bot_data
    dispatcher = bot_data["dispatcher"]
    role = "owner"

    try:
        dispatcher.safety.assert_safe(user_text)
    except Exception as e:
        logger.warning("Unsafe input from %s: %s", agent.name, e)
        return "Не могу выполнить эту команду."

    intent = dispatcher.intent_recognizer.recognize(user_text)

    try:
        response = await asyncio.to_thread(
            dispatcher.response_generator.generate,
            intent,
            user_text,
            role,
            agent,
            chat_id,
            status_cb=status_cb,
            force_tool=force_tool,
        )
    except Exception:
        logger.exception("Generation failed for %s", agent.name)
        return "⚠ Внутренняя ошибка генерации."

    return response or "Не знаю, что ответить."


async def _generate_with_status(
    agent: AgentConfig,
    user_text: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    force_tool: Optional[str] = None,
) -> str:
    """
    Генерация + живой статус из РЕАЛЬНЫХ tool calls: первый вызов тула постит
    сообщение («📰 ищу: курс биткоина…»), следующие редактируют его же,
    по готовности ответа статус удаляется (delete+send — новая нотификация).
    Болтовня без tools не постит ничего — только typing indicator. Статус
    не попадает в history: шлётся напрямую ботом, мимо generate.
    """
    coordinator = context.application.bot_data["coordinator"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def status_cb(text: str) -> None:
        # Вызывается из generation-потока (to_thread) — мост в event loop
        loop.call_soon_threadsafe(queue.put_nowait, text)

    status_msg = None

    async def _status_watcher() -> None:
        nonlocal status_msg
        bot = coordinator.bot_for(agent.name)
        if bot is None:
            return
        while True:
            text = await queue.get()
            body = f"{agent.emoji} {text}"
            try:
                if status_msg is None:
                    status_msg = await bot.send_message(chat_id=chat_id, text=body)
                else:
                    await status_msg.edit_text(body)
            except Exception:
                logger.debug("status update failed", exc_info=True)

    watcher = asyncio.create_task(_status_watcher())
    try:
        response = await _generate(
            agent, user_text, context, chat_id,
            status_cb=status_cb, force_tool=force_tool,
        )
    finally:
        watcher.cancel()
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                logger.warning(
                    "status delete failed (chat %s, msg %s) — останется висеть",
                    chat_id, status_msg.message_id,
                )
    return response


async def _run_delegation(
    delegator: AgentConfig,
    raw_payload: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> None:
    """
    Handoff-делегирование Ньюсеру. In-process: Telegram НЕ доставляет ботам
    сообщения других ботов, поэтому @-меншен в чате — витрина для владельца,
    а сама задача передаётся прямым вызовом генерации.
    Ход делегатора окончен — финальный ответ постит только Newser.
    """
    coordinator = context.application.bot_data["coordinator"]
    router_states: dict = context.application.bot_data["router_states"]
    newser = agent_by_name("Newser")
    if newser is None:
        logger.error("Delegation failed: Newser agent not registered")
        return

    try:
        payload = json.loads(raw_payload or "{}")
    except json.JSONDecodeError:
        payload = {}
    task = str(payload.get("task", "")).strip()
    region = str(payload.get("region", "")).strip()
    if not task:
        logger.warning("Delegation with empty task from %s — ignored", delegator.name)
        return

    logger.info("DELEGATE [%s → Newser]: %s", delegator.name, task[:120])

    # Витрина: кто кому что передал. Ты видишь таск и можешь поправить.
    await coordinator.respond_as(
        delegator.name, chat_id,
        f"Это к Ньюсеру — @{newser.bot_username}, найди: {task}",
        delegator.emoji, "plain",
    )

    envelope = (
        f"(delegated by {delegator.name} on behalf of owner)\n"
        f"TASK: {task}\n"
        + (f"REGION HINT: {region}\n" if region else "")
    )
    async with coordinator.typing(newser.name, chat_id):
        response = await _generate_with_status(newser, envelope, context, chat_id)

    # Sticky на Newser: «а подробнее?» следом уйдёт ему — контекст рисёрча у него.
    state: RouterState = router_states.setdefault(chat_id, RouterState())
    state.add("assistant", response, newser.name)
    state.last_agent_name = newser.name

    await coordinator.respond_as(newser.name, chat_id, response, newser.emoji, newser.output_format)


async def _generate_cipher(user_text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """
    Stub для Cipher (Claude Code CLI subprocess).
    Пока возвращает заглушку — реальная имплементация в core/cipher_wrapper.py
    после того как Node.js + Claude CLI установлены на VM.
    """
    return (
        "🚧 Cipher пока не подключён к Claude Code CLI на этой VM. "
        "Поставь Node.js + `claude login` под Pro подпиской, и я заработаю."
    )


# ---------- Photo handler (скрин графика смен → расписание) ----------

async def redmond_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Фото от Влада в HUB = скрин графика смен (других кейсов с картинками пока нет).
    Groq vision разбирает смены → shifts.json → подтверждение списком в чат.
    """
    if not await _gate(update, context):
        return
    if not _is_from_owner(update) or not update.message or not update.message.photo:
        return
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    coordinator = context.application.bot_data["coordinator"]
    dispatcher = context.application.bot_data["dispatcher"]

    import base64
    photo = update.message.photo[-1]  # максимальное разрешение
    tg_file = await photo.get_file()
    raw = bytes(await tg_file.download_as_bytearray())
    image_b64 = base64.b64encode(raw).decode()

    logger.info("Photo from owner (%d KB) — parsing as shift schedule", len(raw) // 1024)
    from logic.week_schedule import ingest_shift_screenshot
    async with coordinator.typing("Redmond", chat_id):
        result_text, saved_n = await asyncio.to_thread(
            ingest_shift_screenshot, image_b64, dispatcher.config.groq_api_key,
        )
    await coordinator.respond_as("Redmond", chat_id, result_text, "🦞", "html")

    # Смены загружены → Iris сразу составляет план недели (учёба под дедлайны,
    # треньки в лёгкие дни, 1-2 защищённых вечера). Правится потом словами.
    if saved_n > 0:
        iris = agent_by_name("Iris")
        if iris is not None:
            plan_prompt = (
                "(scheduled week-plan) Влад загрузил новый график смен. Составь "
                "план недели по правилам WEEK PLANNING и сохрани его."
            )
            async with coordinator.typing(iris.name, chat_id):
                plan = await _generate_with_status(iris, plan_prompt, context, chat_id)
            if plan.startswith(DELEGATION_MARKER):
                await _run_delegation(iris, plan[len(DELEGATION_MARKER):], context, chat_id)
            else:
                await coordinator.respond_as(iris.name, chat_id, plan, iris.emoji, iris.output_format)


# ---------- Redmond handler (с router) ----------

async def redmond_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Redmond — единственный с router'ом.

    Логика:
      1) Авторизация
      2) Если в тексте @-меншин ДРУГОГО агента → молчу (тот ответит сам)
      3) Если @redmond или нет триггера → запускаю router
         - router выбрал Redmond → отвечаю сам
         - router выбрал другого → генерирую ответ под того агента
           и отправляю через coordinator от его имени (видимая «делегация»)
    """
    if not await _gate(update, context):
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    bot_data = context.application.bot_data
    coordinator = bot_data["coordinator"]
    dispatcher = bot_data["dispatcher"]
    router_states: dict = bot_data["router_states"]

    # 1. Явный @-меншин кого-то другого → этот апдейт не для Redmond
    explicit = find_by_trigger(text)
    if explicit is not None and explicit.name != "Redmond":
        return

    # 2. Снимаем @redmond префикс если есть
    clean_text = explicit.strip_trigger(text) if explicit else text

    # 3. Routing (только если не было явного @redmond — иначе оставляем Redmond).
    # research-флаг считает классификатор (8b) — решение о делегировании в коде,
    # не на воле 120b-модели.
    state: RouterState = router_states.setdefault(chat_id, RouterState())
    groq_key = dispatcher.config.groq_api_key
    if explicit is not None:
        agent = REDMOND
        needs_research = llm_research_flag(clean_text, state, groq_key)
    else:
        agent, needs_research = route(text, state, groq_api_key=groq_key)
    state.add("user", clean_text, agent.name)

    user = update.effective_user
    logger.info(
        "IN [%s %s → %s]: %s",
        user.id,
        user.first_name or user.username,
        agent.name,
        clean_text[:120],
    )

    # 4. Генерация и отправка — typing indicator пока работаем.
    # research + Redmond = принудительное делегирование: единственный tool в
    # вызове — delegate_research с tool_choice forced (модель только
    # формулирует таск с контекстом, решение уже принято).
    force_tool = "delegate_research" if (agent.name == "Redmond" and needs_research) else None
    async with coordinator.typing(agent.name, chat_id):
        response = await _generate_with_status(agent, clean_text, context, chat_id, force_tool=force_tool)

    # Агент делегировал → его ход окончен, оркеструем handoff (ответит Newser)
    if response.startswith(DELEGATION_MARKER):
        await _run_delegation(agent, response[len(DELEGATION_MARKER):], context, chat_id)
        return

    if force_tool:
        # Модель умудрилась ответить текстом вопреки forced tool_choice
        # (или вся цепочка моделей упала) — жёсткий fallback кодом:
        # делегируем с сырым текстом как таском. Гарантия, не пожелание.
        logger.warning("Forced delegation bypassed (%s) — hard fallback", agent.name)
        await _run_delegation(
            agent, json.dumps({"task": clean_text, "region": ""}, ensure_ascii=False),
            context, chat_id,
        )
        return

    state.add("assistant", response, agent.name)
    await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)


# ---------- Slim handler (Iris/Cipher/Newser) ----------

async def slim_agent_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Реагирует только на свой @-меншин. Используется для всех агентов кроме Redmond.

    Логика:
      1) Авторизация (только Влад)
      2) Этот апдейт содержит @-меншин моего агента? Если нет — молчу
      3) Если меншин чужого агента в начале → тоже молчу (это не мне)
      4) Генерирую ответ, отправляю через coordinator (от своего токена)
    """
    if not await _gate(update, context):
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    bot_data = context.application.bot_data
    agent: AgentConfig = bot_data["agent"]
    coordinator = bot_data["coordinator"]
    router_states: dict = bot_data["router_states"]

    # Триггер — должен быть мой (через @-меншин ИЛИ через name-alias типа «Айрис,»)
    if not agent.matches_trigger(text):
        return

    clean_text = agent.strip_trigger(text)
    # Если осталась пустота (юзер написал просто «Iris» без вопроса) — поприветствуем
    if not clean_text:
        clean_text = "(привет)"

    user = update.effective_user
    logger.info(
        "IN [%s %s → %s] (slim): %s",
        user.id,
        user.first_name or user.username or "?",
        agent.name,
        clean_text[:120],
    )

    async with coordinator.typing(agent.name, chat_id):
        response = await _generate_with_status(agent, clean_text, context, chat_id)

    state: RouterState = router_states.setdefault(chat_id, RouterState())
    state.add("user", clean_text, agent.name)

    # Агент делегировал (например Iris → Newser за внешним фактом)
    if response.startswith(DELEGATION_MARKER):
        await _run_delegation(agent, response[len(DELEGATION_MARKER):], context, chat_id)
        return

    # Обновляем router_states — это критично для sticky.
    # Чтобы когда Влад следом напишет без @-меншина, Redmond router увидел
    # «только что отвечала Iris» и сохранил Iris (если LLM согласен).
    state.add("assistant", response, agent.name)
    state.last_agent_name = agent.name

    await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)
