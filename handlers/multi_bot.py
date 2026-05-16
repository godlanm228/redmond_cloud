"""
Handlers для multi-bot режима (4 параллельных Application'а).

Архитектура:
  • Все 4 бота слушают одну группу через polling (Privacy OFF у всех).
  • Каждый Application получает свой апдейт от TG (Iris-апдейт идёт в Iris-app и т.д.).
  • Redmond — главный router (если нет явного @-меншина, решает кому отвечать).
  • Iris/Cipher/Newser — slim handler: реагируют только на свой @-меншин
    (от Влада ИЛИ от другого нашего бота — это видимое делегирование).

Авторизация:
  • Сообщения от Влада (user_id in ALLOWED_USER_IDS) — обрабатываем.
  • Сообщения от наших ботов (is_bot + username из all_bot_usernames()) — обрабатываем
    (поддержка inter-bot делегирования: Cipher пишет "@newser найди X" → Newser отвечает).
  • Всё остальное — игнор.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Set

from telegram import Update
from telegram.ext import ContextTypes

from logic.agents import (
    AgentConfig,
    REDMOND,
    agent_by_name,
    all_bot_usernames,
    find_by_trigger,
)
from logic.agent_router import RouterState, route

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


_OUR_BOTS_LOWER: Optional[set] = None


def _our_bots_lower() -> set:
    global _OUR_BOTS_LOWER
    if _OUR_BOTS_LOWER is None:
        _OUR_BOTS_LOWER = {u.lower() for u in all_bot_usernames()}
    return _OUR_BOTS_LOWER


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


def _is_from_our_bot(update: Update) -> bool:
    """True если сообщение от одного из наших ботов (inter-bot делегирование)."""
    user = update.effective_user
    return (
        user is not None
        and user.is_bot
        and (user.username or "").lower() in _our_bots_lower()
    )


async def _gate(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    """
    Главные ворота безопасности.
    Возвращает True если можно обрабатывать сообщение.

    Правила:
      1. Сообщение должно быть от Влада ИЛИ от нашего бота.
      2. Чат должен быть MAIN_CHAT_ID (если задан в env).
      3. Если правила нарушены и это первое нарушение в этом чате —
         отправляем one-time multilingual фразу. Дальше silent.
    """
    user = update.effective_user
    chat = update.effective_chat
    if user is None or chat is None:
        return False

    is_owner = _is_from_owner(update)
    is_our_bot = _is_from_our_bot(update)
    if not (is_owner or is_our_bot):
        # Не Влад и не наш бот — silent ignore всегда (даже без one-time message).
        # Если ответим — раскрываем что бот живой, плюс лишняя нагрузка.
        return False

    main_chat = _main_chat_id()
    if main_chat is not None and chat.id != main_chat:
        # Влад (или наш бот) пишет, но НЕ в Redberry HUB.
        # One-time multilingual ответ → дальше silent.
        # Inter-bot сообщения (is_our_bot) НЕ должны попадать сюда —
        # они шлются coordinator'ом строго в MAIN_CHAT. Но если попало — тоже silent.
        if is_our_bot:
            return False

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
) -> str:
    """
    Вызывает response_generator для указанного агента.
    chat_id → per-chat history (изоляция Iris/Newser/Redmond контекстов).
    Cipher идёт через subprocess.
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
        )
    except Exception:
        logger.exception("Generation failed for %s", agent.name)
        return "⚠ Внутренняя ошибка генерации."

    return response or "Не знаю, что ответить."


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

    # 3. Routing (только если не было явного @redmond — иначе оставляем Redmond)
    state: RouterState = router_states.setdefault(chat_id, RouterState())
    if explicit is not None:
        agent = REDMOND
    else:
        groq_key = dispatcher.config.groq_api_key
        agent = route(text, state, groq_api_key=groq_key)
    state.add("user", clean_text, agent.name)

    user = update.effective_user
    logger.info(
        "IN [%s %s → %s]: %s",
        user.id,
        user.first_name or user.username,
        agent.name,
        clean_text[:120],
    )

    # 4. Генерация и отправка — typing indicator пока работаем
    async with coordinator.typing(agent.name, chat_id):
        response = await _generate(agent, clean_text, context, chat_id)
    state.add("assistant", response, agent.name)

    await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)


# ---------- Slim handler (Iris/Cipher/Newser) ----------

async def slim_agent_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Реагирует только на свой @-меншин. Используется для всех агентов кроме Redmond.

    Логика:
      1) Авторизация (Влад или другой наш бот)
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
        response = await _generate(agent, clean_text, context, chat_id)

    # Обновляем router_states — это критично для sticky.
    # Чтобы когда Влад следом напишет без @-меншина, Redmond router увидел
    # «только что отвечала Iris» и сохранил Iris (если LLM согласен).
    state: RouterState = router_states.setdefault(chat_id, RouterState())
    state.add("user", clean_text, agent.name)
    state.add("assistant", response, agent.name)
    state.last_agent_name = agent.name

    await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)
