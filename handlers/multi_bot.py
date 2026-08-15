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
from logic.agent_router import (RouterState, get_state, llm_research_flag,
                                reply_target_agent, route)
from logic.tools import DELEGATION_MARKER

logger = logging.getLogger(__name__)

# Потолок на одну генерацию. Groq-вызов сам ограничен 40с × ≤5 хопов × 2 модели;
# этот таймаут — внешняя страховка, чтобы зависшая генерация не держала хендлер.
GENERATION_TIMEOUT_SEC = 200.0


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
        logger.warning("wake detection failed", exc_info=True)

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

    # Влад на связи в HUB сегодня — фиксируем (питает cold-start тикера Iris).
    try:
        from logic.coach_storage import mark_owner_seen
        mark_owner_seen()
    except Exception:
        logger.warning("mark_owner_seen failed", exc_info=True)

    return True


# ---------- mute: стоп-слово кодом, не LLM (надёжность — «Ацтань тудей»
# LLM однажды прочитал как просьбу построить план дня) ----------

# Точное совпадение всего сообщения (после нормализации пунктуации/регистра).
_MUTE_WORDS = {"стоп", "stop", "не пиши", "не пиши сегодня", "не пиши мне", "тишина"}
_UNMUTE_WORDS = {"пиши", "говори", "отмена тишины"}


def _mute_command_response(text: str) -> Optional[str]:
    """Стоп-слово/анмьют точным матчем. Возвращает готовый ответ или None
    (обычное сообщение — идёт в LLM). «пиши» снимает mute только если он
    активен, иначе это обычное слово."""
    from logic.coach_storage import muted_now, set_mute, unmute
    norm = " ".join(
        text.lower().replace(",", " ").replace(".", " ").replace("!", " ").split()
    )
    if norm in _MUTE_WORDS:
        desc = set_mute("today")
        return f"🔇 Принято: тишина {desc}. Вернуть — «пиши» или /unmute."
    if norm in _UNMUTE_WORDS and muted_now():
        unmute()
        return "🔊 Тишина снята, снова на связи."
    return None


async def cmd_mute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/mute — тишина до конца дня; /mute N — на N часов; /mute forever — насовсем."""
    if not await _gate(update, context):
        return
    from logic.coach_storage import set_mute
    arg = (context.args[0].lower() if context.args else "")
    if arg == "forever":
        desc = set_mute("forever")
    elif arg.replace(".", "", 1).isdigit():
        desc = set_mute("hours", hours=float(arg))
    else:
        desc = set_mute("today")
    await update.message.reply_text(f"🔇 Принято: тишина {desc}. Вернуть — /unmute.")


async def cmd_unmute(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _gate(update, context):
        return
    from logic.coach_storage import unmute
    unmute()
    await update.message.reply_text("🔊 Тишина снята, снова на связи.")


# ---------- генерация ответа (общий код) ----------

async def _generate(
    agent: AgentConfig,
    user_text: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int = 0,
    status_cb=None,
    force_tool: Optional[str] = None,
    meta: Optional[dict] = None,
) -> str:
    """
    Вызывает response_generator для указанного агента.
    chat_id → per-chat history (изоляция Iris/Newser/Redmond контекстов).
    status_cb — живой статус из tool-loop; force_tool — принудительное
    делегирование (классификатор решил «research»). Cipher идёт через subprocess.
    meta — необязательный словарь для побочных данных: Cipher кладёт туда
    session_id, чтобы вызывающий привязал к сессии id отправленного сообщения
    (реплай на него потом однозначно указывает, какой разговор продолжать).
    """
    if agent.executor == "cipher_subprocess":
        return await _generate_cipher(user_text, context, chat_id, meta)

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
        response = await asyncio.wait_for(
            asyncio.to_thread(
                dispatcher.response_generator.generate,
                intent,
                user_text,
                role,
                agent,
                chat_id,
                status_cb=status_cb,
                force_tool=force_tool,
            ),
            timeout=GENERATION_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        # Страховка от зависания: даже если поток застрял, хендлер не держит
        # event loop и не копит зависшие задачи (так встал хаб 10.06).
        logger.warning("Generation timed out (%s, %ds)", agent.name, GENERATION_TIMEOUT_SEC)
        return "Что-то долго думаю — видимо, затык с моделью. Попробуй ещё раз через минуту."
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
    meta: Optional[dict] = None,
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
            status_cb=status_cb, force_tool=force_tool, meta=meta,
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
    mode = str(payload.get("mode", "") or "handoff").strip().lower()
    if not task:
        logger.warning("Delegation with empty task from %s — ignored", delegator.name)
        return

    # Target Iris (Redmond передал коуч/еду в её зону) — отвечает владельцу она.
    if str(payload.get("target") or "Newser").strip() == "Iris":
        iris = agent_by_name("Iris")
        if iris is None:
            logger.error("Delegation to Iris failed: not registered")
            return
        logger.info("DELEGATE [%s → Iris]: %s", delegator.name, task[:120])
        await coordinator.respond_as(
            delegator.name, chat_id,
            f"Это к Айрис — @{iris.bot_username}.", delegator.emoji, "plain",
        )
        envelope = (
            f"(передано от {delegator.name} — это запрос владельца в твоей зоне, "
            f"ответь ему сама)\n{task}"
        )
        async with coordinator.typing(iris.name, chat_id):
            response = await _generate_with_status(iris, envelope, context, chat_id)
        if response.startswith(DELEGATION_MARKER):
            # Iris делегировала дальше (напр. факт у Newser) — один уровень вглубь.
            await _run_delegation(iris, response[len(DELEGATION_MARKER):], context, chat_id)
            return
        state: RouterState = get_state(router_states, chat_id)
        state.add("assistant", response, iris.name)
        state.last_agent_name = iris.name
        await coordinator.respond_as(iris.name, chat_id, response, iris.emoji, iris.output_format)
        return

    logger.info("DELEGATE [%s → Newser, %s]: %s", delegator.name, mode, task[:120])

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
    state: RouterState = get_state(router_states, chat_id)
    state.add("assistant", response, newser.name)
    state.last_agent_name = newser.name

    await coordinator.respond_as(newser.name, chat_id, response, newser.emoji, newser.output_format)

    # collect-режим: ресёрч — сырьё, делегатор постит СВОЙ вывод поверх него
    # (совет/вердикт), не пересказывая. +1 LLM-вызов, по делу.
    if mode == "collect":
        verdict_prompt = (
            f"(collect-режим) Ньюсер отресёрчил твой таск: «{task}».\n"
            f"Его ответ уже запощен в чат — НЕ пересказывай его.\n"
            f"РЕЗУЛЬТАТ РИСЁРЧА:\n{response[:2500]}\n\n"
            f"Дай владельцу только твой вывод/совет поверх этих данных, коротко, "
            f"в своём стиле."
        )
        async with coordinator.typing(delegator.name, chat_id):
            verdict = await _generate_with_status(delegator, verdict_prompt, context, chat_id)
        if verdict.startswith(DELEGATION_MARKER):
            logger.warning("Recursive delegation in collect verdict — suppressed")
            return
        state.add("assistant", verdict, delegator.name)
        state.last_agent_name = delegator.name
        await coordinator.respond_as(
            delegator.name, chat_id, verdict, delegator.emoji, delegator.output_format,
        )


async def _generate_cipher(user_text: str, context: ContextTypes.DEFAULT_TYPE,
                           chat_id: int = 0, meta: Optional[dict] = None) -> str:
    """Cipher = Claude Code CLI subprocess на VM (см. core/cipher_wrapper.py)."""
    from core.cipher_wrapper import run_cipher
    reply_to = (meta or {}).get("reply_to_message_id")
    text, session_id = await run_cipher(user_text, chat_id=chat_id,
                                        reply_to_message_id=reply_to)
    if meta is not None:
        meta["cipher_session"] = session_id
    return text


# ---------- Photo handler (скрин графика смен → расписание) ----------

async def redmond_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Фото от Влада: один vision-вызов классифицирует (смены / еда / другое) и
    маршрутизирует. Раньше ВСЕ фото слепо парсились как график смен (фото ужина →
    «смен не увидел»), а на «что видишь?» текстовая модель врала «не вижу картинки».
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

    logger.info("Photo from owner (%d KB) — vision analyze", len(raw) // 1024)
    from logic.vision import analyze_image
    async with coordinator.typing("Redmond", chat_id):
        result = await asyncio.to_thread(
            analyze_image, image_b64, dispatcher.config.groq_api_key,
        )
    logger.info("Vision: type=%s, shifts=%d", result.get("type"), len(result.get("shifts") or []))

    # Архив: фото + полный ответ модели. Без него на вопрос «почему распознал
    # криво» отвечать нечем — 12.08 в логе была ровно одна строка про тип.
    from utils import vision_archive
    from utils.gemini import DEFAULT_MODEL
    archive_id = await asyncio.to_thread(
        vision_archive.save, raw, result, chat_id, DEFAULT_MODEL,
    )

    # Описание фото → в историю чата, чтобы последующие текстовые вопросы
    # («что на фото?») имели контекст и модель не галлюцинировала (кейс «ракумаки»).
    vdesc = result.get("description") or ""
    if vdesc:
        try:
            dispatcher.response_generator.note_to_history(
                chat_id, "(прислал фото)", f"На фото: {vdesc}",
            )
        except Exception:
            logger.warning("note_to_history failed", exc_info=True)

    if result.get("error") and not result.get("description"):
        await coordinator.respond_as(
            "Redmond", chat_id,
            "Не разобрал картинку — попробуй ещё раз или опиши текстом.", "🦞", "plain",
        )
        return

    ptype = result.get("type")
    shifts = result.get("shifts") or []

    # 1. График смен → сохранить + план недели от Iris
    if ptype == "shift_schedule" and shifts:
        from logic.week_schedule import apply_shifts, describe_conflicts, describe_saved_shifts
        shift_items = [
            {**s, "status": "planned", "source": "photo", "confidence": "high"}
            for s in shifts
        ]
        outcome = await asyncio.to_thread(apply_shifts, shift_items)
        n = outcome.saved
        await asyncio.to_thread(
            vision_archive.set_applied, archive_id,
            f"смен сохранено: {n}, спорных: {len(outcome.conflicts)}")
        saved_dates = {c.date for c in outcome.conflicts}
        await coordinator.respond_as(
            "Redmond", chat_id,
            describe_saved_shifts([s for s in shifts if s.get("date") not in saved_dates], n),
            "🦞", "html",
        )
        # Расхождение с тем, что Влад говорил текстом, — не повод молча
        # отклонить: он решит, что график обновился. Спрашиваем сразу.
        if outcome.conflicts:
            await coordinator.respond_as(
                "Redmond", chat_id, describe_conflicts(outcome.conflicts), "🦞", "plain",
            )
        if n > 0:
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
        return

    # 2. Еда → Iris. meal: фиксирует рацион (log_meal с оценками vision) + реакция.
    #    groceries/receipt: пополняем запас детерминированно в коде + тёплое подтверждение.
    if ptype == "food":
        iris = agent_by_name("Iris")
        if iris is None:
            return
        food_kind = result.get("food_kind") or "meal"
        items = result.get("items") or []

        if food_kind in ("groceries", "receipt") and items:
            # Запас обновляем кодом (надёжно), Iris только реагирует — не гоняем её
            # повторно писать инструментом то, что уже записано.
            from logic.coach_storage import pantry_update
            data = await asyncio.to_thread(pantry_update, items, None)
            n = len(data.get("items") or [])
            barcode = result.get("barcode") or ""
            bc_hint = f" Штрихкод: {barcode}." if barcode else ""
            prompt = (
                f"(фото продуктов) Влад пополнил запас: {', '.join(items)}.{bc_hint} "
                f"Я уже добавила это в его запас продуктов (сейчас {n} позиций) — в запас "
                "повторно НЕ добавляй. Можешь вызвать lookup_food (штрихкод или название) для "
                "точной нутриции, затем тёпло и коротко подтвердить + ОДНУ идею что приготовить. "
                "Без лекций."
            )
        else:
            desc = result.get("dish") or result.get("description") or "тарелка с едой"
            kcal_lo, kcal_hi = result.get("kcal_low"), result.get("kcal_high")
            prot = result.get("protein_g")
            est = []
            if kcal_lo or kcal_hi:
                est.append(f"{kcal_lo or kcal_hi}–{kcal_hi or kcal_lo} ккал")
            if prot:
                est.append(f"~{prot} г белка")
            est_line = (" Оценка по фото (честный диапазон, можешь поправить): "
                        + ", ".join(est) + ".") if est else ""
            prompt = (
                f"(фото еды) Влад прислал фото еды: «{desc}».{est_line} "
                "Запиши через log_meal (dish + эти оценки; place определи по STATE: смена "
                "сейчас → 'работа', иначе 'дом'). Дай короткую тёплую реакцию — без лекций о диете."
            )

        async with coordinator.typing(iris.name, chat_id):
            resp = await _generate_with_status(iris, prompt, context, chat_id)
        await asyncio.to_thread(vision_archive.set_applied, archive_id,
                                f"еда: {food_kind}")
        if not resp.startswith(DELEGATION_MARKER):
            await coordinator.respond_as(iris.name, chat_id, resp, iris.emoji, iris.output_format)
        # Sticky → Iris: текстовый follow-up после фото-еды держим за ней.
        st: RouterState = get_state(context.application.bot_data["router_states"], chat_id)
        st.add("assistant", resp, iris.name)
        st.last_agent_name = iris.name
        return

    # 3. Что угодно ещё → Redmond описывает, что видит (без вранья «не вижу картинок»)
    desc = result.get("description") or "Вижу картинку, но не пойму, что именно на ней."
    await asyncio.to_thread(vision_archive.set_applied, archive_id, "ничего не записано")
    await coordinator.respond_as("Redmond", chat_id, desc, "🦞", "plain")


# ---------- Routing core (текст и голос идут одним путём) ----------

async def _route_and_respond(
    text: str,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user,
    handle_foreign: bool = False,
    reply_to_agent: str = "",
) -> None:
    """
    Ядро обработки сообщения владельца: роутинг (явный @-меншин → LLM-классификатор
    → keywords → sticky), research-флаг, генерация, делегирование, отправка.

    handle_foreign: текстовые апдейты чужих @-меншинов обрабатывают их собственные
    Application'ы (False); голосовые видит только Redmond-app, поэтому «Айрис, …»
    голосом он должен исполнить сам от имени Айрис (True).
    """
    bot_data = context.application.bot_data
    coordinator = bot_data["coordinator"]
    dispatcher = bot_data["dispatcher"]
    router_states: dict = bot_data["router_states"]

    state: RouterState = get_state(router_states, chat_id)
    groq_key = dispatcher.config.groq_api_key

    # research-флаг считает классификатор (8b) — решение о делегировании в коде,
    # не на воле 120b-модели.
    explicit = find_by_trigger(text)
    needs_research = False
    if explicit is not None:
        if explicit.name != "Redmond" and not handle_foreign:
            return  # текстовый апдейт — ответит Application того агента
        agent = explicit
        clean_text = explicit.strip_trigger(text) or "(привет)"
        if agent.name == "Redmond":
            needs_research = llm_research_flag(clean_text, state, groq_key)
    else:
        clean_text = text
        agent, needs_research = route(
            text, state, groq_api_key=groq_key, reply_to_agent=reply_to_agent,
        )
        # Роутер решил, что сообщение не к агентам (владелец думает вслух).
        # Молчим — но сообщение всё равно кладём в историю, чтобы следующая
        # реплика имела контекст.
        if agent is None:
            state.add("user", clean_text, state.last_agent_name or "—")
            return

    # Стоп-слово — до LLM и мимо истории: «стоп» должен сработать всегда,
    # даже когда модели лежат или заняты.
    mute_reply = _mute_command_response(clean_text)
    if mute_reply is not None:
        logger.info("Mute command from owner: %s", clean_text[:40])
        await coordinator.respond_as(agent.name, chat_id, mute_reply, agent.emoji, "plain")
        return

    state.add("user", clean_text, agent.name)

    logger.info(
        "IN [%s %s → %s]: %s",
        user.id, user.first_name or user.username, agent.name, clean_text[:120],
    )

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
    state.last_agent_name = agent.name
    await coordinator.respond_as(agent.name, chat_id, response, agent.emoji, agent.output_format)


# ---------- Redmond handler (с router) ----------

async def redmond_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Redmond — единственный с router'ом.

    Логика:
      1) Авторизация
      2) Если в тексте @-меншин ДРУГОГО агента → молчу (тот ответит сам)
      3) Если @redmond или нет триггера → router решает кто отвечает
    """
    if not await _gate(update, context):
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    await _route_and_respond(
        text, context, chat_id, update.effective_user,
        reply_to_agent=reply_target_agent(update),
    )


# ---------- Voice handler (голосовые → Groq Whisper → обычный пайплайн) ----------

async def redmond_voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Голосовое от Влада: Groq Whisper API (whisper-large-v3-turbo, free tier) →
    транскрипт постится в чат (видно, что распознано) → обычный роутинг,
    как будто это текст. Висит только на Redmond-app (как фото), поэтому
    «Айрис, …» голосом исполняется здесь же от её имени (handle_foreign).
    """
    if not await _gate(update, context):
        return
    if not _is_from_owner(update) or not update.message or not update.message.voice:
        return
    chat_id = update.effective_chat.id if update.effective_chat else None
    if chat_id is None:
        return

    coordinator = context.application.bot_data["coordinator"]
    dispatcher = context.application.bot_data["dispatcher"]

    voice = update.message.voice
    if voice.duration and voice.duration > 300:
        await coordinator.respond_as(
            "Redmond", chat_id,
            "Войс длиннее 5 минут — не возьму, накидай короче или текстом.",
            "🦞", "plain",
        )
        return

    tg_file = await voice.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    logger.info("Voice from owner (%d KB, %ss) — transcribing", len(data) // 1024, voice.duration)

    from utils.speech import transcribe_voice
    model = getattr(dispatcher.config, "groq_whisper_model", "whisper-large-v3-turbo")
    async with coordinator.typing("Redmond", chat_id):
        text, err = await asyncio.to_thread(
            transcribe_voice, data, dispatcher.config.groq_api_key, model,
        )

    if err:
        await coordinator.respond_as(
            "Redmond", chat_id, f"Не расслышал (транскрипция упала: {err}).", "🦞", "plain",
        )
        return

    # Транскрипт видим — Влад сразу заметит, если распознало криво
    await coordinator.respond_as("Redmond", chat_id, f"🎙 «{text}»", "🦞", "plain")
    await _route_and_respond(
        text, context, chat_id, update.effective_user, handle_foreign=True,
        reply_to_agent=reply_target_agent(update),
    )


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

    # Стоп-слово с меншеном («Айрис, стоп») — тоже кодом, до LLM.
    mute_reply = _mute_command_response(clean_text)
    if mute_reply is not None:
        logger.info("Mute command from owner (slim): %s", clean_text[:40])
        await coordinator.respond_as(agent.name, chat_id, mute_reply, agent.emoji, "plain")
        return

    # Cipher: реплай на его сообщение — сигнал «продолжаем ЭТОТ разговор».
    meta: dict = {"reply_to_message_id": getattr(
        getattr(update.message, "reply_to_message", None), "message_id", None)}

    async with coordinator.typing(agent.name, chat_id):
        response = await _generate_with_status(agent, clean_text, context, chat_id,
                                               meta=meta)

    state: RouterState = get_state(router_states, chat_id)
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

    sent = await coordinator.respond_as(agent.name, chat_id, response, agent.emoji,
                                        agent.output_format)

    # Ответы Cipher идут мимо response_generator, поэтому в общую память их
    # надо положить руками — иначе остальные агенты (и он сам) не видят, что
    # он вообще отвечал. И привязываем сообщение к сессии для реплаев.
    if agent.executor == "cipher_subprocess":
        session_id = meta.get("cipher_session") or ""
        if session_id and sent:
            from core.cipher_wrapper import bind_message
            await asyncio.to_thread(bind_message, chat_id, session_id, sent[-1])
        try:
            dispatcher = context.application.bot_data["dispatcher"]
            await asyncio.to_thread(
                dispatcher.response_generator.note_to_history,
                chat_id, clean_text, response, agent.name,
            )
        except Exception:
            logger.warning("Cipher: ответ не попал в память", exc_info=True)
