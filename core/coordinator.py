"""
Coordinator — общий объект для 4 параллельных Application'ов.

Держит словарь Bot instances по агентам, чтобы любой handler мог
отправить сообщение от имени любого агента.

Пример:
    coordinator.respond_as("Iris", chat_id=-100..., text="...", emoji="🎯")
    → отправит сообщение в чат через iris_bot.send_message(),
      то есть в Telegram оно будет от Iris с её аватаркой.

Используется когда Redmond-router решил, что отвечать должен другой агент:
он генерирует ответ и через Coordinator шлёт от чужого имени.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
from typing import AsyncIterator, Dict, List, Optional

from telegram import Bot
from telegram.constants import ChatAction, ParseMode

logger = logging.getLogger(__name__)


# Регэкспы для markdown (LLM любит **bold**, ## headers и пр.)
_RX_BOLD = re.compile(r"\*\*([^\*\n]+)\*\*")
_RX_ITALIC = re.compile(r"(?<!\*)\*([^\*\n]+)\*(?!\*)")
_RX_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Header с захватом текста — для конверсии в <b> (html-режим)
_RX_HEADER_LINE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_RX_BACKTICK = re.compile(r"`+([^`\n]+)`+")
# ВАЖНО: используем [ \t]* (не \s*) чтобы не съесть переносы между пунктами.
_RX_BULLET = re.compile(r"^[ \t]*[\*\-•][ \t]+", re.MULTILINE)

# Markdown-ссылка [text](url) — для конверсии в HTML <a>.
_RX_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
# Голые URL — для автоконверсии в кликабельные ссылки.
_RX_BARE_URL = re.compile(r"(?<![\"'>=])(https?://[^\s<>\"]+)")


def _clean_markdown_plain(text: str) -> str:
    """Для plain режима: снимаем все markdown-обозначения."""
    text = _RX_BOLD.sub(r"\1", text)
    text = _RX_ITALIC.sub(r"\1", text)
    text = _RX_HEADER.sub("", text)
    text = _RX_BACKTICK.sub(r"\1", text)
    text = _RX_BULLET.sub("• ", text)
    return text


def _to_html(text: str) -> str:
    """
    Для html режима:
      1. Сохраняем [text](url) как плейсхолдер
      2. HTML-escape всего текста
      3. Конвертируем markdown → Telegram HTML:
         **bold** → <b>, *italic* → <i>, `code` → <code>, ## Header → <b>
      4. Восстанавливаем плейсхолдеры как <a href="url">text</a>
      5. Голые URL → <a href="url">url</a>
    Переносы строк НЕ трогаем — Telegram их отображает как есть.
    """
    # Шаг 1: достаём ссылки и заменяем их временным маркером.
    links: list[tuple[str, str]] = []

    def _stash(match: re.Match) -> str:
        idx = len(links)
        links.append((match.group(1), match.group(2)))
        return f"\x00LINK{idx}\x00"

    text = _RX_MD_LINK.sub(_stash, text)

    # Шаг 2: escape ДО вставки тегов, чтобы наши теги выжили
    text = html.escape(text, quote=False)

    # Шаг 3: markdown → HTML-теги (Telegram рендерит <b>/<i>/<code>)
    text = _RX_HEADER_LINE.sub(r"<b>\1</b>", text)
    text = _RX_BOLD.sub(r"<b>\1</b>", text)
    text = _RX_ITALIC.sub(r"<i>\1</i>", text)
    text = _RX_BACKTICK.sub(r"<code>\1</code>", text)
    text = _RX_BULLET.sub("• ", text)

    # Шаг 4: восстанавливаем ссылки как <a>
    for i, (label, url) in enumerate(links):
        safe_label = html.escape(label, quote=False)
        safe_url = html.escape(url, quote=True)
        text = text.replace(f"\x00LINK{i}\x00", f'<a href="{safe_url}">{safe_label}</a>')

    # Шаг 5: голые URL — превратить в <a>. Только те, что НЕ внутри уже созданного <a>.
    def _bare(match: re.Match) -> str:
        url = match.group(1)
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(url, quote=False)}</a>'

    # Не трогаем то, что уже в <a href="...">URL</a> — но простая регекс это не отличает.
    # Решаем хаком: ищем голый URL только если перед ним нет href=" или >.
    text = _RX_BARE_URL.sub(_bare, text)

    return text


def _chunk(text: str, size: int = 4000):
    """Режем длинные сообщения по границам абзацев/строк/пробелов,
    чтобы не разорвать HTML-тег или слово посередине."""
    while text:
        if len(text) <= size:
            yield text
            return
        cut = text.rfind("\n\n", 0, size)
        if cut < size // 2:
            cut = text.rfind("\n", 0, size)
        if cut < size // 2:
            cut = text.rfind(" ", 0, size)
        if cut <= 0:
            cut = size
        yield text[:cut].rstrip()
        text = text[cut:].lstrip(" \n")


class Coordinator:
    """
    Реестр Bot instances + единая точка отправки сообщений.

    `bots` — словарь agent_name → Bot. Создаётся один раз в main().
    Все Application'ы получают эту ссылку через app.bot_data["coordinator"].
    """

    def __init__(self, bots: Dict[str, Bot]):
        if not bots:
            raise ValueError("Coordinator requires at least one bot")
        self.bots = bots
        logger.info("Coordinator initialized with bots: %s", list(bots.keys()))

    def has(self, agent_name: str) -> bool:
        return agent_name in self.bots

    def bot_for(self, agent_name: str) -> Optional[Bot]:
        return self.bots.get(agent_name)

    @contextlib.asynccontextmanager
    async def typing(self, agent_name: str, chat_id: int) -> AsyncIterator[None]:
        """
        Async context manager: пока внутри — бот в чате показывает «{bot} печатает...».
        Шлёт chat_action каждые 4 секунды (action в TG живёт 5 сек). При выходе —
        останавливает loop. Используется чтобы пользователь видел что бот работает,
        особенно при долгих Groq retries или web_search.

        Usage:
            async with coordinator.typing("Iris", chat_id):
                response = await generate(...)
            await coordinator.respond_as(...)
        """
        bot = self.bots.get(agent_name)
        if bot is None:
            yield
            return

        stop = asyncio.Event()

        async def _loop() -> None:
            while not stop.is_set():
                try:
                    await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                except Exception:
                    # Не валим основную работу из-за typing-indicator
                    logger.debug("typing indicator failed for %s", agent_name, exc_info=True)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=4.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    return

        task = asyncio.create_task(_loop())
        try:
            yield
        finally:
            stop.set()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    async def respond_as(
        self,
        agent_name: str,
        chat_id: int,
        text: str,
        emoji: str = "",
        output_format: str = "plain",
    ) -> List[int]:
        """
        Отправить сообщение от имени указанного агента (его токеном).

        Возвращает id отправленных сообщений (длинный текст режется на куски).
        Нужно, чтобы связать реплай Влада с конкретной сессией Cipher: ответ на
        сообщение — самый точный сигнал «продолжаем этот разговор».

        output_format:
          "plain" — markdown снимается, отправка без parse_mode
          "html"  — [text](url) → <a>, голые URL → <a>, остальное HTML-escape,
                    отправка с parse_mode=HTML (ссылки кликабельные)
        """
        sent: List[int] = []
        bot = self.bots.get(agent_name)
        if bot is None:
            logger.error("Coordinator: no bot registered for agent %r", agent_name)
            return sent
        if not text or not text.strip():
            return sent

        prefix = f"{emoji} " if emoji else ""
        parse_mode = None

        if output_format == "html":
            body = prefix + _to_html(text)
            parse_mode = ParseMode.HTML
        else:
            body = prefix + _clean_markdown_plain(text)

        for chunk in _chunk(body, 4000):
            try:
                msg = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
                if getattr(msg, "message_id", None):
                    sent.append(msg.message_id)
            except Exception:
                logger.exception(
                    "Coordinator: send failed for %s → chat %s (format=%s)",
                    agent_name, chat_id, output_format,
                )
                # Fallback: переотправить как plain без parse_mode
                if parse_mode is not None:
                    try:
                        plain_body = prefix + _clean_markdown_plain(text)
                        for c in _chunk(plain_body, 4000):
                            m = await bot.send_message(chat_id=chat_id, text=c)
                            if getattr(m, "message_id", None):
                                sent.append(m.message_id)
                    except Exception:
                        logger.exception("Coordinator: plain fallback also failed")
                return sent
        return sent
