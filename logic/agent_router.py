"""
Smart router — выбирает агента когда нет явного @-меншина.

Стратегия:
  1. Явный @-префикс (`@coach`, `@redmond`, ...) → точное переключение (логика в agents.find_by_trigger)
  2. Сторонняя LLM-классификация (Groq llama-3.1-8b-instant, дёшево):
     - даём ей описания всех агентов + краткую историю разговора + новое сообщение
     - просим вернуть имя агента одним словом
  3. Если LLM-router не доступен — keyword fallback (по триггерным словам)
  4. Если ничего не сработало — sticky-режим: текущий активный агент (last_agent_per_chat)
  5. По умолчанию — Redmond

Контекст хранится в `RouterState` per-chat — чтобы серия сообщений шла одному агенту,
пока тема не сменилась.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from logic.agents import AGENTS, REDMOND, AgentConfig, agent_by_name, default_agent, find_by_trigger

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ============================================================================
# Router state per chat
# ============================================================================

@dataclass
class RouterState:
    """Состояние роутинга per-chat. Помнит последнего активного агента."""
    last_agent_name: str = "Redmond"
    recent_messages: List[Dict[str, str]] = field(default_factory=list)  # [{role, text, agent}]

    def add(self, role: str, text: str, agent_name: str) -> None:
        self.recent_messages.append({"role": role, "text": text[:500], "agent": agent_name})
        # Храним последние 6 сообщений (3 обмена)
        if len(self.recent_messages) > 6:
            self.recent_messages = self.recent_messages[-6:]


# ============================================================================
# Router LLM
# ============================================================================

_ROUTER_MODEL = "llama-3.1-8b-instant"


def _build_router_prompt() -> str:
    # Cipher не участвует в LLM-routing — он дорогой (Pro лимит).
    # Активируется только через явный @cipher / @c / @шифр.
    routable = [a for a in AGENTS if a.name != "Cipher"]
    agents_desc = "\n".join(
        f"  - {a.name}: {a.description}" for a in routable
    )
    return (
        "Ты — роутер. По сообщению владельца + краткой истории диалога выбери, "
        "какой агент должен ответить.\n\n"
        "АГЕНТЫ:\n"
        f"{agents_desc}\n\n"
        "ПРАВИЛА:\n"
        "  • Цели, дедлайны, дневник, расписание/график, планирование дня, "
        "распорядок, отдых, спорт, тренировки, тесты/учёба (фиксация результата), "
        "усталость, прогресс, дисциплина → Iris\n"
        "  • Актуальные новости, события, цены, релизы, обзоры, факты которые могут "
        "устареть, «что произошло», «найди инфу про…» → Newser\n"
        "  • Погода, текущее время, болтовня, общие вопросы, фоновое объяснение → Redmond\n"
        "\n"
        "STICKY-ПРАВИЛО (КРИТИЧНО):\n"
        "  Если предыдущий ответ был от агента X и текущее сообщение — продолжение "
        "разговора в его зоне → ОСТАВАЙСЯ на X. Не переключайся без явной смены темы.\n"
        "  Пример: Iris спросила про цели → пользователь рассказывает свой график → это ВСЁ ЕЩЁ Iris.\n"
        "  Пример: Newser выдал дайджест → пользователь спросил «а ещё про X» → ВСЁ ЕЩЁ Newser.\n"
        "\n"
        "  Если непонятно — Redmond.\n\n"
        "ОТВЕТЬ ОДНИМ СЛОВОМ — точное имя агента. Без объяснений."
    )


def _llm_route(text: str, history: List[Dict[str, str]], api_key: str) -> Optional[str]:
    """Спрашиваем дешёвую LLM кому отдать. Возвращает имя агента или None."""
    if not api_key or not GROQ_AVAILABLE:
        return None

    history_block = ""
    if history:
        lines = []
        for m in history[-4:]:
            who = "Я" if m["role"] == "user" else m.get("agent", "?")
            lines.append(f"  {who}: {m['text'][:120]}")
        history_block = "ИСТОРИЯ:\n" + "\n".join(lines) + "\n\n"

    user_msg = (
        f"{history_block}"
        f"НОВОЕ СООБЩЕНИЕ ВЛАДЕЛЬЦА: «{text[:400]}»"
    )

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=_ROUTER_MODEL,
            messages=[
                {"role": "system", "content": _build_router_prompt()},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("Router LLM failed: %s", e)
        return None

    # Чистим артефакты — иногда модель добавляет точку, кавычки, и т.п.
    name = raw.strip().strip(".,!?\"'`«»").splitlines()[0].strip()
    if not name:
        return None

    agent = agent_by_name(name)
    if agent is None:
        logger.debug("Router returned unknown agent: %r", raw)
        return None

    return agent.name


# ============================================================================
# Keyword fallback
# ============================================================================

_IRIS_KEYWORDS = (
    "цель", "цел", "дедлайн", "дневник", "запиши", "запис",
    "план на", "планы на", "планирую", "планировать",
    "успева", "просрач", "коуч", "coach", "iris", "айрис",
    "устал", "выгор", "продуктив", "дисциплин",
    "сделал", "выполнил", "закрыл", "прокрастин",
    "график", "расписание", "распорядок", "режим",
    "после работы", "на работе", "до 00", "до полуночи",
    "тренир", "спорт", "зал", "отдых", "выходн",
    "тест", "экзамен", "семестр", "учёба", "учеба", "лекци",
    "вс ", "в сб", "в пн", "во вт", "в ср", "в чт", "в пт", "в вс",
    "сегодня ночью", "ночью прошел", "ночью прошёл",
)

_NEWSER_KEYWORDS = (
    "новост", "что произошл", "что случил", "найди инф", "найди про",
    "поищи", "погугли", "загугли", "что нового",
    "релиз", "обзор", "вышло", "вышел", "вышла",
    "курс ", "цена ", "котиров", "акци", "крипт", "биржев",
    "анонс", "статья", "статьи", "habr", "хабр", "reddit",
    "newser", "ньюсер",
)


def _keyword_route(text: str) -> Optional[str]:
    low = text.lower()
    if any(k in low for k in _NEWSER_KEYWORDS):
        return "Newser"
    if any(k in low for k in _IRIS_KEYWORDS):
        return "Iris"
    return None


# ============================================================================
# Public router
# ============================================================================

def route(text: str, state: RouterState, groq_api_key: str = "") -> AgentConfig:
    """
    Главный роутер. Возвращает выбранного AgentConfig + обновляет state.
    """
    if not text or not text.strip():
        return default_agent()

    # 1. Явный @-меншин (точное переключение, перезаписывает sticky)
    explicit = find_by_trigger(text)
    if explicit is not None:
        state.last_agent_name = explicit.name
        return explicit

    # 2. LLM-классификация
    chosen_name = _llm_route(text, state.recent_messages, groq_api_key)

    # 3. Keyword fallback (если LLM не доступна или вернула мусор)
    if not chosen_name:
        chosen_name = _keyword_route(text)

    # 4. Sticky: если LLM/keywords ничего не дали — оставляем последнего
    if not chosen_name:
        chosen_name = state.last_agent_name or default_agent().name

    agent = agent_by_name(chosen_name) or default_agent()
    state.last_agent_name = agent.name
    logger.info("Router: %r → %s", text[:60], agent.name)
    return agent
