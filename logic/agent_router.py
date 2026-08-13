"""
Smart router — выбирает агента когда нет явного @-меншина.

Стратегия (порядок важен — дешёвые и однозначные сигналы идут первыми):
  1. Явный @-префикс или имя («@coach», «айрис,») → точное переключение
  2. Реплай на сообщение Cipher → Cipher. Единственный жёсткий оверрайд:
     остальным реплай уходит подсказкой в промпт, потому что «отвечаю на
     сообщение Newser'а, но прошу в нём Айрис» — обычное дело
  3. Короткое продолжение («да», «продолжи», «даю разрешение») → текущий агент
     без вызова классификатора: темы там нет, платить за LLM не за что
  4. LLM-классификация (Gemini flash-lite, дёшево; Groq 8B — запасной).
     Ей отдаются описания ВСЕХ агентов (включая Cipher — с пометкой о цене),
     история, кто отвечал последним, на чьё сообщение реплай — и право
     ответить «Никто»
  5. Keyword fallback → sticky → Redmond по умолчанию

Разбор 12.08.2026, почему так. Cipher был исключён из списка вариантов, реплаи
не читались, исхода «промолчать» не существовало. В итоге «Почему гемини упал?»
уходило Redmond'у (тот шёл искать курс валют), а «Даю разрешение» — Newser'у,
который отвечал «это к Cipher». Модель роутера была ни при чём: правильного
ответа просто не было в меню.

Контекст хранится в `RouterState` per-chat — чтобы серия сообщений шла одному агенту,
пока тема не сменилась.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from logic.agents import (AGENTS, REDMOND, AgentConfig, agent_by_name,
                          agent_by_username, default_agent, find_by_trigger)

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

# Роутер на Gemini flash-lite, а не на Groq llama-3.1-8b:
#   • 8B плыла на промпте роутера (~1000 токенов правил) — отсюда промахи вроде
#     «почему гемини упал» → Redmond+web_search про курс валют (12.08.2026);
#   • квота отдельная от Groq — стена TPM в чате больше не роняет роутинг заодно.
# Groq остаётся запасным путём: Gemini молчит → пробуем 8B, потом keywords.
_ROUTER_MODEL = "gemini-3.1-flash-lite"
_ROUTER_MODEL_GROQ = "llama-3.1-8b-instant"


NOBODY = "Никто"

# Короткие продолжения: смысла звать классификатор нет, тема не меняется.
# Список намеренно узкий — «а погода?» тоже короткое, но тему МЕНЯЕТ, поэтому
# липнем только на явных подтверждениях/продолжениях.
_CONTINUATION_MARKERS = frozenset({
    "да", "ага", "угу", "ок", "окей", "хорошо", "давай", "давай же", "плюс",
    "нет", "не", "не надо", "не нужно", "стой", "погоди",
    "продолжи", "продолжай", "дальше", "и что", "а дальше", "ещё", "еще",
    "спасибо", "спс", "понял", "поняла", "принял", "ясно",
    "разрешаю", "даю разрешение", "разрешение даю", "можно", "валяй",
})
_MAX_FOLLOWUP_WORDS = 3


def _is_short_followup(text: str) -> bool:
    """Сообщение не несёт новой темы — держим текущего агента без вызова LLM."""
    t = (text or "").strip().lower().strip(".!?,")
    if not t or len(t.split()) > _MAX_FOLLOWUP_WORDS:
        return False
    return t in _CONTINUATION_MARKERS


def reply_target_agent(update: Any) -> str:
    """Имя агента, на чьё сообщение отвечают реплаем ('' если это не реплай).

    Самый точный сигнал о том, кому адресовано сообщение, и до 13.08.2026 он
    просто выбрасывался: роутер вообще не знал про реплаи.

    Работает по duck-typing, без импорта telegram: это логика роутинга, ей
    незачем тянуть за собой Bot API (и незачем требовать его в тестах).
    """
    msg = getattr(update, "message", None)
    replied = getattr(msg, "reply_to_message", None) if msg else None
    author = getattr(replied, "from_user", None) if replied else None
    if not author or not getattr(author, "is_bot", False):
        return ""
    agent = agent_by_username(getattr(author, "username", "") or "")
    return agent.name if agent else ""


def _build_router_prompt() -> str:
    # Cipher участвует в роутинге, но с явным предупреждением о цене: до
    # 13.08.2026 он был исключён совсем, и технические вопросы физически не
    # могли попасть по адресу — «почему гемини упал?» уходило Redmond'у,
    # тот шёл искать курс валют. Правильного ответа просто не было в меню.
    agents_desc = "\n".join(f"  - {a.name}: {a.description}" for a in AGENTS)
    return (
        "Ты — роутер. По сообщению владельца + краткой истории диалога выбери, "
        "какой агент должен ответить.\n\n"
        "АГЕНТЫ:\n"
        f"{agents_desc}\n\n"
        "ПРАВИЛА:\n"
        "  • Цели, дедлайны, дневник, расписание/график, планирование дня, "
        "распорядок, отдых, спорт, тренировки, тесты/учёба (фиксация результата), "
        "еда/готовка/рецепты/продукты/запас/«что приготовить-поесть»/питание/рацион, "
        "усталость, прогресс, дисциплина → Iris\n"
        "  • Новости, события, рынки/крипта, релизы, обзоры, дайджесты, "
        "«что нового», «что произошло» → Newser\n"
        "  • Практические лукапы: как доехать/добраться, расписания транспорта, "
        "адреса, часы работы, цены товаров/услуг, «посмотри/найди/погугли» бытовое "
        "→ Redmond (сам решит: искать или передать Newser)\n"
        "  • Погода, текущее время, болтовня, общие вопросы, фоновое объяснение → Redmond\n"
        "  • Код, логи, ошибки бота, архитектура, «почему упало/не работает», "
        "диагностика системы → Cipher\n"
        "\n"
        "ПРО CIPHER (ВАЖНО):\n"
        "  Он дорогой — работает на платной подписке владельца с общим лимитом.\n"
        "  Бери его ТОЛЬКО для реального технического запроса про сам бот, его код,\n"
        "  логи или сбои. Обычный вопрос про технологии — это Redmond или Newser.\n"
        "  Если разговор УЖЕ идёт с Cipher — продолжение остаётся на нём.\n"
        "  ВАЖНО про имена нашей инфраструктуры: Gemini, Groq, Telegram, qwen, llama,\n"
        "  VM, Oracle, бот, хаб — это то, НА ЧЁМ мы работаем. «Почему гемини упал»,\n"
        "  «groq лежит», «телега отваливается» — это ЖАЛОБА НА СБОЙ СИСТЕМЫ → Cipher,\n"
        "  а НЕ новость про компанию и НЕ вопрос про курс акций.\n"
        "\n"
        "ВАРИАНТ «Никто»:\n"
        "  Если сообщение не адресовано ни одному агенту — владелец думает вслух,\n"
        "  комментирует, реагирует эмоцией, говорит о чём-то своём и ответа не ждёт —\n"
        "  верни «Никто». Лучше промолчать, чем влезть без спроса.\n"
        "  Прямой вопрос или просьба — это ВСЕГДА агент, а не «Никто».\n"
        "\n"
        "STICKY-ПРАВИЛО (КРИТИЧНО):\n"
        "  Если предыдущий ответ был от агента X и текущее сообщение — продолжение "
        "разговора в его зоне → ОСТАВАЙСЯ на X. Не переключайся без явной смены темы.\n"
        "  Пример: Iris спросила про цели → пользователь рассказывает свой график → это ВСЁ ЕЩЁ Iris.\n"
        "  Пример: Newser выдал дайджест → пользователь спросил «а ещё про X» → ВСЁ ЕЩЁ Newser.\n"
        "  Пример: Newser разбирал крипту → «а в акциях мб?» → ВСЁ ЕЩЁ Newser (та же тема — инвестиции).\n"
        "\n"
        "  Если непонятно — Redmond.\n\n"
        "ФЛАГ research:\n"
        "  Если ответ требует СВЕЖЕГО веб-рисёрча по нескольким источникам — новости, "
        "рынки/инвестиции/крипта/акции, «что пишут про», обзоры тем, сравнения, "
        "«как лучше вложить/заработать» — добавь к имени «+research».\n"
        "  Одиночный быстрый факт (погода, время, один адрес/цена, «когда выйдет X») — "
        "НЕ research.\n\n"
        f"ФОРМАТ ОТВЕТА: «Имя», «Имя+research» или «{NOBODY}». "
        "Одна строка, без объяснений."
    )


def _parse_router_reply(raw: str) -> tuple:
    """«Имя» / «Имя+research» → (имя | None, research: bool). Чистая функция."""
    line = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    if not line:
        return None, False
    parts = line.split("+", 1)
    name = parts[0].strip().strip(".,!?\"'`«»").strip()
    research = len(parts) > 1 and "research" in parts[1].lower()
    return (name or None), research


def _ask_gemini(system: str, user_msg: str) -> str:
    """Роутинг через Gemini flash-lite. '' при любой проблеме — уйдём на Groq."""
    try:
        from utils import gemini
        return gemini.generate_text(
            user_msg, system=system, model=_ROUTER_MODEL,
            temperature=0.0, max_tokens=20,
        ).strip()
    except Exception as e:
        logger.debug("Router Gemini failed: %s", e)
        return ""


def _ask_groq(system: str, user_msg: str, api_key: str) -> str:
    """Запасной роутинг через Groq 8B. '' при любой проблеме — уйдём на keywords."""
    if not api_key or not GROQ_AVAILABLE:
        return ""
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=_ROUTER_MODEL_GROQ,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("Router Groq failed: %s", e)
        return ""


def _llm_route(
    text: str,
    history: List[Dict[str, str]],
    api_key: str,
    reply_to_agent: str = "",
    last_agent: str = "",
) -> tuple:
    """Спрашиваем дешёвую LLM кому отдать + нужен ли research.
    Возвращает (имя агента | NOBODY | None, research: bool).

    reply_to_agent/last_agent — контекст, которого у роутера раньше не было:
    он не видел ни на чьё сообщение отвечают, ни кто говорил последним (история
    обрезана до 120 символов на реплику). Реплай здесь ПОДСКАЗКА, а не приказ:
    ответить на сообщение Newser'а и попросить в нём же «айрис, глянь» — обычное
    дело, и уйти это должно Айрис.
    """
    history_block = ""
    if history:
        lines = []
        for m in history[-4:]:
            who = "Я" if m["role"] == "user" else m.get("agent", "?")
            lines.append(f"  {who}: {m['text'][:120]}")
        history_block = "ИСТОРИЯ:\n" + "\n".join(lines) + "\n\n"

    context_lines = []
    if last_agent:
        context_lines.append(f"  Последним отвечал: {last_agent}")
    if reply_to_agent:
        context_lines.append(
            f"  Это ОТВЕТ на сообщение агента {reply_to_agent} "
            "(сильная подсказка, но текст важнее: если в нём назван другой агент "
            "или сменилась тема — выбирай по тексту)"
        )
    context_block = ("КОНТЕКСТ:\n" + "\n".join(context_lines) + "\n\n") if context_lines else ""

    user_msg = (
        f"{context_block}"
        f"{history_block}"
        f"НОВОЕ СООБЩЕНИЕ ВЛАДЕЛЬЦА: «{text[:400]}»"
    )

    system = _build_router_prompt()
    raw = _ask_gemini(system, user_msg) or _ask_groq(system, user_msg, api_key)
    if not raw:
        return None, False

    name, research = _parse_router_reply(raw)
    if not name:
        return None, False

    if name.lower().strip() == NOBODY.lower():
        return NOBODY, False

    agent = agent_by_name(name)
    if agent is None:
        logger.debug("Router returned unknown agent: %r", raw)
        return None, False

    return agent.name, research


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
    "еда", "поел", "поесть", "покушал", "кушать", "кушаю",
    "приготов", "готовлю", "готовить", "рецепт", "продукт", "запас",
    "холодильник", "обед", "ужин", "завтрак", "перекус", "рацион", "питани",
    "тест", "экзамен", "семестр", "учёба", "учеба", "лекци",
    "вс ", "в сб", "в пн", "во вт", "в ср", "в чт", "в пт", "в вс",
    "сегодня ночью", "ночью прошел", "ночью прошёл",
)

# «найди/поищи/погугли/цена» убраны: практические лукапы — зона Redmond
# (вариант A: Newser = только новости/медиа/рынки), он сам делегирует глубокое.
_NEWSER_KEYWORDS = (
    "новост", "что произошл", "что случил", "что нового",
    "релиз", "обзор", "вышло", "вышел", "вышла",
    "курс ", "котиров", "акци", "крипт", "биржев",
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

def route(
    text: str,
    state: RouterState,
    groq_api_key: str = "",
    reply_to_agent: str = "",
) -> tuple:
    """
    Главный роутер. Возвращает (AgentConfig | None, research: bool) + обновляет state.

    None вместо агента — «никто не отвечает»: владелец думает вслух или
    комментирует. Раньше такого исхода не было вообще, поэтому на любое
    сообщение обязательно кто-то влезал.

    research=True означает «нужен глубокий веб-рисёрч»: если агент при этом
    Redmond — handler принудительно делегирует Ньюсеру (решение в коде,
    не на воле 120b-модели).
    """
    if not text or not text.strip():
        return default_agent(), False

    # 1. Явный @-меншин (точное переключение, перезаписывает всё остальное)
    explicit = find_by_trigger(text)
    if explicit is not None:
        state.last_agent_name = explicit.name
        return explicit, False

    # 2. Реплай на Cipher — единственный жёсткий оверрайд. Он и так участвует
    # в LLM-роутинге, но ответ на его конкретное сообщение это однозначный
    # сигнал, на который не надо тратить вызов классификатора.
    if reply_to_agent == "Cipher":
        cipher = agent_by_name("Cipher")
        if cipher is not None:
            state.last_agent_name = cipher.name
            logger.info("Router: реплай Cipher'у → Cipher (без LLM)")
            return cipher, False

    # 3. Короткое продолжение («да», «продолжи», «даю разрешение») — тема не
    # менялась, держим текущего агента и не платим за вызов классификатора.
    if _is_short_followup(text) and state.last_agent_name:
        agent = agent_by_name(state.last_agent_name)
        if agent is not None:
            logger.info("Router: продолжение %r → %s (без LLM)", text[:40], agent.name)
            return agent, False

    # 4. LLM-классификация (агент + research-флаг одним вызовом)
    chosen_name, research = _llm_route(
        text, state.recent_messages, groq_api_key,
        reply_to_agent=reply_to_agent, last_agent=state.last_agent_name,
    )

    # 4a. Роутер решил, что сообщение вообще не к агентам — молчим.
    # last_agent_name НЕ трогаем: разговор не сменился, он просто не начинался.
    if chosen_name == NOBODY:
        logger.info("Router: %r → никто (не адресовано агентам)", text[:60])
        return None, False

    # 5. Keyword fallback (если LLM не доступна или вернула мусор)
    if not chosen_name:
        chosen_name = _keyword_route(text)

    # 6. Sticky: если LLM/keywords ничего не дали — оставляем последнего
    if not chosen_name:
        chosen_name = state.last_agent_name or default_agent().name

    agent = agent_by_name(chosen_name) or default_agent()
    state.last_agent_name = agent.name
    logger.info("Router: %r → %s%s", text[:60], agent.name, "+research" if research else "")
    return agent, research


def llm_research_flag(text: str, state: RouterState, groq_api_key: str = "") -> bool:
    """Research-флаг для явного @-меншина Redmond (route() там скипается).
    Один дешёвый 8b-вызов; при недоступности LLM — False (Redmond сам решит)."""
    _, research = _llm_route(text, state.recent_messages, groq_api_key)
    return research
