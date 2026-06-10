"""
Agent registry — конфиги всех агентов в Redmond hub.

v2 (multi-bot): каждый агент = отдельный Telegram-бот со своим токеном.
Все 4 бота слушают группу через polling (asyncio.gather).
Координация через общий Coordinator — никакого внутреннего event bus
для основного flow не нужно: видимые @-меншины в чате + общая SQLite.

Каждый агент имеет:
  - name / emoji / triggers (как его звать с @)
  - system_prompt_builder — имя билдера в ResponseGenerator
  - allowed_tools — фильтр доступных tools
  - description — для router-LLM (если выбираем без явного @-меншина)
  - bot_username — TG @username бота (без @), нужен Coordinator'у
  - executor — "groq" (стандартный Groq pipeline) | "cipher_subprocess" (Claude Code CLI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentConfig:
    name: str
    emoji: str
    triggers: List[str]            # @-меншины: ["@redmond", "@r"]
    description: str               # для router-LLM
    system_prompt_builder: str     # имя метода в ResponseGenerator
    bot_username: str = ""         # TG @username бота (без @)
    executor: str = "groq"         # "groq" | "cipher_subprocess"
    allowed_tools: Optional[List[str]] = None  # None = все доступные
    name_aliases: List[str] = field(default_factory=list)  # имена без @: ["айрис", "iris"]
    output_format: str = "plain"   # "plain" | "html" — определяет parse_mode в Coordinator
    # Температура LLM — главный рычаг различия «голосов» агентов:
    # Newser сухой и фактологичный, Iris собранная, Redmond живой.
    temperature: float = 0.5
    # Потолок completion-токенов. Русский текст дорогой (~1 токен на 2-3 символа):
    # дайджест Newser на 800 обрезался посреди ссылки.
    max_tokens: int = 800

    def _all_triggers(self) -> List[str]:
        """@-триггеры + имена-алиасы с разделителями (запятая/пробел/двоеточие).
        Имена без знаков препинания не ловятся — иначе «iris это библиотека» поломалось бы."""
        out = list(self.triggers)
        for alias in self.name_aliases:
            a = alias.lower()
            out.extend([f"{a},", f"{a} ", f"{a}:", f"{a}!", f"{a}?"])
        return out

    def matches_trigger(self, text: str) -> bool:
        lower = text.lower().strip()
        # Точное совпадение «iris» (короткое сообщение без знаков) — тоже триггер.
        if lower in {a.lower() for a in self.name_aliases}:
            return True
        return any(lower.startswith(t.lower()) for t in self._all_triggers())

    def strip_trigger(self, text: str) -> str:
        lower = text.lower().strip()
        # Точное «iris»/«айрис» — отвечаем без вопроса
        if lower in {a.lower() for a in self.name_aliases}:
            return ""
        for t in self._all_triggers():
            if lower.startswith(t.lower()):
                return text[len(t):].lstrip(" ,:!?")
        return text


# ============================================================================
# Agents
# ============================================================================

REDMOND = AgentConfig(
    name="Redmond",
    emoji="🦞",
    triggers=["@redmond", "@r", "@redmond_hub_bot"],
    name_aliases=["redmond", "редмонд"],
    description=(
        "Повседневный ассистент: погода, факты, общие вопросы, веб-поиск, "
        "болтовня, время, информация. Дефолтный агент."
    ),
    system_prompt_builder="build_redmond_system_prompt",
    bot_username="redmond_hub_bot",
    executor="groq",
    output_format="html",
    temperature=0.6,
    allowed_tools=None,  # все tools
)

IRIS = AgentConfig(
    name="Iris",
    emoji="🎯",
    triggers=["@iris", "@i", "@айрис", "@coach", "@iris_redberry_bot"],
    name_aliases=["iris", "айрис"],
    description=(
        "Iris — личный коуч и трекер владельца. Цели, дедлайны, прогресс, дневник, "
        "здоровье, финансы, дисциплина. Жёсткая, конкретная, без воды. Не утешает — гонит вперёд. "
        "Использовать когда речь о планировании, отслеживании, рефлексии, мотивации, "
        "фиксации усталости/настроения, критическом разборе решений владельца."
    ),
    system_prompt_builder="build_iris_system_prompt",
    bot_username="iris_redberry_bot",
    executor="groq",
    output_format="html",
    temperature=0.3,
    max_tokens=1100,  # планы дня/недели обрезались на 800 посреди списка
    allowed_tools=[
        "get_current_time", "read_dossier_section", "update_profile",
        "add_goal", "list_goals", "mark_goal_done",
        "add_diary_entry", "read_diary",
        "add_deadline", "list_deadlines", "mark_deadline_done",
        "snooze_pings", "get_week_schedule",
        "get_week_plan", "save_week_plan",
        "delegate_research",  # внешние факты для советов — через Newser, не гадать
    ],
)

NEWSER = AgentConfig(
    name="Newser",
    emoji="📰",
    triggers=["@newser", "@n", "@news", "@newser_redmond_bot"],
    name_aliases=["newser", "ньюсер", "ньювсер"],  # последнее — твоя опечатка из теста
    description=(
        "Newser — searcher и новости. Веб-поиск, парсинг источников, выжимка с цитированием. "
        "Один качественный заход: ищет в нескольких источниках, проверяет, делает summary с указанием URL. "
        "Если не нашёл — честно говорит. Не делегирует. "
        "Использовать когда нужна актуальная информация: новости, события, факты, цены, релизы, обзоры."
    ),
    system_prompt_builder="build_newser_system_prompt",
    bot_username="newser_redmond_bot",
    executor="groq",
    output_format="html",  # ссылки кликабельные через <a href>
    temperature=0.2,
    max_tokens=1300,  # дайджест: 4 секции × 2 пункта на русском не влезали в 800
    allowed_tools=["get_news_headlines", "web_search", "web_fetch",
                   "get_crypto_market", "get_current_time"],
)

CIPHER = AgentConfig(
    name="Cipher",
    emoji="🧠",
    triggers=["@cipher", "@c", "@шифр", "@cipher_redberry_bot"],
    name_aliases=["cipher", "шифр", "клауд", "claude"],
    description=(
        "Cipher — технический исполнитель (Claude Code через Pro подписку). "
        "Код, архитектура, разработка, рефакторинг, написание/правка функций бота, "
        "анализ ошибок в логах, имплементация новых tools. Может оркестрировать (просить Newser найти инфу). "
        "Использовать только когда нужна реальная разработка или серьёзный технический анализ — лимит Pro ~45 msg/5h."
    ),
    system_prompt_builder="build_cipher_system_prompt",  # не используется (subprocess)
    bot_username="cipher_redberry_bot",
    executor="cipher_subprocess",
    output_format="html",
    allowed_tools=None,
)


AGENTS: List[AgentConfig] = [REDMOND, IRIS, NEWSER, CIPHER]


def find_by_trigger(text: str) -> Optional[AgentConfig]:
    """Найти агента по @-префиксу в начале сообщения. None — нет префикса."""
    for a in AGENTS:
        if a.matches_trigger(text):
            return a
    return None


def default_agent() -> AgentConfig:
    return REDMOND


def agent_by_name(name: str) -> Optional[AgentConfig]:
    name_l = name.lower().strip().lstrip("@")
    for a in AGENTS:
        if a.name.lower() == name_l:
            return a
    return None


def agent_by_username(username: str) -> Optional[AgentConfig]:
    """Найти агента по TG bot_username (без @). Нужно Coordinator'у и
    для идентификации сообщений от 'своих' ботов."""
    if not username:
        return None
    u = username.lower().strip().lstrip("@")
    for a in AGENTS:
        if a.bot_username.lower() == u:
            return a
    return None


def all_bot_usernames() -> List[str]:
    """Все @username наших ботов. Используется для авторизации
    inter-bot сообщений (бот→бот делегирование)."""
    return [a.bot_username for a in AGENTS if a.bot_username]
