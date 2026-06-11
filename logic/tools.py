"""
Tools для function calling.

Каждый tool — это пара:
  - schema (JSON для Groq tools API)
  - executor (Python функция которая исполняет tool)

Все executors детерминированные, возвращают строку (для tool-response).
Если данных нет — возвращают явное "не доступно" / "пусто", а НЕ выдуманные.

Используется в `_generate_with_groq` через tool-loop.

v2 (Tier 1A optimization):
  • Schema descriptions — на английском (экономия токенов в Groq context).
  • read_dossier → read_dossier_section(section) с фильтром по секциям.
  • Размер schema-блока сокращён ~30% без потерь функциональности.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Маркер делегирования: tool возвращает его вместо обычного результата,
# tool-loop немедленно отдаёт его наверх, handler оркеструет handoff.
# Telegram НЕ доставляет ботам сообщения других ботов — поэтому делегирование
# только in-process, а @-меншен в чате — витрина для владельца.
DELEGATION_MARKER = "\x00DELEGATE\x00"


# ============================================================================
# Tool schemas (формат OpenAI / Groq function calling)
# Descriptions намеренно на английском — токенайзер cl100k_base экономит ~50%
# на английском vs русском. См. Tier 1A оптимизации.
# ============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Current weather and short forecast for a city (wttr.in, free). "
                "ALWAYS call when user asks about weather — never invent temps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name (any language). If user did not specify city, "
                            "use owner_profile.current.city."
                        ),
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Web search via Google grounding (primary) or DuckDuckGo (fallback). "
                "Use for current events, news, prices, facts you don't know. "
                "Returns titles + snippets + URLs. For region-specific topics "
                "(transit, local services, local news) write the query in that "
                "region's language AND pass the region parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Results count 1-5", "default": 3},
                    "region": {
                        "type": "string",
                        "description": (
                            "Region hint like 'de-de', 'ru-ru', 'us-en'. "
                            "Use the topic's region (German transit → 'de-de'). "
                            "Default: 'wt-wt' (worldwide)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_headlines",
            "description": (
                "Fresh headlines from trusted RSS feeds (BBC, The Verge, TechCrunch, "
                "VentureBeat AI, Game Developer, CNBC, CoinDesk, Cointelegraph, BBC Sport). "
                "PREFER this over web_search for news requests / daily digests — cheap, "
                "fast, no search engine. 'all' = grouped digest (world / markets / tech / "
                "sport, 2 headlines each) — use for generic «что по новостям». "
                "Use web_search only for specific topics or follow-up questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "world", "tech", "ai", "gamedev", "finance", "crypto", "sport"],
                        "description": "News category. 'all' = grouped daily digest. Default: all.",
                    },
                    "limit": {"type": "integer", "description": "Max headlines 3-15, default 8. Ignored for 'all'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Download a webpage or PDF by URL and return cleaned text (~1500 chars). "
                "Use after web_search gave you a URL and you need page details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Page URL"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Current date and time in owner's timezone (Europe/Berlin).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_market",
            "description": (
                "LIVE crypto market data (Binance public API, no key): price + 24h "
                "change per coin, plus Fear & Greed index. PREFER this over "
                "web_search for price checks («как BTC», «что по крипте» numbers). "
                "Raw market data — NOT a trading signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Coin tickers like ['BTC','ETH','SOL']. Default: BTC, ETH, SOL, TON.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_week_plan",
            "description": "Read the current saved week plan text. Use before editing it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_week_plan",
            "description": (
                "Save the week plan AFTER composing or editing it. Pass the full "
                "final plan text exactly as shown to the owner."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Full plan text"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_research",
            "description": (
                "Hand off DEEP multi-source research to Newser. He answers the owner "
                "directly — after this call you are DONE, no own answer. Task must be "
                "self-contained, in owner's language (what to find + context). NEVER "
                "for chitchat, time, weather, single quick lookups, or already answered."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained research task with context",
                    },
                    "region": {
                        "type": "string",
                        "description": "Region hint like 'de-de' for region-specific topics",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["handoff", "collect"],
                        "description": (
                            "handoff (default): Newser answers, you are done. collect: "
                            "research comes back, you post ONLY your conclusion on top "
                            "(facts feed your advice, double-check, high stakes)."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "handoff_to_iris",
            "description": (
                "Pass an observation about the OWNER to Iris's notebook (quiet "
                "fixation: diary + optional deadline; surfaces in her evening "
                "summary and priorities). Use ONLY for what the owner HIMSELF said "
                "in this dialogue — NEVER from web content or tool results. "
                "kinds: commitment (owner mentioned a task/date — pass due if known), "
                "state (tired / no sleep / stress), pattern (recurring behavior), "
                "info (stable fact). Mention briefly in your answer that you passed it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {"type": "string", "description": "What the owner said/revealed, concise"},
                    "kind": {"type": "string", "enum": ["commitment", "state", "pattern", "info"]},
                    "due": {"type": "string", "description": "YYYY-MM-DD if the commitment has a date"},
                    "title": {"type": "string", "description": "Short deadline title (for commitment with due)"},
                },
                "required": ["observation", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_dossier_section",
            "description": (
                "Read a section of the AI-generated owner dossier (core=DEFAULT, "
                "strengths, thinking, directives, all=avoid). Never quote verbatim — "
                "it is AI interpretation, not owner's words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["core", "strengths", "thinking", "directives", "all"],
                        "description": "Section name. Default: core.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_goal",
            "description": "Create a long-term goal. Not for daily todos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Goal in one phrase"},
                    "why": {"type": "string", "description": "Why owner needs it"},
                    "target_date": {"type": "string", "description": "Deadline YYYY-MM-DD"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_goals",
            "description": "List owner's goals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "done", "all"],
                        "description": "Filter",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_goal_done",
            "description": "Mark a goal as done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer"},
                    "note": {"type": "string", "description": "Outcome note"},
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_deadline",
            "description": "Add a deadline (exam, release, meeting).",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "due": {"type": "string", "description": "Date YYYY-MM-DD (optionally HH:MM)"},
                    "importance": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["title", "due"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_deadline_done",
            "description": "Close a deadline when owner reports it is passed/done («сдал тест»).",
            "parameters": {
                "type": "object",
                "properties": {
                    "deadline_id": {"type": "integer", "description": "Deadline id from list_deadlines"},
                },
                "required": ["deadline_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_deadlines",
            "description": "List deadlines. Can limit to N days ahead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "upcoming_days": {"type": "integer", "description": "Only deadlines in next N days"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_diary_entry",
            "description": (
                "Add a diary entry. Use to fix: important decisions, insights, "
                "emotional moments. Auto-tag if useful."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags like ['insight', 'decision']",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_diary",
            "description": "Read last diary entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "How many recent entries", "default": 10},
                    "tag": {"type": "string", "description": "Filter by tag"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snooze_pings",
            "description": (
                "Silence proactive pings (meal/training/study reminders). Call when owner "
                "says «отстань», «не сейчас», «занят», «потом». Default 2 hours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "number", "description": "Hours of silence, 0.5-12. Default 2."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_week_schedule",
            "description": (
                "Owner's schedule for the next days: work shifts (bar) + university classes. "
                "Call when planning, or when owner asks «когда у меня смены/пары», or to "
                "check if today/tomorrow is busy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "How many days ahead, default 8"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_profile",
            "description": (
                "Add/remove/change a STABLE fact about owner in his profile (new job, "
                "closed project, principle). NOT for transient states (tired/busy now)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["current", "historical", "principles"],
                    },
                    "field": {
                        "type": "string",
                        "description": "Field inside category (e.g. 'active_projects', 'city')",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["set", "append", "remove"],
                    },
                    "value": {
                        "type": "string",
                        "description": "Value. For objects — JSON-encoded string.",
                    },
                },
                "required": ["category", "field", "action", "value"],
            },
        },
    },
]


# ============================================================================
# Executors
# ============================================================================

def execute_tool(name: str, args: Dict[str, Any], rg=None) -> str:
    """
    Диспетчер tool-вызовов. Возвращает строку для tool-response.
    rg — ResponseGenerator (для доступа к searcher/owner_profile).
    """
    logger.info("Tool: %s(%s)", name, args)

    if name == "get_weather":
        return _tool_get_weather(args.get("city", ""))
    if name == "web_search":
        return _tool_web_search(
            args.get("query", ""), int(args.get("top_k", 3)), rg,
            region=str(args.get("region", "") or "wt-wt"),
        )
    if name == "get_news_headlines":
        return _tool_get_news_headlines(args.get("category", "all"), int(args.get("limit", 8)))
    if name == "web_fetch":
        return _tool_web_fetch(args.get("url", ""))
    if name == "get_current_time":
        return _tool_get_current_time()
    if name == "get_crypto_market":
        return _tool_get_crypto_market(args.get("symbols") or None)
    if name == "delegate_research":
        payload = {
            "task": str(args.get("task", "")).strip(),
            "region": str(args.get("region", "")).strip(),
            "mode": str(args.get("mode", "") or "handoff").strip().lower(),
        }
        return DELEGATION_MARKER + json.dumps(payload, ensure_ascii=False)
    if name == "handoff_to_iris":
        return _tool_handoff_to_iris(args)
    if name == "snooze_pings":
        from logic.coach_storage import set_snooze
        until = set_snooze(float(args.get("hours", 2)))
        return f"Пинги выключены до {until}."
    if name == "get_week_schedule":
        from logic.week_schedule import format_week
        return format_week(int(args.get("days", 8)))
    if name == "get_week_plan":
        from logic.coach_storage import get_week_plan
        plan = get_week_plan()
        if not plan.get("text"):
            return "План недели ещё не составлен."
        return f"План недели (обновлён {plan.get('updated', '?')}):\n\n{plan['text']}"
    if name == "save_week_plan":
        from logic.coach_storage import save_week_plan
        t = str(args.get("text", "")).strip()
        if not t:
            return "Пустой план не сохраняю."
        save_week_plan(t)
        return "План недели сохранён."
    if name == "read_dossier_section":
        return _tool_read_dossier_section(args.get("section", "core"))
    # Backward compat — старое имя tool, на случай если LLM где-то его помнит
    if name == "read_dossier":
        return _tool_read_dossier_section("all")
    if name == "update_profile":
        return _tool_update_profile(args, rg)

    if name == "add_goal":
        return _tool_add_goal(args)
    if name == "list_goals":
        return _tool_list_goals(args)
    if name == "mark_goal_done":
        return _tool_mark_goal_done(args)
    if name == "add_deadline":
        return _tool_add_deadline(args)
    if name == "mark_deadline_done":
        return _tool_mark_deadline_done(args)
    if name == "list_deadlines":
        return _tool_list_deadlines(args)
    if name == "add_diary_entry":
        return _tool_add_diary_entry(args)
    if name == "read_diary":
        return _tool_read_diary(args)

    return f"Неизвестный tool: {name}"


# ============================================================================
# Coach tool executors
# ============================================================================

def _tool_add_goal(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    g = coach_storage.add_goal(
        title=args.get("title", ""),
        why=args.get("why", ""),
        target_date=args.get("target_date"),
    )
    return f"Цель #{g['id']} создана: «{g['title']}» (срок: {g['target_date'] or 'без срока'})"


def _tool_list_goals(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    status = args.get("status")
    if status == "all":
        status = None
    goals = coach_storage.list_goals(status=status)
    if not goals:
        return "Целей нет."
    lines = [f"Целей: {len(goals)}"]
    for g in goals:
        lines.append(f"  #{g['id']} [{g['status']}] {g['title']} (срок: {g.get('target_date') or '—'})")
    return "\n".join(lines)


def _tool_mark_goal_done(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    gid = int(args.get("goal_id", 0))
    g = coach_storage.mark_goal_done(gid, note=args.get("note", ""))
    if not g:
        return f"Цель #{gid} не найдена."
    return f"Цель #{gid} «{g['title']}» закрыта."


def _tool_add_deadline(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    d = coach_storage.add_deadline(
        title=args.get("title", ""),
        due=args.get("due", ""),
        importance=args.get("importance", "medium"),
    )
    return f"Дедлайн #{d['id']} «{d['title']}» → {d['due']} ({d['importance']})"


def _tool_mark_deadline_done(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    did = int(args.get("deadline_id", 0))
    d = coach_storage.mark_deadline_done(did)
    if not d:
        return f"Дедлайн #{did} не найден."
    return f"Дедлайн #{did} «{d['title']}» закрыт."


def _tool_list_deadlines(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    days = args.get("upcoming_days")
    if days is not None:
        days = int(days)
    deadlines = coach_storage.list_deadlines(upcoming_days=days)
    if not deadlines:
        scope = f"в ближайшие {days} дн." if days else ""
        return f"Дедлайнов {scope} нет."
    lines = [f"Дедлайны ({len(deadlines)}):"]
    for d in deadlines:
        lines.append(f"  #{d['id']} {d['due']} — {d['title']} [{d.get('importance', '?')}]")
    return "\n".join(lines)


def _tool_add_diary_entry(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    e = coach_storage.add_diary_entry(
        text=args.get("text", ""),
        tags=args.get("tags") or [],
    )
    return f"Запись #{e['id']} в дневник добавлена ({len(e.get('tags') or [])} тегов)."


def _tool_read_diary(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    entries = coach_storage.read_diary(
        last_n=int(args.get("last_n", 10)),
        tag=args.get("tag"),
    )
    if not entries:
        return "Записей в дневнике нет."
    lines = [f"Последние записи ({len(entries)}):"]
    for e in entries:
        tags = ", ".join(e.get("tags") or [])
        tag_part = f" [{tags}]" if tags else ""
        lines.append(f"  {e['timestamp']}{tag_part}: {e['text'][:200]}")
    return "\n".join(lines)


def _tool_get_weather(city: str) -> str:
    """wttr.in — без ключа, JSON, бесплатно."""
    city = (city or "").strip()
    if not city:
        return "Город не указан."
    try:
        url = f"https://wttr.in/{requests.utils.quote(city)}?format=j1&lang=ru"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "redmond-hub/1.0"})
        if resp.status_code != 200:
            return f"wttr.in вернул {resp.status_code} — данные недоступны."
        d = resp.json()
        cur = d.get("current_condition", [{}])[0]
        area = d.get("nearest_area", [{}])[0]
        loc = area.get("areaName", [{}])[0].get("value", city)
        country = area.get("country", [{}])[0].get("value", "")

        temp_c = cur.get("temp_C", "?")
        feels = cur.get("FeelsLikeC", "?")
        desc_block = cur.get("lang_ru") or cur.get("weatherDesc") or [{}]
        desc = desc_block[0].get("value", "?") if isinstance(desc_block, list) else "?"
        wind = cur.get("windspeedKmph", "?")
        humidity = cur.get("humidity", "?")

        forecast = d.get("weather", [{}])[0]
        min_c = forecast.get("mintempC", "?")
        max_c = forecast.get("maxtempC", "?")

        return (
            f"Погода в {loc}, {country}:\n"
            f"  Сейчас: {temp_c}°C (ощущается {feels}°C), {desc}\n"
            f"  Сегодня: от {min_c}°C до {max_c}°C\n"
            f"  Ветер: {wind} км/ч, влажность {humidity}%"
        )
    except Exception as e:
        logger.exception("get_weather failed")
        return f"Ошибка получения погоды: {e}"


# Source quality ranking (для Newser): известные надёжные источники подняты вверх,
# российские агрегаторы / SEO-шлак — вниз или отрезаны.
# Pattern matching по домену из URL.

_TIER_A_DOMAINS = (
    # Tech / gamedev
    "unity.com", "unrealengine.com", "godotengine.org",
    "github.com", "gitlab.com",
    "stackoverflow.com",
    "arxiv.org",
    "gamefromscratch.com", "polygon.com", "gamasutra.com", "gamedeveloper.com",
    # AI / ML
    "openai.com", "anthropic.com", "deepmind.com", "huggingface.co", "ai.meta.com",
    # Финансы / мир
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com", "economist.com",
    # Tech новости
    "techcrunch.com", "theverge.com", "arstechnica.com", "wired.com",
    "engadget.com", "venturebeat.com", "techradar.com",
    # Документация
    "developer.mozilla.org", "docs.python.org", "docs.microsoft.com",
    # Общие
    "wikipedia.org", "bbc.com", "apnews.com",
)

_TIER_C_DOMAINS = (
    # Russian аггрегаторы / SEO / гос. сми — для финансов / мировых новостей не первичные
    "lenta.ru", "ria.ru", "tass.ru", "rbc.ru", "rt.com", "ren.tv",
    "mail.ru", "news.yandex.ru", "rambler.ru",
    "bcs-express.ru", "finam.ru", "quote.rbc.ru", "banki.ru",
    "vesti.ru", "ng.ru", "ng-life.ru", "pikabu.ru",
)


def _domain_of(url: str) -> str:
    """Вернуть netloc в нижнем регистре, без www."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _domain_tier(url: str) -> int:
    """0 = top tier (whitelist), 1 = normal, 2 = downranked (blacklist)."""
    host = _domain_of(url)
    if not host:
        return 1
    for d in _TIER_A_DOMAINS:
        if host == d or host.endswith("." + d):
            return 0
    for d in _TIER_C_DOMAINS:
        if host == d or host.endswith("." + d):
            return 2
    return 1


def _rank_results(results: list) -> list:
    """
    Сортировка search-результатов по tier (0 наверх, 2 вниз) с сохранением
    исходного порядка внутри tier (stable sort).
    """
    indexed = list(enumerate(results))
    indexed.sort(key=lambda pair: (_domain_tier(pair[1].get("url", "")), pair[0]))
    return [r for _, r in indexed]


def _tool_web_search(query: str, top_k: int, rg, region: str = "wt-wt") -> str:
    if not query or not query.strip():
        return "Пустой запрос."

    # 1. Gemini Google-grounding — настоящий Google: модель сама ищет и отдаёт
    # фактуру с источниками. CSE мёртв (403 на уровне аккаунта), DDG шумный.
    from utils.gemini import grounded_search
    grounded = grounded_search(query)
    if grounded:
        answer, sources = grounded
        lines = ["Источник поиска: google (Gemini grounding)", f"Запрос: {query}", "", answer]
        if sources:
            lines += ["", "Источники:"]
            for i, (title, url) in enumerate(sources[:5], 1):
                lines.append(f"[{i}] {title}")
                lines.append(f"    URL: {url}")
        return "\n".join(lines)
    logger.warning("Gemini grounding недоступен — fallback на CSE/DDG цепочку")

    # 2. Fallback: старая цепочка (CSE → DDG/Brave/Yandex)
    if rg is None or rg.searcher is None:
        return "Поиск недоступен."
    try:
        # Ищем больше чем нужно (5x), потом ранжируем и обрезаем — даёт пространство
        # вытащить надёжные источники наверх если они есть в выдаче.
        raw_k = max(1, min(top_k, 5))
        search_k = max(raw_k * 2, raw_k + 3)
        results, source = rg.searcher.search(query, top_k=search_k, region=region)
    except Exception as e:
        return f"Ошибка поиска: {e}"

    if not results:
        return f"По запросу «{query}» ничего не найдено (источник: {source})."

    ranked = _rank_results(results)[:raw_k]

    lines = [f"Источник поиска: {source}", f"Запрос: {query}", ""]
    for i, r in enumerate(ranked, 1):
        url = r.get("url", "")
        tier = _domain_tier(url)
        tier_mark = ""
        if tier == 0:
            tier_mark = " [trusted]"
        elif tier == 2:
            tier_mark = " [low-quality]"
        lines.append(f"[{i}]{tier_mark} {r.get('title', '')}")
        snip = r.get("snippet", "")
        if snip:
            lines.append(f"    {snip}")
        if url:
            lines.append(f"    URL: {url}")
    return "\n".join(lines)


# RSS-фиды для get_news_headlines. Только доверенные источники (tier A) —
# поэтому stdlib-XML-парсер приемлем (фиды захардкожены, не пользовательский ввод).
_RSS_FEEDS: Dict[str, List[Tuple[str, str]]] = {
    "world": [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ],
    "tech": [
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
    ],
    "ai": [
        ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
        ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ],
    "gamedev": [
        ("Game Developer", "https://www.gamedeveloper.com/rss.xml"),
        ("GameFromScratch", "https://gamefromscratch.com/feed/"),
    ],
    "finance": [
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
    ],
    "sport": [
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
    ],
}

# Секции дайджеста (category="all"): заголовок → категории-источники.
# По 2 заголовка на секцию — компактно; подробности юзер просит по секции.
_DIGEST_SECTIONS: List[Tuple[str, List[str]]] = [
    ("Мир", ["world"]),
    ("Экономика и рынки", ["finance", "crypto"]),
    ("Tech / AI", ["tech"]),
    ("Спорт", ["sport"]),
]
_DIGEST_PER_SECTION = 2


def _parse_feed(xml_bytes: bytes, max_items: int) -> List[Tuple[str, str, str]]:
    """(title, link, date) из RSS 2.0 или Atom. Stdlib ET, bytes на входе
    (str с encoding-декларацией ET не принимает)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items: List[Tuple[str, str, str]] = []
    # RSS 2.0: <item><title><link><pubDate>
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        date = (it.findtext("pubDate") or "").strip()[:16]  # «Tue, 10 Jun 2026»
        if title and link:
            items.append((title, link, date))
        if len(items) >= max_items:
            return items
    if items:
        return items

    # Atom: <entry><title><link href=...><updated>
    ns = "{http://www.w3.org/2005/Atom}"
    for e in root.iter(f"{ns}entry"):
        title = (e.findtext(f"{ns}title") or "").strip()
        link_el = e.find(f"{ns}link")
        link = (link_el.get("href") or "").strip() if link_el is not None else ""
        date = (e.findtext(f"{ns}updated") or e.findtext(f"{ns}published") or "").strip()[:10]
        if title and link:
            items.append((title, link, date))
        if len(items) >= max_items:
            break
    return items


def _fetch_feed_items(name: str, url: str, n: int) -> List[Tuple[str, str, str]]:
    """Скачать и распарсить один фид. Пустой список при любой ошибке."""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; redmond-hub/1.0)"},
        )
        if resp.status_code != 200:
            logger.warning("RSS %s → HTTP %s", name, resp.status_code)
            return []
        return _parse_feed(resp.content, n)
    except Exception as e:
        logger.warning("RSS %s failed: %s", name, e)
        return []


def collect_headlines(category: str, limit: int) -> List[Tuple[str, str, str, str]]:
    """Структурные заголовки категории: (title, url, date, source_name).
    Общий сборщик для tool-формата и кодового дайджеста (logic/digest.py)."""
    feeds = _RSS_FEEDS.get(category) or []
    out: List[Tuple[str, str, str, str]] = []
    for name, url in feeds:
        if len(out) >= limit:
            break
        for title, link, date in _fetch_feed_items(name, url, limit - len(out)):
            out.append((title, link, date, name))
            if len(out) >= limit:
                break
    return out


def _headlines_for_category(category: str, limit: int) -> List[str]:
    """Строки «• title (date)\\n  URL: …» для категории, с пометкой источника."""
    return [
        f"• {title}{f' ({date})' if date else ''} — {name}\n  URL: {link}"
        for title, link, date, name in collect_headlines(category, limit)
    ]


def _digest_all() -> str:
    """Сгруппированный дайджест: по секциям, по 2 заголовка. Один tool-вызов."""
    blocks: List[str] = []
    for section_title, categories in _DIGEST_SECTIONS:
        lines: List[str] = []
        # Для секций из двух категорий (finance+crypto) — по 1 заголовку из каждой.
        per_cat = max(1, _DIGEST_PER_SECTION // len(categories))
        for cat in categories:
            lines.extend(_headlines_for_category(cat, per_cat))
        if lines:
            blocks.append(f"=== {section_title} ===\n" + "\n".join(lines[:_DIGEST_PER_SECTION]))
    if not blocks:
        return "RSS-ленты недоступны. Используй web_search."
    return (
        "Дайджест по секциям (источники доверенные). Формат ответа: жирный заголовок "
        "секции, пункты со ссылками. БЕЗ служебных концовок и предложений услуг.\n\n"
        + "\n\n".join(blocks)
    )


def _tool_get_news_headlines(category: str, limit: int) -> str:
    """Заголовки из доверенных RSS — дешёвая альтернатива web_search для новостей."""
    category = (category or "all").strip().lower()
    if category == "all":
        return _digest_all()

    if category not in _RSS_FEEDS:
        return f"Неизвестная категория: {category}. Доступны: all, {', '.join(_RSS_FEEDS)}."

    limit = max(3, min(limit or 8, 15))
    lines = _headlines_for_category(category, limit)
    if not lines:
        return "RSS-ленты недоступны. Используй web_search."
    return f"Свежие заголовки ({category}), источники доверенные:\n\n" + "\n".join(lines)


_DEFAULT_CRYPTO = ["BTC", "ETH", "SOL", "TON"]
# data-api.binance.vision — официальное public-data зеркало Binance
# (только market data, без ключей, меньше гео-блоков чем api.binance.com).
_BINANCE_URL = "https://data-api.binance.vision/api/v3/ticker/24hr"
_FNG_URL = "https://api.alternative.me/fng/"


def _tool_get_crypto_market(symbols: Optional[List[str]] = None) -> str:
    """Цены + 24h change с Binance public API + Fear & Greed. Без ключей.
    Минимальная интеграция «крипто-данные» (полный rbry — отдельная история)."""
    coins = [s.strip().upper().replace("USDT", "") for s in (symbols or _DEFAULT_CRYPTO) if s.strip()]
    coins = coins[:8] or _DEFAULT_CRYPTO
    pairs = json.dumps([f"{c}USDT" for c in coins], separators=(",", ":"))

    lines = ["Крипторынок (Binance, live):"]
    try:
        resp = requests.get(_BINANCE_URL, params={"symbols": pairs}, timeout=10)
        resp.raise_for_status()
        for t in sorted(resp.json(), key=lambda x: x.get("symbol", "")):
            sym = t["symbol"].replace("USDT", "")
            price = float(t["lastPrice"])
            chg = float(t["priceChangePercent"])
            price_str = f"${price:,.0f}" if price >= 100 else f"${price:,.4g}"
            lines.append(f"• {sym}: {price_str} ({chg:+.1f}% за 24ч)")
    except Exception as e:
        logger.warning("Binance API failed: %s", e)
        lines.append(f"(Binance недоступен: {e} — используй web_search)")

    try:
        fng = requests.get(_FNG_URL, timeout=10).json()["data"][0]
        lines.append(f"Fear & Greed: {fng['value']} ({fng['value_classification']})")
    except Exception:
        logger.debug("FnG API failed", exc_info=True)

    lines.append("(сырые данные рынка, не торговый сигнал)")
    return "\n".join(lines)


def _tool_handoff_to_iris(args: Dict[str, Any]) -> str:
    """Наблюдение Redmond → блокнот Iris. Чистый Python, ноль LLM-вызовов:
    дневник (+дедлайн для датированных обязательств). Видимость — через
    вечерний итог Iris и её TOP PRIORITIES, без лишнего шума в чате."""
    from logic import coach_storage

    obs = str(args.get("observation", "")).strip()
    kind = str(args.get("kind", "info")).strip().lower()
    if not obs:
        return "Пустое наблюдение — не передал."

    tags_by_kind = {
        "commitment": ["обязательство"],
        "state": ["состояние"],
        "pattern": ["паттерн"],
        "info": ["факт"],
    }
    tags = ["от Redmond"] + tags_by_kind.get(kind, ["факт"])
    coach_storage.add_diary_entry(obs, tags=tags)
    out = [f"Передано Iris (дневник, {kind})."]

    due = str(args.get("due", "")).strip()
    if kind == "commitment" and re.match(r"^\d{4}-\d{2}-\d{2}$", due):
        title = str(args.get("title", "")).strip() or obs[:60]
        d = coach_storage.add_deadline(title, due, importance="medium")
        out.append(f"Дедлайн #{d['id']} «{d['title']}» → {due}.")
    return " ".join(out)


def _tool_web_fetch(url: str) -> str:
    """Скачать страницу, вытащить текст. Без тяжёлых либ — regex/bs4."""
    if not url:
        return "Пустой URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; redmond-hub/1.0)",
                "Accept-Language": "ru,en;q=0.7,de;q=0.5",
            },
        )
        if resp.status_code != 200:
            return f"HTTP {resp.status_code} — страница недоступна."

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype or resp.content[:5] == b"%PDF-":
            text = _extract_pdf_text(resp.content)
        else:
            text = _extract_text(resp.text)
        # 1500, не 3000: в сырой текст попадает навигация/меню, а каждый лишний
        # символ пересылается в Groq на каждом следующем хопе tool-loop (TPM 8K).
        if len(text) > 1500:
            text = text[:1500] + "\n…[обрезано]"
        return f"URL: {url}\n\n{text}"
    except Exception as e:
        logger.exception("web_fetch failed")
        return f"Ошибка загрузки: {e}"


def _extract_pdf_text(data: bytes) -> str:
    """Текст из PDF (расписания транспорта, прайсы — частый случай в выдаче).
    Раньше PDF отдавался как бинарный мусор и модель жгла хопы впустую."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "(PDF — извлечение текста недоступно: pypdf не установлен)"
    import io
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages[:5]]
        text = re.sub(r"\s+", " ", " ".join(pages)).strip()
        return text or "(PDF без текстового слоя — вероятно скан)"
    except Exception as e:
        return f"(PDF не разобрался: {e})"


def _extract_text(html: str) -> str:
    """Дешёвая очистка HTML без bs4 — regex + декод."""
    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    entities = {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&laquo;": "«", "&raquo;": "»",
    }
    for e, v in entities.items():
        html = html.replace(e, v)
    text = re.sub(r"\s+", " ", html).strip()
    return text


def _tool_get_current_time() -> str:
    from utils.time import now_local
    return now_local().strftime("%Y-%m-%d %H:%M:%S %Z").strip()


# ============================================================================
# Dossier sectioning
# ============================================================================

# Маппинг ключ-секции → ID раздела в файле owner_dossier.md.
# Заголовки в файле начинаются с "## NN Название".
# Парсим по номеру раздела, а не по русскому названию, чтобы не зависеть
# от формулировок.
_DOSSIER_SECTION_MAP: Dict[str, Tuple[str, ...]] = {
    "core": ("01",),           # Профиль и портфель (имя/языки/проекты)
    "strengths": ("02",),      # Сильные стороны
    "thinking": ("03",),       # Стиль работы / коммуникации
    "directives": ("04",),     # Рекомендованные директивы для AI-агентов
}


def _tool_read_dossier_section(section: str) -> str:
    """
    Возвращает указанный раздел досье.
    section ∈ {core, strengths, thinking, directives, all}.
    """
    section = (section or "core").strip().lower()
    text = _read_dossier_raw()
    if not text:
        return "Досье не найдено."

    if section == "all":
        return text

    ids = _DOSSIER_SECTION_MAP.get(section)
    if not ids:
        return f"Неизвестная секция: {section}. Доступны: core, strengths, thinking, directives, all."

    extracted = _extract_sections(text, ids)
    if not extracted.strip():
        return f"Секция '{section}' не найдена в досье."
    return extracted


def _read_dossier_raw() -> str:
    candidates = [
        Path("data/owner_dossier.md"),
        Path(__file__).parent.parent / "data" / "owner_dossier.md",
    ]
    for p in candidates:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Dossier read failed: %s", e)
                return ""
    return ""


def _extract_sections(text: str, include_ids: Tuple[str, ...]) -> str:
    """Вытащить разделы по их числовому ID (## 01, ## 02, ...)."""
    # Разбиваем по заголовкам уровня 2 ("## NN ...")
    blocks = re.split(r"(?=^## \d{2}\s)", text, flags=re.MULTILINE)
    out = []
    for b in blocks:
        m = re.match(r"^## (\d{2})\s", b)
        if m and m.group(1) in include_ids:
            out.append(b.rstrip())
    return "\n\n".join(out)


def _filter_dossier(text: str, exclude_section_ids: Tuple[str, ...]) -> str:
    """Вернуть досье без указанных разделов."""
    blocks = re.split(r"(?=^## \d{2}\s)", text, flags=re.MULTILINE)
    out = []
    for b in blocks:
        m = re.match(r"^## (\d{2})\s", b)
        if m and m.group(1) in exclude_section_ids:
            continue
        out.append(b.rstrip())
    return "\n\n".join(out)


# ============================================================================
# Profile update
# ============================================================================

def _tool_update_profile(args: Dict[str, Any], rg) -> str:
    """Обновляет config/owner_profile.json и in-memory копию в rg."""
    category = args.get("category", "")
    field = args.get("field", "")
    action = args.get("action", "")
    value = args.get("value", "")

    if not all([category, field, action]):
        return "Не хватает параметров (category/field/action)."

    candidates = [
        Path("config/owner_profile.json"),
        Path(__file__).parent.parent / "config" / "owner_profile.json",
    ]
    profile_path = None
    for p in candidates:
        if p.exists():
            profile_path = p
            break
    if profile_path is None:
        return "owner_profile.json не найден."

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Ошибка чтения профиля: {e}"

    if category not in profile:
        profile[category] = {} if category != "historical" else []

    parsed_value: Any = value
    if value and value.strip().startswith(("{", "[")):
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            pass

    target = profile[category]

    if action == "set":
        if isinstance(target, dict):
            target[field] = parsed_value
        elif isinstance(target, list):
            return f"Cannot set field on list category '{category}'."
    elif action == "append":
        if isinstance(target, list):
            target.append(parsed_value)
        elif isinstance(target, dict):
            if field not in target or not isinstance(target.get(field), list):
                target[field] = []
            target[field].append(parsed_value)
    elif action == "remove":
        if isinstance(target, dict) and field in target:
            if isinstance(target[field], list):
                target[field] = [x for x in target[field] if x != parsed_value and (not isinstance(x, dict) or x.get("name") != parsed_value)]
            else:
                del target[field]
    else:
        return f"Неизвестное действие: {action}"

    if category == "current":
        profile.setdefault("current", {})["_last_updated"] = datetime.now().strftime("%Y-%m-%d")

    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    if rg is not None:
        try:
            rg.owner_profile = profile
        except Exception:
            pass

    return f"OK: {category}.{field} {action} {value[:80]}"
