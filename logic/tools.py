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
                        "type": ["string", "null"],
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
                        "type": ["string", "null"],
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
                    "due": {"type": ["string", "null"], "description": "YYYY-MM-DD if the commitment has a date (omit or null otherwise)"},
                    "title": {"type": ["string", "null"], "description": "Short deadline title for a dated commitment (omit or null otherwise)"},
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
                    "why": {"type": ["string", "null"], "description": "Why owner needs it"},
                    "target_date": {"type": ["string", "null"], "description": "Deadline YYYY-MM-DD"},
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
                    "note": {"type": ["string", "null"], "description": "Outcome note"},
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
            "description": (
                "Close a deadline as DONE when owner reports it is passed/completed "
                "(«сдал тест»). If he asks to REMOVE a wrong or irrelevant deadline "
                "(«удали/убери») use delete_deadline instead — done pollutes stats."
            ),
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
            "name": "delete_deadline",
            "description": (
                "Remove a deadline ENTIRELY («удали/убери дедлайн» — wrong entry or "
                "no longer relevant). Unlike mark_deadline_done (owner DID it / it "
                "passed), delete leaves no trace in history or stats."
            ),
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
                "Log a REAL event/state/decision of the owner (meal, training, sleep, "
                "study, mood, commitment) with a tag. NEVER log meta — that he messaged "
                "you, acknowledgements, your own actions, or content-free reactions. "
                "If nothing real happened, do not call this."
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
                    "tag": {"type": ["string", "null"], "description": "Filter by tag"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_diary_entry",
            "description": (
                "Delete one or more diary entries by id (e.g. a wrong meal). Find ids "
                "via read_diary (shown as #N). To FIX an entry: delete it, then log the "
                "correct one. NEVER claim a deletion without calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_ids": {"type": "array", "items": {"type": "integer"}, "description": "Diary entry ids to delete"},
                },
                "required": ["entry_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_meal",
            "description": (
                "Log a meal the owner ate, with HONEST estimates. Use on «поел…», a food "
                "photo, or when he describes what he ate. Estimates are rough — give tight "
                "ranges (~±15%), never fake precision; use null when you truly can't tell."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dish": {"type": "string", "description": "What he ate, short Russian («курица с рисом и овощами»)"},
                    "kcal_low": {"type": ["integer", "null"], "description": "Lower calorie estimate"},
                    "kcal_high": {"type": ["integer", "null"], "description": "Upper calorie estimate (keep the range tight)"},
                    "protein_g": {"type": ["integer", "null"], "description": "Approx protein grams"},
                    "place": {"type": ["string", "null"], "description": "Where eaten: дом / работа / вне"},
                },
                "required": ["dish"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pantry",
            "description": (
                "Read the owner's current food stock (pantry). Call BEFORE suggesting what to "
                "cook. If empty or flagged stale, ask him what he's got, then update_pantry."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_pantry",
            "description": (
                "Incrementally update food stock: add what he bought, remove what he cooked / "
                "ran out of. Items are plain product names (Russian). Not a full snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "add": {"type": "array", "items": {"type": "string"}, "description": "Products to add"},
                    "remove": {"type": "array", "items": {"type": "string"}, "description": "Products to remove"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mute_notifications",
            "description": (
                "Silence proactive bot messages. Default scope='pings' mutes ONLY "
                "day-ticker check-ins (meal/training/study/checkin) — morning digest, "
                "deadline reminders and evening summary KEEP arriving. scope='all' = "
                "total silence, use ONLY when owner explicitly wants everything off "
                "(«вообще ничего не присылай», «полная тишина»). Replies to his OWN "
                "messages always work. «отстань/не сейчас/занят» → hours=2; «не пиши "
                "сегодня/стоп» → mode='today'; «вообще не пиши/отключись» → "
                "mode='forever' + scope='all'; «пиши/можешь писать» → mode='off'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": ["string", "null"],
                        "description": "today (till end of day) / forever / off (resume). Omit if using hours.",
                    },
                    "hours": {"type": "number", "description": "Hours of silence, 0.5-168."},
                    "scope": {
                        "type": ["string", "null"],
                        "description": "'pings' (default: only ticker check-ins) or 'all' (total silence incl. digests).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "postpone_deadline",
            "description": (
                "Move an EXISTING deadline to a new date («перенесём на неделю», «сдвинь "
                "на пятницу»). ALWAYS use this for postponements — add_deadline would "
                "create a duplicate and both would nag the owner."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "deadline_id": {"type": "integer", "description": "Id from list_deadlines"},
                    "new_due": {"type": "string", "description": "New date YYYY-MM-DD"},
                },
                "required": ["deadline_id", "new_due"],
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
            "name": "save_work_shift",
            "description": (
                "Save or correct one work shift in the owner's schedule from a text "
                "message. Use when owner says he has/has had a shift with explicit "
                "hours, e.g. «сегодня смена 17-23», «да, с 17 до 23». If date is "
                "omitted, use today's date in Europe/Berlin. Do not use when owner "
                "only says «на работе» without hours — log diary instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": ["string", "null"],
                        "description": "Shift date YYYY-MM-DD. Omit/null for today.",
                    },
                    "start": {"type": "string", "description": "Start time HH:MM, e.g. 17:00"},
                    "end": {"type": "string", "description": "End time HH:MM, e.g. 23:00"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_photo",
            "description": (
                "Find a photo the owner sent earlier, by meaning or by the label he gave "
                "it. Use for «кинь тот график, что я скидывал», «что было на том скрине», "
                "«найди фото еды за прошлую неделю». Returns what was recognised and "
                "what was recorded from it — say it in words; the file itself is not "
                "attached automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for: «график смен», «оливье», «заработок»",
                    },
                    "limit": {"type": "integer", "description": "How many, 1-5", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_shift_conflict",
            "description": (
                "Answer the pending question about a schedule photo that contradicts what "
                "the owner said earlier. Use when he replies «бери с фото» / «график "
                "правильный» (take='photo') or «оставь как есть» / «я же сказал» "
                "(take='mine'). Set always=true when he says «всегда», «и дальше так», "
                "«больше не спрашивай» — then the same choice is applied automatically "
                "next time and the question stops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "take": {
                        "type": "string",
                        "enum": ["photo", "mine"],
                        "description": "Whose version wins: the photo or his own correction",
                    },
                    "always": {
                        "type": ["boolean", "null"],
                        "description": "Remember this choice for future photos",
                    },
                },
                "required": ["take"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_work_shift_status",
            "description": (
                "Confirm or cancel an existing work shift, or mark it uncertain. Use for "
                "messages like «смена в силе», «сегодня не иду», «возможно позже». If "
                "date is omitted, use today's date in Europe/Berlin. For changed hours "
                "use save_work_shift instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": ["string", "null"],
                        "description": "Shift date YYYY-MM-DD. Omit/null for today.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["confirmed", "cancelled", "uncertain"],
                    },
                    "note": {"type": ["string", "null"], "description": "Short reason/context"},
                },
                "required": ["status"],
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
    {
        "type": "function",
        "function": {
            "name": "lookup_food",
            "description": (
                "Look up EXACT nutrition of a PACKAGED/store product in OpenFoodFacts "
                "(free DB) by barcode or name — per 100g. Use BEFORE logging numbers for "
                "packaged food, instead of guessing. Returns 'not found' if absent — then "
                "estimate honestly. Home-cooked-from-scratch food → just estimate, skip this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "barcode": {"type": ["string", "null"], "description": "EAN/barcode digits if known"},
                    "name": {"type": ["string", "null"], "description": "Product name (fallback if no barcode)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_iris",
            "description": (
                "Hand the owner's request to Iris (the coach) and let HER answer him. "
                "Use for anything in her zone — food/eating/cooking/groceries/pantry, "
                "diary, goals, deadlines, training, schedule & study tracking, mood. "
                "Do NOT handle these yourself. After calling this you are DONE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "The owner's request, self-contained, in his language"},
                },
                "required": ["request"],
            },
        },
    },
]


# Что в выдаче инструмента обязано пережить сжатие на следующих хопах.
#
# Знание принадлежит ИНСТРУМЕНТУ, а не компрессору. Раньше компрессор
# догадывался по виду текста: сначала умел строки `URL:` (писался под
# web_search), потом его доучили `#id` ради дневника. Любой инструмент с
# другим форматом ссылки ломался бы молча — ровно так 17.08.2026 модель,
# потеряв номера записей, назвала порядковый и снесла непричастную.
#
# "ids"  — выдача содержит ссылки вида `#N`, по ним модель адресует записи
# "urls" — выдача содержит строки `URL:`, они нужны для цитирования
# отсутствие ключа — сжимать без сохранений
OUTPUT_ESSENTIALS: Dict[str, str] = {
    # читающие — отдают списки записей, по которым модель потом адресует
    "read_diary": "ids",
    "list_goals": "ids",
    "list_deadlines": "ids",
    "find_photo": "ids",
    # пишущие — отдают ссылку на только что созданную или изменённую запись
    "add_goal": "ids",
    "mark_goal_done": "ids",
    "add_deadline": "ids",
    "mark_deadline_done": "ids",
    "delete_deadline": "ids",
    "postpone_deadline": "ids",
    "add_diary_entry": "ids",
    "delete_diary_entry": "ids",
    "log_meal": "ids",
    # поисковые — ссылки на источники нужны для цитирования
    "web_search": "urls",
    "web_fetch": "urls",
    "get_news_headlines": "urls",
}


# ============================================================================
# Executors
# ============================================================================

def execute_tool(name: str, args: Dict[str, Any], rg=None) -> str:
    """Диспетчер tool-вызовов с записью ИСХОДА, а не только вызова.

    Раньше в лог уходила одна строка «Tool: name(args)», и что из этого
    вышло — нигде. Поэтому инцидент 17.08.2026 (удалили не ту запись
    дневника) пришлось раскапывать запросами к базе: в логе было видно
    `delete_diary_entry({'entry_ids': [3]})` и ничего о том, что удалилось.

    Исключение инструмента здесь же и остаётся. Без этого оно всплывало
    в `_generate_with_providers` и записывалось как `Provider groq failed` —
    баг инструмента выглядел отказом провайдера (инвариант И1: сбой
    отчитывается там, где случился, и по своему последствию).
    """
    logger.info("Tool: %s(%s)", name, args)
    try:
        result = _dispatch_tool(name, args, rg)
    except Exception as e:  # noqa: BLE001
        from utils import failures
        failures.report(f"инструмент {name}", e,
                        consequence=failures.DEGRADED, args=args)
        return f"Инструмент {name} не отработал: {e}"
    logger.info("Tool result: %s → %s", name, _short_result(result))
    return result


def _short_result(result: Any, limit: int = 160) -> str:
    """Исход в лог — коротко, но узнаваемо: по нему должно быть видно,
    что именно произошло с данными владельца."""
    text = " ".join(str(result or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _dispatch_tool(name: str, args: Dict[str, Any], rg=None) -> str:
    """Собственно разбор имени. Возвращает строку для tool-response.
    rg — ResponseGenerator (для доступа к searcher/owner_profile)."""
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
            "target": "Newser",
        }
        return DELEGATION_MARKER + json.dumps(payload, ensure_ascii=False)
    if name == "ask_iris":
        payload = {"task": str(args.get("request", "")).strip(), "target": "Iris"}
        return DELEGATION_MARKER + json.dumps(payload, ensure_ascii=False)
    if name == "handoff_to_iris":
        return _tool_handoff_to_iris(args)
    if name in ("mute_notifications", "snooze_pings"):  # snooze_pings — старое имя (compat)
        from logic.coach_storage import muted_now, set_mute, unmute
        mode = str(args.get("mode") or "").strip().lower()
        if mode == "off":
            if not muted_now():
                return "Тишина и не была включена — всё работает."
            unmute()
            return "Тишина снята — проактивные сообщения снова включены."
        hours = float(args.get("hours") or 0)
        if not mode and not hours:
            hours = 2.0  # голое «отстань» = как старый snooze
        scope = str(args.get("scope") or "pings").strip().lower()
        desc = set_mute(mode=mode or "hours", hours=hours, scope=scope)
        if scope == "all":
            return (f"Полная тишина {desc} — включая утренний дайджест, "
                    f"напоминания о дедлайнах и вечерний итог.")
        return (f"Пинги-напоминалки выключены {desc}. Утренний дайджест и "
                f"вечерний итог остаются — если нужно убрать и их, скажи "
                f"«вообще ничего не присылай».")
    if name == "get_week_schedule":
        from logic.week_schedule import format_week
        return format_week(int(args.get("days", 8)))
    if name == "save_work_shift":
        return _tool_save_work_shift(args)
    if name == "set_work_shift_status":
        return _tool_set_work_shift_status(args)
    if name == "resolve_shift_conflict":
        return _tool_resolve_shift_conflict(args)
    if name == "find_photo":
        return _tool_find_photo(args)
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
    if name == "delete_deadline":
        return _tool_delete_deadline(args)
    if name == "postpone_deadline":
        return _tool_postpone_deadline(args)
    if name == "list_deadlines":
        return _tool_list_deadlines(args)
    if name == "add_diary_entry":
        return _tool_add_diary_entry(args)
    if name == "read_diary":
        return _tool_read_diary(args)
    if name == "delete_diary_entry":
        return _tool_delete_diary_entry(args)
    if name == "log_meal":
        return _tool_log_meal(args)
    if name == "get_pantry":
        return _tool_get_pantry()
    if name == "update_pantry":
        return _tool_update_pantry(args)
    if name == "lookup_food":
        return _tool_lookup_food(args)

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
    # Оставшиеся pending — в результат: иначе Iris закрывает один из дублей
    # и рапортует «всё чисто», а второй продолжает долбить по утрам.
    rest = [x for x in coach_storage.list_deadlines() if x.get("status") == "pending"]
    msg = f"Дедлайн #{did} «{d['title']}» закрыт."
    if rest:
        listing = "; ".join(f"#{x['id']} «{x['title']}» → {x['due']}" for x in rest)
        msg += (f" ВНИМАНИЕ, ещё открыты: {listing}. Если что-то из этого — тот же "
                f"дедлайн (дубль/устаревший перенос), закрой и его.")
    return msg


def _tool_delete_deadline(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    did = int(args.get("deadline_id", 0))
    d = coach_storage.delete_deadline(did)
    if not d:
        return f"Дедлайн #{did} не найден."
    return f"Дедлайн #{did} «{d['title']}» удалён насовсем — в историю и статистику не попадёт."


def _tool_postpone_deadline(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    did = int(args.get("deadline_id", 0))
    new_due = str(args.get("new_due", "")).strip()
    try:
        datetime.strptime(new_due, "%Y-%m-%d")
    except ValueError:
        return f"Дата «{new_due}» не в формате YYYY-MM-DD — перенос не сделан."
    d = coach_storage.update_deadline(did, due=new_due)
    if not d:
        return f"Дедлайн #{did} не найден."
    return f"Дедлайн #{did} «{d['title']}» перенесён → {new_due}."


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
    if not e or not e.get("id"):
        return "Пустая/служебная заметка — в дневник не пишу."
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
        lines.append(f"  #{e.get('id', '?')} {e['timestamp']}{tag_part}: {e['text'][:200]}")
    return "\n".join(lines)


def _norm_hm(raw: Any) -> Optional[str]:
    """Normalize 17 / 17:00 / 17.30 values to HH:MM."""
    s = str(raw or "").strip().lower()
    if not s:
        return None
    s = s.replace(".", ":").replace(",", ":")
    m = re.match(r"^(\d{1,2})(?::(\d{1,2}))?$", s)
    if not m:
        return None
    h = int(m.group(1))
    minute = int(m.group(2) or 0)
    if not (0 <= h <= 23 and 0 <= minute <= 59):
        return None
    return f"{h:02d}:{minute:02d}"


def _tool_save_work_shift(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    from logic.week_schedule import save_shifts
    from utils.time import now_local

    shift_date = str(args.get("date") or "").strip() or now_local().strftime("%Y-%m-%d")
    try:
        datetime.strptime(shift_date, "%Y-%m-%d")
    except ValueError:
        return f"Дата смены «{shift_date}» не в формате YYYY-MM-DD — смену не сохранила."

    start = _norm_hm(args.get("start"))
    end = _norm_hm(args.get("end"))
    if not start or not end:
        return "Нужны часы смены в формате HH:MM — смену не сохранила."

    n = save_shifts([{
        "date": shift_date,
        "start": start,
        "end": end,
        "status": "confirmed",
        "source": "text",
        "confidence": "high",
    }])
    if not n:
        return "Смену не сохранила — не хватило даты или времени."

    # Дневниковый факт нужен вечернему итогу, shifts.json — будущим пингам.
    coach_storage.add_diary_entry(
        f"Смена {shift_date} с {start} до {end}",
        tags=["работа"],
        data={"date": shift_date, "start": start, "end": end},
    )
    return f"Смена сохранена в расписание: {shift_date} {start}–{end}."


def _tool_set_work_shift_status(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    from logic.week_schedule import get_shift_record, save_shifts
    from utils.time import now_local

    shift_date = str(args.get("date") or "").strip() or now_local().strftime("%Y-%m-%d")
    try:
        dt = datetime.strptime(shift_date, "%Y-%m-%d").date()
    except ValueError:
        return f"Дата смены «{shift_date}» не в формате YYYY-MM-DD — статус не изменила."

    status = str(args.get("status") or "").strip().lower()
    if status not in {"confirmed", "cancelled", "uncertain"}:
        return "Статус смены должен быть confirmed/cancelled/uncertain."

    current = get_shift_record(dt) or {}
    item: Dict[str, Any] = {
        "date": shift_date,
        "status": status,
        "source": "text",
        "confidence": "high" if status in {"confirmed", "cancelled"} else "medium",
        "note": str(args.get("note") or "").strip(),
    }
    if current.get("start"):
        item["start"] = current["start"]
    if current.get("end"):
        item["end"] = current["end"]

    n = save_shifts([item])
    if not n:
        return "Не нашла смену с часами на эту дату. Если часы известны — сохрани через save_work_shift."

    if status == "confirmed":
        text = f"Смена {shift_date} подтверждена"
        reply = f"Смена {shift_date} подтверждена."
    elif status == "cancelled":
        text = f"Смена {shift_date} отменена"
        reply = f"Смена {shift_date} помечена как отменённая."
    else:
        text = f"Смена {shift_date} под вопросом"
        reply = f"Смена {shift_date} помечена как под вопросом."
    if item["note"]:
        text += f": {item['note']}"

    coach_storage.add_diary_entry(text, tags=["работа"], data={"date": shift_date, "status": status})
    return reply


def _tool_find_photo(args: Dict[str, Any]) -> str:
    """Поиск по архиву разборов фото.

    До 15.08.2026 присланные фото нигде не хранились: «кинь тот график» было
    невыполнимо, а «почему распознал криво» — непроверяемо.
    """
    from utils import vision_archive

    query = str(args.get("query") or "").strip()
    try:
        limit = max(1, min(int(args.get("limit") or 3), 5))
    except (TypeError, ValueError):
        limit = 3
    found = vision_archive.search(query, limit=limit)
    if not found:
        return f"По «{query}» в архиве фото ничего нет."

    lines = []
    for r in found:
        when = str(r["ts"])[:16].replace("T", " ")
        label = f" «{r['label']}»" if r["label"] else ""
        applied = f" · записано: {r['applied']}" if r["applied"] else ""
        gone = "" if r["exists"] else " · файл уже удалён по сроку, остался разбор"
        lines.append(f"#{r['id']} {when}{label} — {r['description'][:110]}{applied}{gone}")
    return "Нашла в архиве:\n" + "\n".join(lines)


def _tool_resolve_shift_conflict(args: Dict[str, Any]) -> str:
    """Ответ Влада на вопрос «что верно — фото или твои слова».

    Без этого приоритет источников работал бы как молчаливый отказ: фото
    отклонено, Влад об этом не знает и считает, что график обновился.
    """
    from logic.week_schedule import pending_conflicts, resolve_pending_conflicts

    take = str(args.get("take") or "").strip().lower()
    if take not in ("photo", "mine"):
        return "Неясно, чью версию оставить: 'photo' или 'mine'."
    pending = pending_conflicts()
    if not pending:
        return "Открытых расхождений по сменам нет — спрашивать не о чем."

    always = bool(args.get("always"))
    applied = resolve_pending_conflicts(take, remember=always)
    tail = " Дальше буду решать так же, без вопросов." if always else ""
    if take == "photo":
        return (f"Принял график: обновил смен — {applied}. Твоё решение записано "
                f"как ручное, следующее фото его не перетрёт.{tail}")
    return f"Оставил как ты говорил, график не трогаю (спорных дней: {len(pending)}).{tail}"


def _tool_delete_diary_entry(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    raw = args.get("entry_ids")
    if raw is None:
        raw = args.get("entry_id")
    if isinstance(raw, bool):
        ids = []
    elif isinstance(raw, (int, float)):
        ids = [int(raw)]
    elif isinstance(raw, str):
        ids = [int(p) for p in raw.replace("#", " ").replace(",", " ").split() if p.isdigit()]
    elif isinstance(raw, list):
        ids = [int(x) for x in raw if str(x).strip().lstrip("#").isdigit()]
    else:
        ids = []
    if not ids:
        return "Не указано какие записи удалять (нужны id из read_diary)."
    removed = coach_storage.delete_diary_entries(ids)
    if not removed:
        return (f"Записи {ids} не найдены — нечего удалять. "
                f"Прочитай дневник заново (read_diary) и возьми id оттуда.")
    # Отчитываемся ТЕКСТОМ удалённого, а не номером: номер владельцу ничего
    # не говорит, и подмену («снесли не ту») по нему не заметить. 17.08.2026
    # ответ «Удалила записи: #3» был правдой о неверном действии.
    lines = [f"#{e['id']}: {(e.get('text') or '').strip()[:90]}" for e in removed]
    missing = sorted(set(ids) - {e["id"] for e in removed})
    tail = f" Не найдены и не тронуты: {missing}." if missing else ""
    return "Удалила — " + "; ".join(lines) + "." + tail


_PLACE_ALIASES = {"home": "дом", "work": "работа", "out": "вне"}


def _tool_log_meal(args: Dict[str, Any]) -> str:
    """Запись приёма пищи в рацион: дневник тег [питание] + структурный data
    (dish/kcal/protein/place) для аналитики Этапа 3. Оценки честные — диапазоны."""
    from logic import coach_storage
    dish = str(args.get("dish", "")).strip()
    if not dish:
        return "Что именно поел — не указано, в рацион не пишу."

    lo, hi = args.get("kcal_low"), args.get("kcal_high")
    kcal: Optional[List[int]] = None
    if lo is not None or hi is not None:
        lo = int(lo) if lo is not None else int(hi)
        hi = int(hi) if hi is not None else int(lo)
        kcal = [min(lo, hi), max(lo, hi)]

    protein = args.get("protein_g")
    protein = int(protein) if protein is not None else None

    place = str(args.get("place") or "").strip().lower()
    place = _PLACE_ALIASES.get(place, place)

    data: Dict[str, Any] = {"dish": dish}
    if kcal:
        data["kcal"] = kcal
    if protein is not None:
        data["protein_g"] = protein
    if place:
        data["place"] = place

    summary = dish + (f" ({place})" if place else "")
    e = coach_storage.add_diary_entry(f"Поел: {summary}", tags=["питание"], data=data)
    if not e or not e.get("id"):
        return "Дубль/пусто — в рацион не пишу."

    est = []
    if kcal:
        est.append(f"~{kcal[0]}–{kcal[1]} ккал")
    if protein is not None:
        est.append(f"~{protein} г белка")
    tail = (" · " + ", ".join(est)) if est else ""
    return f"Записал в рацион: {summary}{tail}."


def _tool_get_pantry() -> str:
    from logic import coach_storage
    data = coach_storage.get_pantry()
    items = data.get("items") or []
    if not items:
        return "Запас пуст — спроси у Влада, что есть из продуктов."
    head = f"Запас (обновлён {data.get('updated', '?')}"
    age = coach_storage.pantry_age_days()
    if age is not None and age >= 5:
        head += f", {age} дн. назад — уточни, ещё актуально"
    return head + "):\n" + ", ".join(items)


def _tool_lookup_food(args: Dict[str, Any]) -> str:
    from utils import openfoodfacts
    barcode = str(args.get("barcode") or "").strip()
    name = str(args.get("name") or "").strip()
    if not barcode and not name:
        return "Нужен штрихкод или название продукта."
    res = openfoodfacts.lookup(barcode=barcode, name=name)
    if not res:
        return f"«{name or barcode}» в OpenFoodFacts не найдено — дай честную оценку."
    title = res["name"] or name or "продукт"
    if res.get("brands"):
        title += f" ({res['brands']})"
    facts = []
    if res.get("kcal_100g") is not None:
        facts.append(f"~{res['kcal_100g']} ккал/100г")
    if res.get("protein_100g") is not None:
        facts.append(f"~{res['protein_100g']} г белка/100г")
    if res.get("quantity"):
        facts.append(f"упаковка {res['quantity']}")
    return f"OpenFoodFacts: {title}" + (" — " + ", ".join(facts) if facts else " (нутриция не указана)")


def _tool_update_pantry(args: Dict[str, Any]) -> str:
    from logic import coach_storage
    add = args.get("add") or []
    remove = args.get("remove") or []
    if isinstance(add, str):
        add = [add]
    if isinstance(remove, str):
        remove = [remove]
    if not add and not remove:
        return "Нечего обновлять в запасе."
    data = coach_storage.pantry_update(add=add, remove=remove)
    items = data.get("items") or []
    return f"Запас обновлён ({len(items)} поз.): " + (", ".join(items) if items else "пусто")


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

# Российские гос/пропаганда источники — для украинца-владельца ОТРЕЗАЕМ полностью
# (не просто вниз), особенно по войне/политике. Матчим по подстроке домена.
_RU_BLOCK_DOMAINS = (
    "ria.ru", "tass.ru", "rt.com", "lenta.ru", "gazeta.ru", "vesti.ru",
    "rbc.ru", "ren.tv", "1tv.ru", "vz.ru", "kp.ru", "iz.ru", "tsargrad",
    "rg.ru", "aif.ru", "mk.ru", "regnum", "rian.ru", "ria.com", "smotrim.ru",
    "gov.ru", "mil.ru",
)


def _is_ru_blocked(text: str) -> bool:
    """Источник из RU-пропаганда/гос blocklist? Проверяем и по URL, и по
    title (у Gemini-grounding реальный URL спрятан за redirect, домен виден
    только в title)."""
    s = (text or "").lower()
    return any(d in s for d in _RU_BLOCK_DOMAINS)


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
    results = [r for r in results if not _is_ru_blocked(r.get("url", ""))]
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
        # Отрезаем RU-пропаганда источники (украинец-владелец). URL у grounding —
        # redirect, поэтому матчим и по title, и по url.
        sources = [(t, u) for (t, u) in sources if not _is_ru_blocked(t) and not _is_ru_blocked(u)]
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
            pass  # не JSON — оставляем как строку (намеренный fallback, не ошибка)

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
            logger.debug("owner_profile in-memory sync failed", exc_info=True)

    return f"OK: {category}.{field} {action} {value[:80]}"
