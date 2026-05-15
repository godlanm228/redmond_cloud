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
                "Web search via Google (primary) or DuckDuckGo (fallback). "
                "Use for current events, news, prices, facts you don't know. "
                "Returns titles + snippets + URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Results count 1-5", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Download a webpage by URL and return cleaned text (~3000 chars). "
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
            "name": "read_dossier_section",
            "description": (
                "Read a section of AI-generated owner dossier. "
                "Sections: 'core' (name, projects, portfolio — DEFAULT), "
                "'strengths', 'thinking' (cognition style), 'directives' (AI recommendations), "
                "'all' (whole file — avoid unless needed). "
                "Do NOT quote verbatim — dossier is AI interpretation, not owner's words."
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
            "name": "update_profile",
            "description": (
                "Update config/owner_profile.json — add/remove/change a fact about owner. "
                "Use when learning a stable new fact (new job, closed project, new principle). "
                "Do NOT use for transient states (tired/busy now)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["current", "historical", "principles"],
                        "description": "current = current state, historical = past, principles = values",
                    },
                    "field": {
                        "type": "string",
                        "description": "Field name inside category (e.g. 'active_projects', 'city')",
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
        return _tool_web_search(args.get("query", ""), int(args.get("top_k", 3)), rg)
    if name == "web_fetch":
        return _tool_web_fetch(args.get("url", ""))
    if name == "get_current_time":
        return _tool_get_current_time()
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


def _tool_web_search(query: str, top_k: int, rg) -> str:
    if not query or not query.strip():
        return "Пустой запрос."
    if rg is None or rg.searcher is None:
        return "Поиск недоступен."
    try:
        # Ищем больше чем нужно (5x), потом ранжируем и обрезаем — даёт пространство
        # вытащить надёжные источники наверх если они есть в выдаче.
        raw_k = max(1, min(top_k, 5))
        search_k = max(raw_k * 2, raw_k + 3)
        results, source = rg.searcher.search(query, top_k=search_k)
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

        text = _extract_text(resp.text)
        if len(text) > 3000:
            text = text[:3000] + "\n…[обрезано]"
        return f"URL: {url}\n\n{text}"
    except Exception as e:
        logger.exception("web_fetch failed")
        return f"Ошибка загрузки: {e}"


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
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Europe/Berlin"))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


# ============================================================================
# Dossier sectioning
# ============================================================================

# Маппинг ключ-секции → ID раздела в файле owner_dossier.md.
# Заголовки в файле начинаются с "## NN Название".
# Парсим по номеру раздела, а не по русскому названию, чтобы не зависеть
# от формулировок.
_DOSSIER_SECTION_MAP: Dict[str, Tuple[str, ...]] = {
    "core": ("01",),           # Профиль и портфель
    "strengths": ("02",),      # Сильные стороны
    "thinking": ("04",),       # Стиль мышления
    "directives": ("06",),     # Рекомендованные директивы
    # 03 (слабые) и 05 (нагрузка) намеренно НЕ выдаются по умолчанию —
    # владелец просил не муссировать тему усталости.
}


def _tool_read_dossier_section(section: str) -> str:
    """
    Возвращает указанный раздел досье.
    section ∈ {core, strengths, thinking, directives, all}.
    Если 'all' — возвращает целый файл (без секций 03/05).
    """
    section = (section or "core").strip().lower()
    text = _read_dossier_raw()
    if not text:
        return "Досье не найдено."

    if section == "all":
        # Удалить только разделы 03 и 05 — остальное всё (включая шапку с дисклеймером)
        return _filter_dossier(text, exclude_section_ids=("03", "05"))

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
