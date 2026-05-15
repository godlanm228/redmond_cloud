"""Web-поиск по явному интенту 'search'."""

import logging
from typing import Optional

from handlers import register

logger = logging.getLogger(__name__)


@register("search")
def handle_search(ctx) -> Optional[str]:
    query = ctx.intent.slots.get("query", ctx.user_text)
    searcher = ctx.rg.searcher if ctx.rg else None

    if not searcher:
        return f"Поиск недоступен. Запрос: «{query}»."

    try:
        results, source = searcher.search(query, top_k=3)
    except Exception:
        logger.exception("Search error")
        return "Произошла ошибка при поиске. Попробуйте позже."

    if not results:
        return f"По запросу «{query}» ничего не найдено."

    prefix = ""
    if source == "duckduckgo":
        prefix = "⚠ Источник: DuckDuckGo (резервный, точность ниже). "

    lines = [prefix + "Результаты:"] if prefix else ["Результаты:"]
    for i, r in enumerate(results[:3], 1):
        lines.append(f"{i}. {r.get('title', '')}")
        lines.append(f"   {r.get('snippet', '')}")
        lines.append(f"   {r.get('url', '')}")
    return "\n".join(lines)
