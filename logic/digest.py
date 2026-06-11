"""
Утренний дайджест — собирается КОДОМ из RSS, без LLM tool-loop.

История: LLM-дайджест на max_tokens=1300 обрезался на полуслове (finish_reason
length), жёг самый жирный scheduled-вызов дня и лепил служебную жвачку после
каждой секции. Теперь формат и секции — детерминированный код: не обрезается,
ноль Groq-токенов, режется по границам секций средствами Coordinator._chunk.

Gemini используется ТОЛЬКО для перевода заголовков на русский одним вызовом;
при его недоступности дайджест уходит вовремя с оригинальными заголовками.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple

from logic.tools import _DIGEST_SECTIONS, collect_headlines
from utils.time import now_local

logger = logging.getLogger(__name__)

_SECTION_EMOJI = {
    "Мир": "🌍",
    "Экономика и рынки": "💹",
    "Tech / AI": "🤖",
    "Спорт": "⚽",
}
_PER_SECTION = 2

_WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
_MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня",
               "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def _translate_titles(titles: List[str]) -> Optional[List[str]]:
    """Один Gemini-вызов: заголовки → русский. None при любой проблеме —
    дайджест важнее перевода, уходит с оригиналами."""
    from utils.gemini import generate_text
    if not titles:
        return None
    prompt = (
        "Переведи заголовки новостей на русский: кратко, естественно, без кавычек "
        "вокруг всего заголовка. Верни СТРОГО JSON-массив строк той же длины и "
        "в том же порядке, без пояснений.\n\n"
        + json.dumps(titles, ensure_ascii=False)
    )
    raw = generate_text(prompt, temperature=0.1, max_tokens=1200)
    if not raw:
        return None
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(out, list) or len(out) != len(titles):
        return None
    return [str(t).strip() or orig for t, orig in zip(out, titles)]


def _safe_link_label(title: str) -> str:
    """Квадратные скобки в заголовке ломают [label](url) — заменяем на круглые."""
    return title.replace("[", "(").replace("]", ")")


def build_digest() -> str:
    """Готовый текст дайджеста в Markdown (Coordinator конвертирует в Telegram
    HTML, ссылки кликабельные). '' если RSS целиком недоступен — вызывающий
    решает, что делать (fallback на LLM-путь)."""
    sections: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    for section_title, categories in _DIGEST_SECTIONS:
        items: List[Tuple[str, str, str]] = []  # (headline, url, source)
        per_cat = max(1, _PER_SECTION // len(categories))
        for cat in categories:
            for title, url, _date, source in collect_headlines(cat, per_cat):
                items.append((title, url, source))
        if items:
            sections.append((section_title, items[:_PER_SECTION]))
    if not sections:
        return ""

    # Все заголовки одним вызовом — порядок сохраняется, режем обратно по секциям.
    flat = [t for _, items in sections for (t, _, _) in items]
    translated = _translate_titles(flat)
    if translated:
        it = iter(translated)
        sections = [
            (s, [(next(it), url, src) for (_, url, src) in items])
            for s, items in sections
        ]
    else:
        logger.warning("Digest: перевод недоступен — заголовки в оригинале")

    now = now_local()
    header = (
        f"📰 **Утренний дайджест — "
        f"{_WEEKDAYS[now.weekday()]}, {now.day} {_MONTHS_GEN[now.month - 1]}**"
    )
    blocks = [header]
    for section_title, items in sections:
        emoji = _SECTION_EMOJI.get(section_title, "•")
        lines = [f"{emoji} **{section_title}**"]
        for title, url, source in items:
            lines.append(f"• [{_safe_link_label(title)}]({url}) · {source}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
