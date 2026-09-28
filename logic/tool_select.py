"""Which tools to put in front of the model for this message.

Every offered schema costs tokens on every hop, and a model choosing among
many similar tools picks the wrong one more often (on Aug 26, 2026 Iris muted
notifications in reply to "how did we hit the limits?"). So a message gets:

  * the agent's core tools, always (cheap and needed without warning);
  * the few tools closest in meaning to the message (vectors, utils/embeddings);
  * `load_tools` - a compact catalogue of everything else. If the model needs a
    tool that was not offered, it asks for it and gets it on the next step.

The last point is what makes selection safe. The shadow experiment of August
showed that any selector misses sometimes (a keyword selector missed
add_diary_entry on Sep 14); with load_tools a miss costs one extra step
instead of a wrong or missing action. Every load is logged as a miss, which
is the metric for tuning the selector.

Without embeddings (API down) selection falls back to keywords, and
load_tools still covers the misses.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

LOAD_TOOLS = "load_tools"

# Always offered, per agent (model-facing names).
CORE: Dict[str, Set[str]] = {
    "Iris": {"get_current_time", "diary"},
    "Redmond": {"get_current_time", "web_search", "ask_iris"},
    "Newser": {"get_current_time", "web_search", "get_news_headlines"},
}
TOP_K = 3
# Small tool sets are not worth selecting from.
MIN_TOOLS_TO_SELECT = 7


def _name(schema: dict) -> str:
    return schema["function"]["name"]


def tool_text(schema: dict) -> str:
    f = schema["function"]
    return f"{f['name']}: {f.get('description', '')}"


def _summary(schema: dict) -> str:
    first = (schema["function"].get("description") or "").strip().split("\n", 1)[0]
    return first[:110]


def load_tools_schema(deferred: Sequence[dict]) -> dict:
    catalogue = "\n".join(f"- {_name(s)}: {_summary(s)}" for s in deferred)
    return {"type": "function", "function": {
        "name": LOAD_TOOLS,
        "description": ("More tools exist but are not loaded to save tokens. If you need "
                        "any of them, call load_tools with their names; they become "
                        "available on your next step. Do not guess their arguments.\n"
                        + catalogue),
        "parameters": {"type": "object", "properties": {
            "names": {"type": "array", "items": {"type": "string"},
                      "description": "Tool names from the list"}},
            "required": ["names"]},
    }}


def _rank_by_vectors(user_text: str, candidates: Sequence[dict],
                     query_vec: Optional[Sequence[float]]) -> Optional[List[str]]:
    if query_vec is None:
        return None
    from utils import embeddings
    embeddings.sync("tool", [(_name(s), tool_text(s)) for s in candidates])
    have = embeddings.load("tool")
    if not all(_name(s) in have for s in candidates):
        return None
    scored = sorted(((embeddings.cosine(query_vec, have[_name(s)][1]), _name(s))
                     for s in candidates), reverse=True)
    return [n for _, n in scored]


# Russian stems per tool, for the fallback only. Tool descriptions are English,
# so matching message words against them (the August shadow selector) could
# not work for Russian messages at all.
STEMS: Dict[str, Tuple[str, ...]] = {
    "diary": ("запис", "дневник", "удали", "исправ", "проснул", "спал", "самочув"),
    "goals": ("цел",),
    "deadlines": ("дедлайн", "срок", "сдач", "сдал", "экзамен", "тест", "перенес", "клаузур"),
    "food": ("поел", "ел ", "еда", "еду", "обед", "ужин", "завтрак", "приготов", "продукт",
             "запас", "ккал", "белок", "холодильник", "поесть", "перекус"),
    "schedule": ("смен", "работ", "расписан", "пар", "лекц", "план", "недел", "универ"),
    "mute_notifications": ("мут", "тишин", "не пиши", "отстань", "стоп", "можешь писать"),
    "find_photo": ("фото", "скрин", "картин", "снимок"),
    "update_profile": ("профил", "переехал", "теперь я", "меня зовут"),
    "read_dossier_section": ("досье", "обо мне", "помнишь"),
    "delegate_research": ("найди", "поищи", "разузнай", "исследу"),
    "get_weather": ("погод", "дожд", "холодн", "жарк", "градус"),
    "web_search": ("найди", "поищи", "сколько стоит", "где ", "адрес"),
    "web_fetch": ("http", "ссылк", "сайт"),
    "get_news_headlines": ("новост", "что нового", "дайджест"),
    "get_crypto_market": ("биткоин", "btc", "крипт", "эфир", "курс"),
    "ask_iris": ("айрис", "дневник", "еда", "цел", "дедлайн"),
    "handoff_to_iris": ("айрис", "обещал", "решил"),
}


def _rank_by_keywords(user_text: str, candidates: Sequence[dict]) -> List[str]:
    """Only tools whose stems occur in the message. No match - no pick: a
    greeting gets the core tools, not the first three in the list."""
    text = f" {(user_text or '').lower()} "
    scored = [(sum(1 for stem in STEMS.get(_name(s), ()) if stem in text), _name(s))
              for s in candidates]
    return [name for sc, name in sorted(scored, key=lambda x: -x[0]) if sc > 0]


def select(agent_name: str, user_text: str, tools: Sequence[dict],
           query_vec: Optional[Sequence[float]] = None) -> Tuple[List[dict], List[dict], str]:
    """(offered, deferred, how). Offered keeps the original order of `tools`."""
    if len(tools) < MIN_TOOLS_TO_SELECT:
        return list(tools), [], "all"
    core = CORE.get(agent_name, {"get_current_time"})
    candidates = [s for s in tools if _name(s) not in core]
    ranked = _rank_by_vectors(user_text, candidates, query_vec)
    how = "vectors"
    if ranked is None:
        ranked, how = _rank_by_keywords(user_text, candidates), "keywords"
    chosen = core | set(ranked[:TOP_K])
    offered = [s for s in tools if _name(s) in chosen]
    deferred = [s for s in tools if _name(s) not in chosen]
    return offered, deferred, how


def apply_load(requested: Iterable[str], deferred: List[dict],
               offered: List[dict]) -> Tuple[List[str], List[str]]:
    """Move requested tools from deferred to offered. Returns (loaded, unknown)."""
    by_name = {_name(s): s for s in deferred}
    loaded, unknown = [], []
    for name in requested or []:
        name = str(name).strip()
        if name in by_name:
            offered.append(by_name.pop(name))
            loaded.append(name)
        elif not any(_name(s) == name for s in offered):
            unknown.append(name)
    deferred[:] = [s for s in deferred if _name(s) in by_name]
    return loaded, unknown


def load_result(loaded: List[str], unknown: List[str]) -> str:
    parts = []
    if loaded:
        parts.append(f"Loaded: {', '.join(loaded)}. They are available now - call them.")
    if unknown:
        parts.append(f"Unknown or already available: {', '.join(unknown)}.")
    return " ".join(parts) or "Nothing to load."
