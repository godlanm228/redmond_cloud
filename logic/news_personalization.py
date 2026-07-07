"""
Personal relevance ranking for Newser digest.

No fetching here. RSS collection stays in `logic.tools.collect_headlines`; this
module only scores already-collected headlines against the owner's interests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class NewsCandidate:
    title: str
    url: str
    date: str
    source: str
    category: str


@dataclass(frozen=True)
class PersonalPick:
    item: NewsCandidate
    score: int
    reason: str


_CATEGORY_WEIGHT = {
    "crypto": 34,
    "finance": 28,
    "ai": 24,
    "gamedev": 22,
    "tech": 12,
    "world": 4,
    "sport": -12,
}

_SOURCE_WEIGHT = {
    "coindesk": 8,
    "cnbc": 7,
    "techcrunch": 6,
    "venturebeat": 6,
    "game developer": 6,
    "gamefromscratch": 5,
    "bbc business": 4,
    "the verge": 2,
}

_TOPIC_WEIGHTS = [
    (("bitcoin", "btc", "ether", "ethereum", "solana", "crypto", "stablecoin", "coinbase"), 22, "крипта/рынки"),
    (("etf", "inflow", "outflow", "fed", "rate", "inflation", "stocks", "nasdaq", "earnings"), 18, "финрынки"),
    (("openai", "anthropic", "gemini", "llm", "ai model", "agent", "developer", "github"), 16, "AI/dev"),
    (("unity", "unreal", "game engine", "gamedev", "steam", "gpu", "nvidia"), 16, "игры/Unity"),
    (("germany", "essen", "ukraine", "kyiv", "eu", "europe"), 8, "личный контекст"),
    (("launch", "release", "regulation", "security", "lawsuit", "ban", "breakthrough"), 6, "может повлиять на решения"),
]

_NOISE_TERMS = (
    "deal", "sale", "discount", "celebrity", "trailer", "recap", "opinion",
    "podcast", "watch", "tiktok", "instagram", "football", "soccer",
)


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(t in text for t in terms)


def _source_score(source: str) -> int:
    low = source.lower()
    for key, score in _SOURCE_WEIGHT.items():
        if key in low:
            return score
    return 0


def score_candidate(item: NewsCandidate) -> PersonalPick:
    text = f"{item.title} {item.source} {item.category}".lower()
    score = _CATEGORY_WEIGHT.get(item.category, 0) + _source_score(item.source)
    reasons: List[str] = []

    for terms, weight, reason in _TOPIC_WEIGHTS:
        if _contains_any(text, terms):
            score += weight
            reasons.append(reason)

    if _contains_any(text, _NOISE_TERMS):
        score -= 16

    # Specific but not too clickbaity beats generic category filler.
    if 45 <= len(item.title) <= 130:
        score += 3
    if "?" in item.title:
        score -= 4

    # Stable order, concise reason.
    if not reasons:
        reasons.append("похоже на твои темы")
    return PersonalPick(item=item, score=score, reason=", ".join(list(dict.fromkeys(reasons))[:2]))


def rank_personal_news(
    candidates: Iterable[NewsCandidate],
    *,
    exclude_urls: Iterable[str] = (),
    limit: int = 3,
    min_score: int = 42,
) -> List[PersonalPick]:
    excluded = {u for u in exclude_urls if u}
    seen = set()
    scored: List[PersonalPick] = []
    for item in candidates:
        key = item.url or item.title.lower()
        if not item.title or key in excluded or key in seen:
            continue
        seen.add(key)
        pick = score_candidate(item)
        if pick.score >= min_score:
            scored.append(pick)

    scored.sort(key=lambda p: (-p.score, p.item.category, p.item.title))
    out: List[PersonalPick] = []
    used_categories = set()
    for pick in scored:
        # Keep the section varied: do not let crypto fill all slots if AI/dev is
        # also relevant today.
        if pick.item.category in used_categories and len(out) < limit - 1:
            continue
        out.append(pick)
        used_categories.add(pick.item.category)
        if len(out) >= limit:
            break
    return out
