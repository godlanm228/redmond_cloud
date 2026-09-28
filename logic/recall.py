"""Recall from long-term memory: full-text and vector search, merged.

Before Sep 28, 2026 memory search was full-text only (FTS5) with a relevance
filter `score > 0.5` that, as the Aug 21 audit found, filtered nothing by
construction (defect K10): any word overlap pulled an old exchange into the
prompt. Full-text search also misses meaning - "у меня болит живот" does not
match an earlier "проблемы с поджелудочной".

Now, when vectors are available:
  * vector similarity decides what is relevant at all (MIN_COSINE);
  * full-text rank still lifts exact-word matches (names, dates, numbers);
  * the two rankings are merged with reciprocal rank fusion.
Without vectors (API down) the old full-text path is used unchanged.

Bot-generated exchanges ("(scheduled ...)") are not indexed: they are the
bot's own output, not something the owner said (defect K4).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils import db

logger = logging.getLogger(__name__)

# Below this cosine a memory is not relevant. Calibrated on the production
# database (see tests/test_recall.py for the values it was set from).
MIN_COSINE = 0.62
_RRF_K = 60
_CANDIDATES = 8
_BOT_CHARS = 300


def _doc(user: str, bot: str) -> str:
    return f"{user} => {(bot or '')[:_BOT_CHARS]}"


def _rows(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    sql = ("SELECT id, user, bot FROM memory WHERE user NOT LIKE '(scheduled%' "
           "ORDER BY id DESC" + (" LIMIT ?" if limit else ""))
    rows = db.query(sql, (limit,) if limit else ())
    return [(str(r["id"]), _doc(r["user"], r["bot"])) for r in rows]


def backfill(batch: int = 50, pause: float = 35.0, sleep=None, max_stalls: int = 5) -> int:
    """Index every memory row that has no fresh vector yet. Safe to repeat.

    Paced for the free tier (about 100 texts a minute): a batch, a pause, the
    next batch. A 429 or an outage stalls a round; after a longer wait the
    work resumes, and after `max_stalls` stalled rounds in a row it stops -
    the next start picks up where this one ended.
    """
    import time
    from utils import embeddings
    sleep = sleep or time.sleep
    total, stalls = 0, 0
    while embeddings.pending("memory", _rows()) and stalls < max_stalls:
        done = embeddings.sync("memory", _rows(), limit=batch)
        total += done
        if done:
            stalls = 0
            sleep(pause)
        else:
            stalls += 1
            sleep(pause * 2)
    return total


def recall(mem: Any, text: str, query_vec: Optional[Sequence[float]],
           top_k: int = 3) -> List[str]:
    """Relevant past exchanges as 'user => bot' lines, best first."""
    fts: List[Dict[str, Any]] = []
    if mem is not None:
        try:
            fts = mem.search(text, top_k=_CANDIDATES) or []
        except Exception as e:  # noqa: BLE001
            from utils import failures
            failures.report("поиск по памяти", e, consequence=failures.DEGRADED)

    if not query_vec:
        # The old path, unchanged: full text only.
        return [_doc(r["user"], r["bot"]) for r in fts if r.get("score", 0) > 0.5][:top_k]

    from utils import embeddings
    try:
        embeddings.sync("memory", _rows(limit=50))  # fresh exchanges since the last call
        hits = embeddings.nearest("memory", query_vec, k=_CANDIDATES, min_score=MIN_COSINE)
    except Exception as e:  # noqa: BLE001
        from utils import failures
        failures.report("векторный поиск по памяти", e, consequence=failures.DEGRADED)
        return [_doc(r["user"], r["bot"]) for r in fts if r.get("score", 0) > 0.5][:top_k]

    relevant = {ref for ref, _ in hits}
    score: Dict[str, float] = {}
    for rank, (ref, _) in enumerate(hits):
        score[ref] = score.get(ref, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, r in enumerate(fts):
        ref = str(r["id"])
        if ref in relevant:  # full text lifts, but cannot admit on its own
            score[ref] += 1.0 / (_RRF_K + rank)

    best = sorted(score, key=lambda ref: -score[ref])[:top_k]
    if not best:
        return []
    marks = ",".join("?" * len(best))
    rows = {str(r["id"]): r for r in db.query(
        f"SELECT id, user, bot FROM memory WHERE id IN ({marks})", [int(b) for b in best])}
    return [_doc(rows[b]["user"], rows[b]["bot"]) for b in best if b in rows]
