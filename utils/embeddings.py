"""Text embeddings through the Gemini API, cached in SQLite.

Why an API and not a local model. The hub runs on 1 GB of RAM; a multilingual
sentence-embedding model plus its runtime needs several hundred MB, which the
VM does not have (MemoryStore has run in "lite mode, no vector search" since
June). The Gemini embedding endpoint has its own free quota, and our volume is
tiny: ~700 memory and diary records to index once, then one vector per owner
message. When the hub moves to a bigger machine, a local model can replace
`_request()` without touching the callers.

Vectors are a cache, not data: stored with the hash of their text and the
model id, recomputed when either changes. Every failure returns None and the
caller falls back to what it did before (keyword / full-text search).

Search itself is brute force: a few thousand 768-float vectors take
milliseconds, so no index structure is needed.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from array import array
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from utils import db, failures

logger = logging.getLogger(__name__)

_API = "https://generativelanguage.googleapis.com/v1beta"
MODELS = (os.getenv("REDMOND_EMBED_MODEL", "gemini-embedding-2"), "gemini-embedding-001")
DIM = 768
_BATCH = 100
_TIMEOUT = 20.0

QUERY = "query"
DOCUMENT = "document"

# gemini-embedding-2 takes the task as a text prefix; 001 takes task_type.
_PREFIX = {QUERY: "task: search result | query: ", DOCUMENT: "title: none | text: "}
_TASK_TYPE = {QUERY: "RETRIEVAL_QUERY", DOCUMENT: "RETRIEVAL_DOCUMENT"}

_model_in_use: Optional[str] = None


def _key() -> str:
    return os.getenv("REDMOND_GEMINI_API_KEY", "")


def text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _body(model: str, texts: Sequence[str], kind: str) -> dict:
    reqs = []
    for t in texts:
        item = {"model": f"models/{model}", "output_dimensionality": DIM}
        if model.startswith("gemini-embedding-2"):
            item["content"] = {"parts": [{"text": _PREFIX[kind] + t}]}
        else:
            item["content"] = {"parts": [{"text": t}]}
            item["task_type"] = _TASK_TYPE[kind]
        reqs.append(item)
    return {"requests": reqs}


def _request(texts: Sequence[str], kind: str) -> Optional[List[List[float]]]:
    """One batch through the first model that answers. None on failure."""
    global _model_in_use
    key = _key()
    if not key or not texts:
        return None
    models = [_model_in_use] if _model_in_use else list(dict.fromkeys(MODELS))
    for model in models:
        try:
            r = requests.post(f"{_API}/models/{model}:batchEmbedContents",
                              headers={"x-goog-api-key": key},
                              json=_body(model, texts, kind), timeout=_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            failures.report("эмбеддинги", e, consequence=failures.DEGRADED, model=model)
            return None
        failure = failures.check(r)
        if failure is None:
            vectors = [e.get("values") or [] for e in r.json().get("embeddings", [])]
            if len(vectors) != len(texts) or not all(vectors):
                failures.report("эмбеддинги", "ответ неполный", consequence=failures.DEGRADED,
                                model=model)
                return None
            _model_in_use = model
            return [normalize(v) for v in vectors]
        if failure.status == 404 and model != models[-1]:
            continue  # model not available on this key: try the next one
        failures.report("эмбеддинги", failure, consequence=failures.DEGRADED, model=model)
        return None
    return None


def embed(texts: Sequence[str], kind: str = DOCUMENT) -> Optional[List[List[float]]]:
    """Vectors for texts (unit length), batched. None if the API is unavailable."""
    out: List[List[float]] = []
    for i in range(0, len(texts), _BATCH):
        part = _request(list(texts[i:i + _BATCH]), kind)
        if part is None:
            return None
        out.extend(part)
    return out


def embed_query(text: str) -> Optional[List[float]]:
    got = embed([text], QUERY)
    return got[0] if got else None


def model_id() -> str:
    return _model_in_use or MODELS[0]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def normalize(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Both vectors are stored normalised, so this is a dot product."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _pack(v: Sequence[float]) -> bytes:
    return array("f", v).tobytes()


def _unpack(blob: bytes) -> List[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def load(kind: str) -> Dict[str, Tuple[str, List[float]]]:
    """ref -> (text hash, vector) for the current model."""
    rows = db.query("SELECT ref, hash, vec FROM embeddings WHERE kind=? AND model=?",
                    (kind, model_id()))
    return {r["ref"]: (r["hash"], _unpack(r["vec"])) for r in rows}


def sync(kind: str, items: Iterable[Tuple[str, str]], limit: int = 500) -> int:
    """Make sure every (ref, text) has a fresh vector. Returns how many were computed.

    Only missing or changed texts are sent, at most `limit` per call, so the
    first backfill of a large table spreads over a few calls instead of one
    burst. Failure leaves the cache as it was.
    """
    have = load(kind)
    todo = [(ref, text) for ref, text in items
            if text and (ref not in have or have[ref][0] != text_hash(text))][:limit]
    if not todo:
        return 0
    vectors = embed([t for _, t in todo], DOCUMENT)
    if vectors is None:
        return 0
    with db.transaction() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(kind, ref, model, hash, dim, vec) "
            "VALUES(?,?,?,?,?,?)",
            [(kind, ref, model_id(), text_hash(t), len(v), _pack(v))
             for (ref, t), v in zip(todo, vectors)],
        )
    logger.info("Эмбеддинги %s: посчитано %d", kind, len(todo))
    return len(todo)


def nearest(kind: str, query_vec: Sequence[float], k: int = 5,
            min_score: float = 0.0) -> List[Tuple[str, float]]:
    """Top-k refs by cosine similarity to the query, best first."""
    scored = [(ref, cosine(query_vec, vec)) for ref, (_, vec) in load(kind).items()]
    scored = [s for s in scored if s[1] >= min_score]
    scored.sort(key=lambda s: -s[1])
    return scored[:k]
