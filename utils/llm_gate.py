"""One place that knows how much each model can still take.

Why. Every caller used to learn about a limit by hitting it. On Sep 29, 2026
two measurements showed how far the code was from the real limits:
  • Gemini free tier allows 20 requests a day per model per project
    (429 body: GenerateRequestsPerDayPerProjectPerModel-FreeTier, 20); the code
    and the comments assumed 1500. Iris, the router, web search, vision and the
    digest translation all drew on the same 20 requests of gemini-3.6-flash.
  • Groq allows 8000 tokens a minute per model with no prompt caching; two
    3.7K-token prompts in a row empty the bucket, the third gets 429. An Iris
    answer is ~2 hops of ~4.5K tokens, so the second hop hit the wall and the
    answer fell through the fallback chain.

What the gate does:
  • learns the limits from the providers themselves: Groq reports its limits
    and what is left in x-ratelimit-* headers on every reply; Gemini names the
    exhausted quota in the 429 body (quotaId …PerDay/…PerMinute, quotaValue,
    retryDelay);
  • blocks a model until its limit resets, so no call is spent to learn it
    again (a daily Gemini block lasts until midnight Pacific time);
  • answers "which of these models can take ~N tokens now, and if none, how
    long until one can" — the caller waits the exact time instead of failing;
  • keeps part of each daily quota for the owner's messages: background work
    (digest translation, memory indexing) may not spend it;
  • survives restarts (kv 'llm_gate'): a daily block must not be forgotten
    by a restart at 06:00.

It never calls a provider itself and never raises: a broken ledger must not
stop an answer.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

OWNER = "owner"            # a reply to the owner: may use everything, may wait a little
BACKGROUND = "background"  # jobs nobody waits for: must leave the reserve untouched

# Share of a daily request quota kept for the owner's messages.
_RESERVE_SHARE = 0.4
_RESERVE_MIN = 2

# Pauses after failures that don't say how long to wait.
_OVERLOAD_PAUSE = 60.0       # 503 "high demand": try another model for a minute
_OVERLOAD_PAUSE_MAX = 600.0
_SERVER_PAUSE = 30.0         # other 5xx
_GONE_PAUSE = 24 * 3600.0    # 404: model withdrawn, only a config change fixes it

_KV_KEY = "llm_gate"
_SAVE_EVERY = 5.0            # seconds between persisting plain counters

_now = time.time             # replaced in tests

try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover
    _PACIFIC = None


def provider_of(model: str) -> str:
    return "gemini" if (model or "").startswith("gemini") else "groq"


@dataclass
class ModelState:
    rpd: Optional[int] = None          # requests per day
    rpm: Optional[int] = None          # requests per minute
    tpm: Optional[int] = None          # tokens per minute
    day: str = ""                      # provider day the counter belongs to
    used_today: int = 0                # our own count (Gemini sends no headers)
    remaining_requests: Optional[int] = None  # Groq header, as of `seen`
    remaining_tokens: Optional[int] = None    # Groq header, as of `seen`
    seen: float = 0.0
    blocked_until: float = 0.0
    reason: str = ""
    overloads: int = 0                 # 503s in a row
    minute: List[float] = field(default_factory=list)  # request times, last 60 s


# Which models may do which job, in order of preference. The only place model
# ids live besides config: before Sep 29, 2026 the router, vision and search
# kept theirs in code, the weekly model check never saw them, and two of them
# (llama-3.1-8b-instant, llama-4-scout) had been withdrawn by Groq unnoticed.
DEFAULT_POOLS: Dict[str, List[str]] = {
    # agent answers with tools, per provider (a tool loop can't switch provider)
    "chat_groq": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
    "chat_gemini": ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.8-flash"],
    # one call without tools: compose an answer from what was gathered
    "compose": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemini-2.5-flash",
                "gemini-3.6-flash"],
    # tiny classification on every owner message: cheap models with big quotas
    "router": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "gemini-3.1-flash-lite"],
    "vision": ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.8-flash"],
    "search": ["gemini-3.6-flash", "gemini-2.5-flash"],
    # work nobody waits for (digest translation): never the owner's reserve
    "background": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "gemini-2.5-flash"],
}

_lock = threading.RLock()
_states: Dict[str, ModelState] = {}
_seeds: Dict[str, Dict[str, int]] = {}
_pools: Dict[str, List[str]] = {k: list(v) for k, v in DEFAULT_POOLS.items()}
_loaded = False
_configured = False
_last_save = 0.0


# ---------------------------------------------------------------------------
# Setup and persistence
# ---------------------------------------------------------------------------

def configure(known_limits: Optional[Mapping[str, Mapping[str, int]]] = None,
              pools: Optional[Mapping[str, Sequence[str]]] = None) -> None:
    """Limits known in advance (config 'model_limits'), e.g. {"gemini-3.6-flash":
    {"rpd": 20}} — what the providers report later overrides them; and pools
    (config 'model_pools') over the defaults."""
    global _configured
    with _lock:
        _configured = True
        _seeds.clear()
        for model, lim in (known_limits or {}).items():
            _seeds[model] = {k: int(v) for k, v in dict(lim).items() if k in ("rpd", "rpm", "tpm")}
        for model, st in _states.items():
            _apply_seed(model, st)
        _pools.clear()
        _pools.update({k: list(v) for k, v in DEFAULT_POOLS.items()})
        for task, models in (pools or {}).items():
            _pools[task] = [m for m in dict.fromkeys(models) if m]


def configure_from(config: Any) -> None:
    """configure() from AppConfig: the legacy primary/fallback fields lead the
    chat pools, so they keep meaning what they meant."""
    pools = {k: list(v) for k, v in (getattr(config, "model_pools", None) or {}).items()}
    if "chat_groq" not in pools:
        lead = [getattr(config, "groq_model", ""), getattr(config, "groq_fallback_model", "")]
        pools["chat_groq"] = list(dict.fromkeys(m for m in lead + DEFAULT_POOLS["chat_groq"] if m))
    if "chat_gemini" not in pools:
        lead = [getattr(config, "gemini_model", "")] + list(
            getattr(config, "gemini_fallback_models", []) or [])
        pools["chat_gemini"] = list(dict.fromkeys(m for m in lead + DEFAULT_POOLS["chat_gemini"] if m))
    configure(getattr(config, "model_limits", None) or {}, pools)


def is_configured() -> bool:
    return _configured


def pool(task: str) -> List[str]:
    with _lock:
        return list(_pools.get(task) or DEFAULT_POOLS.get(task, []))


def pools() -> Dict[str, List[str]]:
    with _lock:
        return {k: list(v) for k, v in _pools.items()}


def replace_model(old: str, new: str) -> None:
    """A withdrawn model is replaced by its successor everywhere (model watch)."""
    with _lock:
        for task, models in _pools.items():
            _pools[task] = list(dict.fromkeys(new if m == old else m for m in models))


def reset() -> None:
    """Forget everything (tests)."""
    global _loaded, _last_save, _configured
    with _lock:
        _configured = False
        _states.clear()
        _seeds.clear()
        _pools.clear()
        _pools.update({k: list(v) for k, v in DEFAULT_POOLS.items()})
        _loaded = False
        _last_save = 0.0


def _apply_seed(model: str, st: ModelState) -> None:
    for k, v in _seeds.get(model, {}).items():
        if getattr(st, k) is None:
            setattr(st, k, v)


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        from utils import db
        raw = db.kv_get(_KV_KEY, {}) or {}
        for model, data in raw.items():
            st = ModelState(**{k: v for k, v in data.items() if k in ModelState.__dataclass_fields__})
            _states[model] = st
    except Exception:  # noqa: BLE001 — a lost ledger means relearning, not failing
        logger.warning("llm_gate: состояние не прочитано — начинаю с чистого", exc_info=True)


def _save(force: bool = False) -> None:
    global _last_save
    now = _now()
    if not force and now - _last_save < _SAVE_EVERY:
        return
    _last_save = now
    try:
        from utils import db
        db.kv_set(_KV_KEY, {m: asdict(s) for m, s in _states.items()})
    except Exception:  # noqa: BLE001
        logger.debug("llm_gate: состояние не сохранено", exc_info=True)


def _state(model: str) -> ModelState:
    _load()
    st = _states.get(model)
    if st is None:
        st = _states[model] = ModelState()
        _apply_seed(model, st)
    day = _day_key(model)
    if st.day != day:
        st.day, st.used_today = day, 0
    return st


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def _day_key(model: str) -> str:
    now = datetime.fromtimestamp(_now(), _PACIFIC if provider_of(model) == "gemini" and _PACIFIC
                                 else None)
    return now.strftime("%Y-%m-%d")


def next_pacific_midnight(now: Optional[float] = None) -> float:
    """Gemini daily quotas reset at midnight Pacific time (09:00 in Berlin)."""
    now = _now() if now is None else now
    if _PACIFIC is None:
        return now + 24 * 3600
    local = datetime.fromtimestamp(now, _PACIFIC)
    tomorrow = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.timestamp()


_DURATION_RX = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?"
                          r"(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(value: Any) -> Optional[float]:
    """'7.66s', '2m59.56s', '3h21m', '29s', '120ms', '16' → seconds."""
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = _DURATION_RX.match(s)
    if not m or not any(m.groups()):
        return None
    h, mnt, sec, ms = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mnt * 60 + sec + ms / 1000


_TRY_AGAIN_RX = re.compile(r"try again in\s+([\dhms.]+)", re.I)


# ---------------------------------------------------------------------------
# Reports from the callers
# ---------------------------------------------------------------------------

def report(model: str, status: Optional[int], headers: Optional[Mapping[str, str]] = None,
           body: Any = None) -> None:
    """What a call to `model` returned. status None = no HTTP answer (network)."""
    try:
        with _lock:
            _report(model, status, {k.lower(): v for k, v in (headers or {}).items()}, body)
    except Exception:  # noqa: BLE001
        logger.debug("llm_gate.report failed", exc_info=True)


def _report(model: str, status: Optional[int], headers: Dict[str, str], body: Any) -> None:
    st = _state(model)
    now = _now()
    _read_headers(st, headers, now)
    changed = False
    if status == 200:
        st.used_today += 1
        st.minute = [t for t in st.minute if now - t < 60] + [now]
        st.overloads = 0
        if st.blocked_until and st.blocked_until <= now:
            st.blocked_until, st.reason = 0.0, ""
    elif status == 429:
        _learn_429(model, st, headers, body, now)
        changed = True
    elif status == 503:
        st.overloads += 1
        pause = min(_OVERLOAD_PAUSE * st.overloads, _OVERLOAD_PAUSE_MAX)
        _block(model, st, now + pause, "перегружена у провайдера (503)")
        changed = True
    elif status == 404:
        _block(model, st, now + _GONE_PAUSE, "снята провайдером (404)")
        changed = True
    elif status is not None and status >= 500:
        _block(model, st, now + _SERVER_PAUSE, f"ошибка сервера ({status})")
        changed = True
    _save(force=changed)


def _read_headers(st: ModelState, h: Dict[str, str], now: float) -> None:
    """Groq: x-ratelimit-limit-requests is the daily request limit,
    x-ratelimit-limit-tokens the per-minute token limit."""
    def num(key: str) -> Optional[int]:
        try:
            return int(float(h[key]))
        except (KeyError, TypeError, ValueError):
            return None

    rpd, tpm = num("x-ratelimit-limit-requests"), num("x-ratelimit-limit-tokens")
    left_r, left_t = num("x-ratelimit-remaining-requests"), num("x-ratelimit-remaining-tokens")
    if rpd:
        st.rpd = rpd
    if tpm:
        st.tpm = tpm
    if left_r is not None or left_t is not None:
        st.seen = now
        st.remaining_requests = left_r if left_r is not None else st.remaining_requests
        st.remaining_tokens = left_t if left_t is not None else st.remaining_tokens
    if left_r == 0:
        wait = parse_duration(h.get("x-ratelimit-reset-requests")) or 3600.0
        st.blocked_until = max(st.blocked_until, now + wait)
        st.reason = "дневной лимит запросов исчерпан"


def _learn_429(model: str, st: ModelState, headers: Dict[str, str], body: Any, now: float) -> None:
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False) if body else ""
    data = body if isinstance(body, dict) else _json(text)

    if provider_of(model) == "gemini":
        quota, value, retry = _gemini_quota(data)
        if "perday" in quota.lower():
            if value:
                st.rpd = value
            _block(model, st, next_pacific_midnight(now), f"дневной лимит ({value or '?'} запросов) исчерпан")
            return
        if "perminute" in quota.lower():
            if value and "token" in quota.lower():
                st.tpm = value
            elif value:
                st.rpm = value
            _block(model, st, now + (retry or 60.0), "минутный лимит")
            return
        _block(model, st, now + (retry or 60.0), "лимит (429)")
        return

    # Groq: retry-after header, else "Please try again in 21.8s" in the message.
    wait = parse_duration(headers.get("retry-after"))
    if wait is None:
        m = _TRY_AGAIN_RX.search(text)
        wait = parse_duration(m.group(1)) if m else None
    low = text.lower()
    daily = "per day" in low or "(rpd)" in low or "(tpd)" in low
    if daily:
        _block(model, st, now + (wait or 3600.0), "дневной лимит исчерпан")
    else:
        # Per-minute: the bucket is empty now; what the header says is left
        # is what the next caller must plan with.
        st.remaining_tokens, st.seen = 0, now
        _block(model, st, now + (wait or 20.0), "минутный лимит токенов")


def _gemini_quota(data: Any) -> Tuple[str, Optional[int], Optional[float]]:
    """(quotaId, quotaValue, retryDelay seconds) from a Gemini 429 body."""
    quota, value, retry = "", None, None
    try:
        err = data.get("error", data) if isinstance(data, dict) else {}
        for d in err.get("details", []) or []:
            for v in d.get("violations", []) or []:
                quota = quota or str(v.get("quotaId", ""))
                try:
                    value = value or int(v.get("quotaValue"))
                except (TypeError, ValueError):
                    pass
            if "retryDelay" in d:
                retry = parse_duration(d.get("retryDelay"))
    except Exception:  # noqa: BLE001
        pass
    return quota, value, retry


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {}


def _block(model: str, st: ModelState, until: float, reason: str) -> None:
    if until > st.blocked_until:
        st.blocked_until = until
    st.reason = reason
    from utils.time import OWNER_TZ
    logger.warning("Модель %s: %s — не беру до %s", model, reason,
                   datetime.fromtimestamp(st.blocked_until, OWNER_TZ).strftime("%d.%m %H:%M:%S"))


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def wait_for(model: str, tokens: int = 0, priority: str = OWNER) -> float:
    """Seconds until `model` can take a request of ~`tokens` tokens.
    0 = now; math.inf = not today (blocked for long, over quota, too large)."""
    try:
        with _lock:
            return _wait_for(model, max(0, int(tokens)), priority)
    except Exception:  # noqa: BLE001
        logger.debug("llm_gate.wait_for failed", exc_info=True)
        return 0.0


def _wait_for(model: str, tokens: int, priority: str) -> float:
    st = _state(model)
    now = _now()
    waits = [0.0]
    if st.blocked_until > now:
        waits.append(st.blocked_until - now)

    # Daily requests: the owner may take the last ones, background may not.
    left = _requests_left(st)
    if left is not None:
        reserve = max(_RESERVE_MIN, math.ceil((st.rpd or 0) * _RESERVE_SHARE))
        if left <= 0 or (priority == BACKGROUND and left <= reserve):
            return math.inf

    # Requests per minute (known only after a Gemini 429 taught us).
    if st.rpm:
        recent = sorted(t for t in st.minute if now - t < 60)
        if len(recent) >= st.rpm:
            waits.append(60 - (now - recent[len(recent) - st.rpm]))

    # Tokens per minute: Groq's bucket refills continuously.
    if st.tpm and tokens:
        if tokens > st.tpm:
            return math.inf  # never fits this model
        if st.remaining_tokens is not None:
            rate = st.tpm / 60.0
            have = min(st.tpm, st.remaining_tokens + rate * max(0.0, now - st.seen))
            if tokens > have:
                waits.append((tokens - have) / rate)

    wait = max(waits)
    return math.inf if wait > 6 * 3600 else wait


def _requests_left(st: ModelState) -> Optional[int]:
    if st.remaining_requests is not None and st.seen:
        return st.remaining_requests
    if st.rpd:
        return st.rpd - st.used_today
    return None


def choose(models: Sequence[str], tokens: int = 0, priority: str = OWNER,
           max_wait: float = 0.0) -> Tuple[Optional[str], float]:
    """First model (in the caller's order of preference) that can take the
    request now. If none can, the one that frees up soonest, if within
    `max_wait`. Returns (model, seconds to wait) or (None, soonest wait)."""
    best: Tuple[Optional[str], float] = (None, math.inf)
    for m in dict.fromkeys(m for m in models if m):
        w = wait_for(m, tokens, priority)
        if w == 0:
            return m, 0.0
        if w < best[1]:
            best = (m, w)
    if best[0] is not None and best[1] <= max_wait:
        return best
    return None, best[1]


def too_large(models: Sequence[str], tokens: int) -> bool:
    """The request exceeds the per-minute token limit of every model whose
    limit is known (and at least one is known): waiting won't help."""
    with _lock:
        limits = [_state(m).tpm for m in models if m]
    known = [t for t in limits if t]
    return bool(known) and len(known) == len(limits) and all(tokens > t for t in known)


def open_models(models: Sequence[str], priority: str = OWNER) -> List[str]:
    """Models that are not blocked and have daily requests left (order kept)."""
    return [m for m in dict.fromkeys(m for m in models if m)
            if wait_for(m, 0, priority) < 120]


def gone(model: str) -> bool:
    """The provider answered 404 for this model recently (withdrawn)."""
    with _lock:
        st = _state(model)
        return st.blocked_until > _now() and "404" in st.reason


def blocked(model: str) -> bool:
    """Blocked for longer than a short pause: don't spend a call on it."""
    return wait_for(model, 0, OWNER) >= 120


def acquire(models: Sequence[str], tokens: int = 0, priority: str = OWNER,
            max_wait: float = 0.0, on_wait=None, sleep=None) -> Optional[str]:
    """choose() and, if a wait is needed and allowed, wait it out."""
    model, wait = choose(models, tokens, priority, max_wait)
    if model and wait > 0:
        if on_wait:
            try:
                on_wait(model, wait)
            except Exception:  # noqa: BLE001
                pass
        (sleep or time.sleep)(wait)
    return model


# ---------------------------------------------------------------------------
# For people and for the bot's own facts
# ---------------------------------------------------------------------------

def next_free(models: Sequence[str], tokens: int = 0, priority: str = OWNER) -> Optional[float]:
    """When (epoch) the first of `models` can take a request; None if one can
    now or nobody knows."""
    try:
        with _lock:
            now = _now()
            times = []
            for m in dict.fromkeys(m for m in models if m):
                w = _wait_for(m, max(0, int(tokens)), priority)
                if w == 0:
                    return None
                if w != math.inf:
                    times.append(now + w)
                    continue
                st = _state(m)
                if st.blocked_until > now:
                    times.append(st.blocked_until)
                elif provider_of(m) == "gemini" and st.rpd and st.used_today >= st.rpd:
                    times.append(next_pacific_midnight(now))
            return min(times) if times else None
    except Exception:  # noqa: BLE001
        return None


def describe(models: Optional[Sequence[str]] = None) -> List[str]:
    """One line per model that is limited right now, in plain Russian."""
    out = []
    try:
        with _lock:
            _load()
            from utils.time import OWNER_TZ
            now = _now()
            for m in (models or list(_states)):
                st = _state(m)
                if st.blocked_until > now:
                    until = datetime.fromtimestamp(st.blocked_until, OWNER_TZ)
                    day = "" if until.date() == datetime.fromtimestamp(now, OWNER_TZ).date()                         else until.strftime(" %d.%m")
                    out.append(f"{m}: {st.reason}, до{day} {until.strftime('%H:%M')}")
                elif st.rpd and _requests_left(st) is not None and _requests_left(st) <= 0:
                    out.append(f"{m}: дневной лимит исчерпан")
    except Exception:  # noqa: BLE001
        logger.debug("llm_gate.describe failed", exc_info=True)
    return out


def estimate_tokens(*parts: Any) -> int:
    """Rough token count of a request. ~3.3 characters per token: Russian text
    tokenizes denser than English, and underestimating here means a 429.
    Calibrated on the VM, Sep 29, 2026: the real Iris prompt with its tools
    was 19144 characters and 5692 tokens for Groq (3.36 per token)."""
    total = 0
    for p in parts:
        if p is None:
            continue
        s = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
        total += len(s)
    return int(total / 3.3)
