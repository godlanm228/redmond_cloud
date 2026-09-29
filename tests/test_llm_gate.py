"""utils/llm_gate: limits learned from the providers, decisions made before a call.

The shapes below are the real ones, captured on the VM on Sep 29, 2026:
the Gemini 429 that revealed the 20-requests-a-day free tier, and the Groq
headers and 429 from two 3.7K-token prompts that emptied an 8000-token minute.
"""

import math
from datetime import datetime
from types import SimpleNamespace

import pytest

from utils import llm_gate

GEMINI_DAILY_429 = {"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota, please check your plan and billing details.",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.Help",
         "links": [{"description": "Learn more", "url": "https://ai.google.dev/gemini-api/docs/rate-limits"}]},
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "quotaDimensions": {"location": "global", "model": "gemini-3.6-flash"},
            "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "29s"},
    ]}}

GEMINI_MINUTE_429 = {"error": {"code": 429, "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
        "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "10"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
]}}

GROQ_OK_HEADERS = {
    "x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "998",
    "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "741",
    "x-ratelimit-reset-tokens": "54.442s",
}
GROQ_TPM_429 = {"error": {
    "message": "Rate limit reached for model `openai/gpt-oss-20b` in organization `org_x` service tier "
               "`on_demand` on tokens per minute (TPM): Limit 8000, Used 7483, Requested 3743. "
               "Please try again in 24.169999999s.",
    "type": "tokens", "code": "rate_limit_exceeded"}}

# Tuesday Sep 29, 2026, 19:00 in Berlin = 10:00 Pacific
T0 = datetime.fromisoformat("2026-09-29T17:00:00+00:00").timestamp()


@pytest.fixture
def clock(monkeypatch):
    now = {"t": T0}
    monkeypatch.setattr(llm_gate, "_now", lambda: now["t"])
    return now


def test_a_daily_gemini_429_blocks_the_model_until_midnight_pacific(clock):
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    assert llm_gate.blocked("gemini-3.6-flash")
    # 10:00 Pacific → 14 hours to the reset, not the 29 s of retryDelay
    clock["t"] = T0 + 13.9 * 3600
    assert llm_gate.blocked("gemini-3.6-flash")
    clock["t"] = T0 + 14 * 3600 + 1
    assert not llm_gate.blocked("gemini-3.6-flash")
    assert llm_gate._state("gemini-3.6-flash").rpd == 20, "the limit was not learned"


def test_a_per_minute_gemini_429_is_a_short_pause(clock):
    llm_gate.report("gemini-2.5-flash", 429, body=GEMINI_MINUTE_429)
    w = llm_gate.wait_for("gemini-2.5-flash")
    assert 16 <= w <= 17
    assert llm_gate._state("gemini-2.5-flash").rpm == 10


def test_the_daily_quota_is_counted_before_the_provider_says_no(clock):
    llm_gate.configure({"gemini-3.6-flash": {"rpd": 20}})
    for _ in range(20):
        llm_gate.report("gemini-3.6-flash", 200)
    assert llm_gate.wait_for("gemini-3.6-flash") == math.inf
    # a new Pacific day starts at 09:00 Berlin
    clock["t"] = T0 + 15 * 3600
    assert llm_gate.wait_for("gemini-3.6-flash") == 0


def test_background_work_leaves_the_owners_reserve(clock):
    llm_gate.configure({"gemini-2.5-flash": {"rpd": 20}})
    for _ in range(12):
        llm_gate.report("gemini-2.5-flash", 200)
    assert llm_gate.wait_for("gemini-2.5-flash", priority=llm_gate.BACKGROUND) == math.inf
    assert llm_gate.wait_for("gemini-2.5-flash", priority=llm_gate.OWNER) == 0


def test_groq_headers_give_the_exact_wait_for_tokens(clock):
    llm_gate.report("openai/gpt-oss-120b", 200, GROQ_OK_HEADERS)
    st = llm_gate._state("openai/gpt-oss-120b")
    assert (st.tpm, st.rpd) == (8000, 1000)
    # 741 left, refill 8000/60 per second: 3743 tokens need ~22.5 s
    w = llm_gate.wait_for("openai/gpt-oss-120b", 3743)
    assert 22 <= w <= 23
    clock["t"] += 23
    assert llm_gate.wait_for("openai/gpt-oss-120b", 3743) == 0


def test_a_hop_goes_to_the_model_that_has_room(clock):
    llm_gate.report("openai/gpt-oss-120b", 200, GROQ_OK_HEADERS)
    model, wait = llm_gate.choose(["openai/gpt-oss-120b", "qwen/qwen3.8-27b"], 3743)
    assert (model, wait) == ("qwen/qwen3.8-27b", 0.0)


def test_when_every_model_is_busy_the_soonest_is_awaited_if_allowed(clock):
    llm_gate.report("openai/gpt-oss-20b", 429, {"retry-after": "25"}, GROQ_TPM_429)
    llm_gate.report("qwen/qwen3.8-27b", 429, {"retry-after": "40"}, GROQ_TPM_429)
    chain = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    assert llm_gate.choose(chain, 3000, max_wait=0)[0] is None
    assert llm_gate.choose(chain, 3000, max_wait=30) == ("openai/gpt-oss-20b", 25.0)


def test_a_prompt_larger_than_the_minute_never_fits(clock):
    llm_gate.report("qwen/qwen3.8-27b", 200, GROQ_OK_HEADERS)
    assert llm_gate.wait_for("qwen/qwen3.8-27b", 9000) == math.inf


def test_overload_and_withdrawal(clock):
    llm_gate.report("gemini-3.8-flash", 503)
    assert 59 <= llm_gate.wait_for("gemini-3.8-flash") <= 60
    clock["t"] += 61
    llm_gate.report("gemini-3.8-flash", 503)
    assert llm_gate.wait_for("gemini-3.8-flash") > 100, "repeated overload must back off longer"
    llm_gate.report("llama-3.1-8b-instant", 404)
    assert llm_gate.blocked("llama-3.1-8b-instant")


def test_a_block_survives_a_restart(clock):
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    llm_gate._states.clear()
    llm_gate._loaded = False  # a new process reads the ledger from the database
    assert llm_gate.blocked("gemini-3.6-flash")


def test_describe_speaks_berlin_time(clock):
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    lines = llm_gate.describe(["gemini-3.6-flash"])
    assert lines and "дневной лимит (20 запросов) исчерпан" in lines[0]
    assert "09:00" in lines[0]


@pytest.mark.parametrize("raw,sec", [("7.66s", 7.66), ("2m59.56s", 179.56), ("3h21m", 12060),
                                     ("29s", 29), ("120ms", 0.12), ("16", 16), ("", None)])
def test_durations_in_every_shape_providers_use(raw, sec):
    assert llm_gate.parse_duration(raw) == (pytest.approx(sec) if sec is not None else None)


def test_legacy_fields_lead_the_chat_pools():
    cfg = SimpleNamespace(groq_model="openai/gpt-oss-120b", groq_fallback_model="qwen/qwen3.8-27b",
                          gemini_model="gemini-3.6-flash", gemini_fallback_models=["gemini-2.5-flash"],
                          model_pools={"router": ["openai/gpt-oss-20b"]}, model_limits={})
    llm_gate.configure_from(cfg)
    assert llm_gate.pool("chat_groq")[:2] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    assert llm_gate.pool("chat_gemini")[:2] == ["gemini-3.6-flash", "gemini-2.5-flash"]
    assert llm_gate.pool("router") == ["openai/gpt-oss-20b"]
    llm_gate.replace_model("qwen/qwen3.8-27b", "qwen/qwen3.9-27b")
    assert "qwen/qwen3.9-27b" in llm_gate.pool("chat_groq")


# ---------- the clients ----------

class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = str(body)

    def json(self):
        return self._body


def test_a_blocked_gemini_model_is_not_called(clock, monkeypatch):
    from utils import gemini
    calls = []
    monkeypatch.setattr(gemini.requests, "post", lambda *a, **kw: calls.append(1) or _Resp(200))
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    assert gemini.generate_text("x", model="gemini-3.6-flash", api_key="k") == ""
    assert calls == []


def test_a_daily_429_is_not_retried(clock, monkeypatch):
    from utils import gemini
    calls = []
    monkeypatch.setattr(gemini.requests, "post",
                        lambda *a, **kw: calls.append(1) or _Resp(429, GEMINI_DAILY_429))
    monkeypatch.setattr("time.sleep", lambda s: None)
    gemini.generate_text("x", model="gemini-3.6-flash", api_key="k")
    assert calls == [1], "a daily limit was asked twice"
    assert llm_gate.blocked("gemini-3.6-flash")


def test_groq_client_reports_the_headers(clock, monkeypatch):
    from utils import groq
    reply = {"choices": [{"message": {"content": "ок"}}]}
    monkeypatch.setattr(groq.requests, "post", lambda *a, **kw: _Resp(200, reply, GROQ_OK_HEADERS))
    completion, err = groq.chat("openai/gpt-oss-120b", [{"role": "user", "content": "x"}], api_key="k")
    assert groq.text_of(completion) == "ок" and err == ""
    assert llm_gate._state("openai/gpt-oss-120b").remaining_tokens == 741


def test_text_skips_what_the_gate_knows_is_out(clock, monkeypatch):
    from utils import llm
    called = []
    monkeypatch.setattr(llm, "complete", lambda m, p, **kw: called.append(m) or ("ответ" if m == "b" else ""))
    llm_gate.report("a", 404)
    out, model = llm.text(["a", "c", "b"], "вопрос")
    assert (out, model) == ("ответ", "b")
    assert called == ["c", "b"], "a withdrawn model was called"


def test_gpt_oss_gets_room_to_reason(monkeypatch):
    from utils import groq, llm
    seen = {}
    monkeypatch.setattr(groq, "chat", lambda m, msgs, **kw: (seen.update(kw) or
                                                            ({"choices": [{"message": {"content": "Iris"}}]}, "")))
    assert llm.complete("openai/gpt-oss-20b", "кому?", max_tokens=20) == "Iris"
    assert seen["max_tokens"] > 300 and seen["extra"] == {"reasoning_effort": "low"}


def test_a_model_that_cannot_think_minimal_is_asked_again_at_its_floor(clock, monkeypatch):
    """gemini-3.7-flash, Sep 29: 400 «Thinking level MINIMAL is not supported»."""
    from utils import gemini
    bodies = []

    def post(url, headers=None, json=None, timeout=None):
        import copy
        bodies.append(copy.deepcopy(json))
        if json["generationConfig"]["thinkingConfig"].get("thinkingLevel") == "minimal":
            return _Resp(400, {"error": {"code": 400, "message":
                         "Thinking level MINIMAL is not supported for this model. "
                         "Please retry with other thinking level."}})
        return _Resp(200, {"candidates": [{"content": {"parts": [{"text": "Берлин"}]}}]})

    monkeypatch.setattr(gemini.requests, "post", post)
    gemini._LEVEL_FLOOR.pop("gemini-3.7-flash", None)
    try:
        assert gemini.generate_text("столица?", model="gemini-3.7-flash", max_tokens=20,
                                    api_key="k") == "Берлин"
        second = bodies[1]["generationConfig"]
        assert second["thinkingConfig"] == {"thinkingLevel": "low"}
        assert second["maxOutputTokens"] == 20 + gemini.THINKING_ALLOWANCE["low"], \
            "thinking would eat the 20-token answer"
        # the next call starts at the floor: no wasted 400
        bodies.clear()
        gemini.generate_text("ещё", model="gemini-3.7-flash", max_tokens=20, api_key="k")
        assert len(bodies) == 1
    finally:
        gemini._LEVEL_FLOOR.pop("gemini-3.7-flash", None)
