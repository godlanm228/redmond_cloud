"""The response generator under real limits (utils/llm_gate in the tool loops).

Sep 29, 2026: an Iris answer is ~2 hops of ~4.5K tokens; Groq gives 8000
tokens a minute per model; Gemini's free tier gives 20 requests a day per
model. Before the gate the second hop hit the wall, the chain fell over, and
Gemini was asked again and again after its daily quota was gone. These tests
drive the generator with a fake clock and fake providers that behave like the
real ones did that day.
"""

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from logic import response_generator as rgm
from logic.intent_recognizer import Intent
from utils import gemini, llm_gate

T0 = datetime.fromisoformat("2026-09-29T17:00:00+00:00").timestamp()
GROQ = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
GEMINI_DAILY_429 = {"error": {"code": 429, "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}


def _headers(left_tokens):
    return {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "900",
            "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": str(left_tokens)}


def _text(t):
    return {"choices": [{"finish_reason": "stop", "message": {"content": t}}]}


@pytest.fixture
def world(monkeypatch):
    """A fake clock that sleeping advances, and a generator with fake providers."""
    now = {"t": T0}
    slept = []
    monkeypatch.setattr(llm_gate, "_now", lambda: now["t"])

    def sleep(s):
        slept.append(s)
        now["t"] += s

    monkeypatch.setattr(llm_gate.time, "sleep", sleep)
    llm_gate.configure({m: {"rpd": 1000, "tpm": 8000} for m in GROQ} | {"gemini-3.6-flash": {"rpd": 20}},
                       {"chat_groq": GROQ, "chat_gemini": ["gemini-3.6-flash", "gemini-2.5-flash"],
                        "compose": ["openai/gpt-oss-120b", "gemini-2.5-flash"]})

    rg = object.__new__(rgm.ResponseGenerator)
    rg.config = SimpleNamespace(gemini_api_key="k", groq_api_key="k", groq_api_base="",
                                llm_provider_order=["groq", "gemini"])
    rg.mem, rg.top_k, rg.max_history = None, 3, 6
    rg.history_by_chat, rg._history_guard, rg._history_loaded = {}, threading.RLock(), set()
    rg._build_system_prompt = lambda ctx: "system " * 200
    rg._build_user_message = lambda ctx: ctx.user_text
    rg._save_interaction = lambda *a, **kw: None
    groq_calls, gemini_calls, statuses = [], [], []
    replies = {}

    def groq_chat(api_key, model, messages, tools=None, **kw):
        groq_calls.append(model)
        reply = replies.get(model, _text(f"ответ {model}"))
        llm_gate.report(model, 200, _headers(3000))
        return reply, ""

    def gen_contents(contents, **kw):
        gemini_calls.append(kw["model"])
        return {"candidates": [{"content": {"parts": [{"text": "ответ gemini"}]}}]}

    rg._groq_chat = groq_chat
    monkeypatch.setattr(gemini, "generate_contents", gen_contents)
    monkeypatch.setattr(gemini, "generate_text", lambda prompt, **kw: gemini_calls.append(kw.get("model")) or "")
    monkeypatch.setattr("logic.tools.execute_tool", lambda *a, **kw: "ok")
    return SimpleNamespace(rg=rg, now=now, slept=slept, groq=groq_calls, gemini=gemini_calls,
                           statuses=statuses)


IRIS = SimpleNamespace(name="Iris", provider_order=["groq", "gemini"], allowed_tools=[],
                       temperature=0.5, max_tokens=800, emoji="🎯")


def _ask(w, text="привет", include_history=True):
    return w.rg.generate(Intent(name="chat", slots={}), text, "owner", IRIS, 1,
                         status_cb=w.statuses.append, include_history=include_history)


def test_a_hop_goes_to_the_model_with_room_in_its_minute(world):
    llm_gate.report("openai/gpt-oss-120b", 200, _headers(200))  # the minute is spent
    reply = _ask(world)
    assert world.groq == ["qwen/qwen3.8-27b"], "the spent model was called anyway"
    assert reply.startswith("ответ qwen")


def test_when_every_model_is_busy_the_owner_waits_instead_of_failing(world):
    for m in GROQ:
        llm_gate.report(m, 200, _headers(0))
    reply = _ask(world)
    assert world.slept and 0 < world.slept[0] <= 30
    assert any("жду лимит" in s for s in world.statuses), "the owner was not told about the wait"
    assert not reply.failed and reply.startswith("ответ")


def test_gemini_out_for_the_day_is_not_asked_at_all(world):
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    llm_gate.report("gemini-2.5-flash", 429, body=GEMINI_DAILY_429)
    for m in GROQ:  # Groq out too: only Gemini would be left
        llm_gate.report(m, 429, {"retry-after": "7200"},
                        {"error": {"message": "Rate limit reached on requests per day (RPD)"}})
    reply = _ask(world)
    assert world.gemini == [] and world.groq == [], "a call was spent on a known-exhausted model"
    assert reply.failed
    # the real reason and the real time (Groq frees first: 19:00 + 2 h), not «повтори позже»
    assert "лимиты бесплатного тарифа" in reply and "21:00" in reply, reply


def test_a_scheduled_ping_leaves_the_owners_gemini_reserve(world):
    for m in GROQ:
        llm_gate.report(m, 429, {"retry-after": "7200"},
                        {"error": {"message": "Rate limit reached on requests per day (RPD)"}})
    for _ in range(12):  # 8 of 20 left: exactly the owner's reserve
        llm_gate.report("gemini-3.6-flash", 200)
    llm_gate.report("gemini-2.5-flash", 429, body=GEMINI_DAILY_429)
    _ask(world, "(scheduled, пинг дня) поздоровайся", include_history=False)
    assert world.gemini == [], "a ping spent the owner's reserve"
    _ask(world, "привет")
    assert world.gemini == ["gemini-3.6-flash"], "the owner could not use his reserve"


def test_the_owner_hears_the_real_wait_when_it_is_too_long(world):
    for m in GROQ:
        llm_gate.report(m, 429, {"retry-after": "95"}, {"error": {"message": "tokens per minute (TPM)"}})
    llm_gate.report("gemini-3.6-flash", 429, body=GEMINI_DAILY_429)
    llm_gate.report("gemini-2.5-flash", 429, body=GEMINI_DAILY_429)
    reply = _ask(world)
    assert reply.failed
    assert "95" in reply or "~2 мин" in reply or "сек" in reply, reply


MARKUP = ("<tool_call>\n<function=diary>\n<parameter=action>\nadd\n</parameter>\n"
          "<parameter=text>\nДовести до ума\n</parameter>\n</function>\n</tool_call>")


def test_a_tool_call_written_as_text_is_not_an_answer(world, monkeypatch):
    """Sep 29 run: qwen answered «Че за галлюцинации?» with raw tool markup."""
    from utils import groq
    replies = {"openai/gpt-oss-120b": ({"choices": [{"message": {"content": MARKUP}}]}, ""),
               "qwen/qwen3.8-27b": (_text("Записала план как план, не как сделанное."), "")}
    called = []

    def chat(model, messages, **kw):
        called.append(model)
        return replies.get(model, (_text("ок"), ""))

    monkeypatch.setattr(groq, "chat", chat)
    del world.rg._groq_chat  # the real one, through utils.groq
    reply = _ask(world)
    assert "<tool_call>" not in reply
    assert called[:2] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def test_the_rules_catch_tool_markup_as_a_leak():
    from evals import checks
    from types import SimpleNamespace as NS
    turn = NS(kind="owner", expect={})
    res = NS(replies=[MARKUP], agent="Iris", tool_calls=[], errors=[], provider=[], log=[],
             seconds=1.0, mute_after=None, skipped="")
    assert any(v.startswith("leak") for v in checks.check(turn, res, ["x"]))
