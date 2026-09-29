"""The owner's dialogue of Sep 29, 2026, replayed.

Every message here is verbatim from that day. The replies he got were:
  * his plan for the day -> an "honest failure" whose receipt listed refused
    diary writes ("Не записано: ...") as if they were done, and the plan
    itself vanished from the history;
  * "Это как?" -> Iris reviewing the text of her own scheduled ping, because
    the ping's prompt sat in the history as the owner's message;
  * "В больнице лежу, мут до первого числа" -> only a question; the mute
    had to be asked for twice.
Gemini answered 503 all day, so every turn fell through to Groq.
"""

import threading
from types import SimpleNamespace

from logic import response_generator as rgm
from logic.intent_recognizer import Intent
from utils import gemini

PING_PROMPT = ("(scheduled, пинг дня) Влад сегодня ещё не на связи, записей за день нет. "
               "Поздоровайся тепло, по-человечески спроси как он и какие планы на день")
PING_REPLY = "Добрый день, Влад! Чем планируешь заняться сегодня?"
PLAN = ("Довести до ума и презентабельности вид редмонд клауда, тоесть вас "
        "и подготовиться к собеседованию")

IRIS = SimpleNamespace(name="Iris", provider_order=["gemini"], allowed_tools=None,
                       temperature=0.5, max_tokens=800, emoji="🎯")


def _text(t):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]}}]}


def _call(name, args):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": name, "args": args}, "thoughtSignature": "sig"}]}}]}


def _rg(monkeypatch, replies_by_model, fallbacks=()):
    rg = object.__new__(rgm.ResponseGenerator)
    rg.config = SimpleNamespace(gemini_api_key="k", gemini_model="gemini-3.6-flash",
                                gemini_fallback_models=list(fallbacks),
                                groq_api_key="", groq_model="", groq_fallback_model="")
    rg.mem, rg.top_k, rg.max_history = None, 3, 6
    rg.history_by_chat, rg._history_guard, rg._history_loaded = {}, threading.RLock(), set()
    rg._build_system_prompt = lambda ctx: "system"
    rg._build_user_message = lambda ctx: ctx.user_text
    saved, seen = [], []
    rg._save_interaction = lambda user, resp, chat_id=0, agent="", history_only=False: \
        saved.append((user, resp, history_only))

    def fake(contents, **kw):
        seen.append((kw["model"], [d["name"] for t in (kw.get("tools") or [])
                                   for d in t.get("functionDeclarations", [])]))
        queue = replies_by_model.get(kw["model"], [])
        return queue.pop(0) if queue else None

    monkeypatch.setattr(gemini, "generate_contents", fake)
    monkeypatch.setattr(gemini, "generate_text", lambda prompt, **kw: "")
    monkeypatch.setattr("logic.tools.execute_tool",
                        lambda name, args, rg=None, session=None: f"{name} ok")
    monkeypatch.delenv("REDMOND_GEMINI_API_KEY", raising=False)
    return rg, saved, seen


def _ask(rg, text):
    return rg.generate(Intent(name="chat", slots={}), text, "owner", IRIS, 1)


def test_scheduled_ping_is_not_shown_as_the_owners_words():
    rg = object.__new__(rgm.ResponseGenerator)
    ctx = rgm.GenerationContext(intent=Intent(name="chat", slots={}), user_text="Это как?",
                                history=[{"user": PING_PROMPT, "bot": PING_REPLY}])
    rendered = rgm.ResponseGenerator._build_user_message(rg, ctx)
    assert "Я: (scheduled" not in rendered
    assert "Поздоровайся" not in rendered
    assert f"Ты (сама, по расписанию): {PING_REPLY}" in rendered


def test_owners_words_survive_a_failed_turn(monkeypatch):
    rg, saved, _ = _rg(monkeypatch, {})
    reply = _ask(rg, PLAN)
    assert reply.failed
    assert saved and saved[-1][0] == PLAN and saved[-1][2] is True, \
        "his plan must reach the history even when no model answered"


def test_refused_writes_are_not_reported_as_done(monkeypatch):
    """The model invented 'Завтрак после пробуждения.' - refused, and not in the receipt."""
    rg, _saved, _ = _rg(monkeypatch, {"gemini-3.6-flash": [
        _call("diary", {"action": "add", "text": "Завтрак после пробуждения."}),
        _text("Записала твой план на день."),
    ]})
    reply = _ask(rg, PLAN)
    assert "Не записано" not in reply
    assert "📝" not in reply


def test_an_overloaded_gemini_model_is_replaced_before_giving_up(monkeypatch):
    rg, _saved, seen = _rg(monkeypatch, {"gemini-3.8-flash": [_text("Привет! Что за план?")]},
                           fallbacks=["gemini-3.8-flash"])
    reply = _ask(rg, "привет")
    assert reply == "Привет! Что за план?"
    assert [m for m, _ in seen] == ["gemini-3.6-flash", "gemini-3.8-flash"]


def test_an_explicit_mute_is_carried_out_in_an_acute_situation(monkeypatch):
    rg, _saved, seen = _rg(monkeypatch, {"gemini-3.6-flash": [_text("Держись. Что случилось?")]})
    _ask(rg, "В больнице лежу , мут до первого числа")
    offered = seen[0][1]
    assert offered == ["mute_notifications"], offered
