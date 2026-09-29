"""The understanding step inside the bot: code acts, the answering model is told.

Cases from the real conversations (Sep 2026) the scenario runs replayed.
"""

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from logic import coach_storage
from logic import response_generator as rgm
from logic import understanding as und
from logic.intent_recognizer import Intent
from utils import gemini
from utils.time import OWNER_TZ, set_clock

HOSPITAL = "В больнице лежу , мут до первого числа"
PLAN = "Довести до ума и презентабельности вид редмонд клауда , тоесть вас и подготовиться к собеседованию"


def _u(message, **kw):
    base = dict(addressee="Iris", about="", refers_to="", urgency="none", urgency_quote="")
    base.update(kw)
    facts = [und.Fact(**f) for f in base.pop("facts", [])]
    commands = [und.Command(**c) for c in base.pop("commands", [])]
    return und.Understanding(facts=facts, commands=commands, **base)


@pytest.fixture
def at_1644():
    set_clock(lambda: datetime(2026, 9, 29, 16, 44, tzinfo=OWNER_TZ))
    yield
    set_clock(None)


@pytest.fixture
def rg(monkeypatch):
    """Iris on Gemini with a fake model that records what it was offered and told."""
    g = object.__new__(rgm.ResponseGenerator)
    g.config = SimpleNamespace(gemini_api_key="k", groq_api_key="", llm_provider_order=["gemini"])
    g.mem, g.top_k, g.max_history = None, 3, 6
    g.history_by_chat, g._history_guard, g._history_loaded = {}, threading.RLock(), set()
    g._build_system_prompt = rgm.ResponseGenerator._build_system_prompt.__get__(g)
    g._save_interaction = lambda *a, **kw: None
    g.seen = {"tools": [], "user": [], "system": []}

    def fake(contents, **kw):
        g.seen["tools"].append({d["name"] for t in (kw.get("tools") or [])
                                for d in t.get("functionDeclarations", [])})
        g.seen["user"].append(contents[0]["parts"][0]["text"])
        g.seen["system"].append(kw.get("system", ""))
        return {"candidates": [{"content": {"parts": [{"text": "Держись. Что случилось?"}]}}]}

    monkeypatch.setattr(gemini, "generate_contents", fake)
    monkeypatch.setattr(g, "_build_system_prompt", lambda ctx: "SYSTEM" + (
        "\n" + und.ATTENTION if ctx.attention else "") + ("\nDISTRESS" if ctx.distress else ""))
    return g


IRIS = SimpleNamespace(name="Iris", provider_order=["gemini"], allowed_tools=None,
                       temperature=0.5, max_tokens=800, emoji="🎯")


def _ask(g, text, u):
    return g.generate(Intent(name="chat", slots={}), text, "owner", IRIS, 1, understanding=u)


def test_hospital_and_mute_are_done_by_code_and_answered_calmly(rg, at_1644):
    u = _u(HOSPITAL, urgency="attention", urgency_quote="В больнице лежу",
           facts=[dict(quote="В больнице лежу", fact="Лежит в больнице", when="now", topic="здоровье")],
           commands=[dict(quote="мут до первого числа", type="mute", until="2026-10-01", scope="all")])
    reply = _ask(rg, HOSPITAL, u)

    assert coach_storage.mute_info()["until"].startswith("2026-10-01T00:00")
    entries = coach_storage.read_diary(last_n=3)
    assert entries[-1]["text"] == "В больнице лежу", "the diary must hold his words"
    assert "сейчас" in entries[-1]["tags"]
    assert "mute_notifications" not in rg.seen["tools"][0], "the mute was already done"
    ctx = SimpleNamespace(understanding=u, code_actions=u.done)
    assert rgm.ResponseGenerator._excluded_tools(ctx) == {"add_diary_entry", "mute_notifications"}
    assert und.ATTENTION in rg.seen["system"][0] and "DISTRESS" not in rg.seen["system"][0]
    assert "[Как код понял сообщение владельца]" in rg.seen["user"][0]
    assert "🔕" in reply and "📝" in reply, "the receipt under the answer"


def test_a_plan_goes_to_the_diary_as_a_plan_in_his_words(rg, at_1644):
    u = _u(PLAN, facts=[dict(quote="Довести до ума и презентабельности вид редмонд клауда",
                             fact="Планирует довести Redmond Cloud до ума", when="plan", topic="работа")])
    _ask(rg, PLAN, u)
    e = coach_storage.read_diary(last_n=1)[-1]
    assert e["text"] == "Довести до ума и презентабельности вид редмонд клауда"
    assert e["tags"][:1] == ["план"]


def test_the_model_is_not_offered_the_diary_write_when_code_recorded(rg, at_1644):
    from logic import toolbox
    u = _u("поел борщ", facts=[dict(quote="поел борщ", fact="Поел борщ", when="done", topic="еда")])
    _ask(rg, "поел борщ", u)
    diary_schema = next(s for s in toolbox.model_tools(None, exclude={"add_diary_entry"})
                        if s["function"]["name"] == "diary")
    assert "add" not in str(diary_schema["function"]["parameters"]["properties"]["action"].get("enum"))


def test_a_handoff_does_not_act_twice(rg, at_1644):
    u = _u("поел борщ", facts=[dict(quote="поел борщ", fact="Поел борщ", when="done", topic="еда")])
    _ask(rg, "поел борщ", u)
    reply = _ask(rg, "поел борщ\n\n(передано от Redmond)", u)
    assert [e["text"] for e in coach_storage.read_diary(last_n=5)].count("Поел борщ") == 1
    assert "📝" in reply, "the one who answers shows the receipt"


def test_a_crisis_is_still_the_crisis_path(rg, at_1644):
    u = _u("Суисайд", urgency="crisis", urgency_quote="Суисайд")
    _ask(rg, "Суисайд", u)
    assert "DISTRESS" in rg.seen["system"][0]


def test_the_router_takes_the_addressee_and_never_drops_a_fact():
    from logic import agent_router
    state = agent_router.RouterState()
    agent, research = agent_router.route("что нового по крипте", state, "",
                                         understood=_u("", addressee="Newser", research=True))
    assert (agent.name, research) == ("Newser", True)
    kalyan = _u("Курю кальян", addressee=und.NOBODY,
                facts=[dict(quote="Курю кальян", fact="Курит кальян", when="now", topic="отдых")])
    agent, _ = agent_router.route("Курю кальян", agent_router.RouterState(), "", understood=kalyan)
    assert agent is not None and agent.name == "Iris", "a fact must not be dropped by silence"


def test_cipher_only_when_addressed():
    """Probe, Sep 29: «Да но ты и шифр два разных бота / Поч он отвечал вместо тебя»
    was read as addressed to Cipher - it is about him."""
    from logic import agent_router
    about_cipher = _u("", addressee="Cipher")
    agent, _ = agent_router.route("Да но ты и шифр два разных бота\nПоч он отвечал вместо тебя",
                                  agent_router.RouterState(), "", understood=about_cipher)
    assert agent.name == "Redmond"
    agent, _ = agent_router.route("шифр, перезапусти сервис", agent_router.RouterState(), "",
                                  understood=about_cipher)
    assert agent.name == "Cipher", "an explicit address still reaches him"


def test_nothing_goes_to_the_diary_in_a_crisis(rg, at_1644):
    """Run 4 (Sep 30): «Суисайд» → the diary got «Влад сообщил о суициде» - the
    very interpretation that went there on Sep 4."""
    u = _u("Суисайд", urgency="crisis", urgency_quote="Суисайд",
           facts=[dict(quote="Суисайд", fact="Влад сообщил о суициде", when="now", topic="здоровье")])
    before = len(coach_storage.read_diary(last_n=50))
    _ask(rg, "Суисайд", u)
    assert len(coach_storage.read_diary(last_n=50)) == before
