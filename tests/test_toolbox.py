"""Grouped tools (logic/toolbox) and on-demand loading (logic/tool_select)."""

import copy
import json
import threading
from types import SimpleNamespace

from logic import tool_select, toolbox
from logic.agents import IRIS, REDMOND
from logic.tools import TOOL_SCHEMAS


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


def _tokens(schemas):
    return round(len(json.dumps(schemas, ensure_ascii=False)) / 3.5)


# ---------- grouping ----------

def test_every_original_tool_stays_reachable_exactly_once():
    offered = toolbox.model_tools(None)
    reachable = []
    for s in offered:
        name = s["function"]["name"]
        if name in toolbox.GROUPS:
            enum = s["function"]["parameters"]["properties"]["action"]["enum"]
            reachable += [toolbox.GROUPS[name][1][a] for a in enum]
        else:
            reachable.append(name)
    assert sorted(reachable) == sorted(_names(TOOL_SCHEMAS))


def test_allowed_tools_restrict_the_actions_too():
    """The evening job may read the diary and list goals - and nothing else."""
    offered = toolbox.model_tools(["read_diary", "list_goals"])
    assert _names(offered) == ["diary", "goals"]
    enums = {s["function"]["name"]: s["function"]["parameters"]["properties"]["action"]["enum"]
             for s in offered}
    assert enums == {"diary": ["read"], "goals": ["list"]}


def test_grouping_makes_iris_schemas_lighter():
    """Sep 28, 2026: 27 schemas, ~3.9k tokens by this estimate."""
    before = [t for t in TOOL_SCHEMAS if t["function"]["name"] in IRIS.allowed_tools]
    after = toolbox.model_tools(IRIS.allowed_tools)
    assert len(after) <= 12
    assert _tokens(after) < 0.8 * _tokens(before)


def test_resolve_maps_to_the_original_tool_and_drops_foreign_fields():
    name, args, err = toolbox.resolve("diary", {"action": "add", "text": "Поел",
                                                "tags": ["питание"], "last_n": None})
    assert (name, err) == ("add_diary_entry", "")
    assert args == {"text": "Поел", "tags": ["питание"]}


def test_unknown_action_is_an_explained_error():
    name, _, err = toolbox.resolve("deadlines", {"action": "remove"})
    assert name == "deadlines" and "postpone" in err


def test_original_names_pass_through():
    """Old history or a model repeating an older prompt still works."""
    assert toolbox.resolve("add_diary_entry", {"text": "x"}) == ("add_diary_entry", {"text": "x"}, "")


def test_prompts_are_translated_to_the_offered_names():
    text = "read_diary (ids show as #N) → delete_diary_entry; postpone_deadline(id, new_due)"
    assert toolbox.rename_refs(text) == (
        "diary(action=read) (ids show as #N) → diary(action=delete); "
        "deadlines(action=postpone, id, new_due)")


def test_offered_descriptions_never_mention_hidden_names():
    """A description saying 'call list_deadlines' names a tool the model cannot call."""
    for s in toolbox.model_tools(None):
        desc = s["function"]["description"]
        assert toolbox.rename_refs(desc) == desc, s["function"]["name"]


def test_renaming_is_idempotent():
    text = "log_meal, then food(action=log_meal); read_diary"
    once = toolbox.rename_refs(text)
    assert toolbox.rename_refs(once) == once
    assert once == "food(action=log_meal), then food(action=log_meal); diary(action=read)"


# ---------- selection ----------

def _unit(i, n=8):
    v = [0.0] * n
    v[i] = 1.0
    return v


def test_small_tool_sets_are_offered_whole():
    small = toolbox.model_tools(["read_diary", "list_goals"])
    offered, deferred, how = tool_select.select("Iris", "что по целям?", small)
    assert (offered, deferred, how) == (small, [], "all")


def test_vectors_pick_the_closest_tools_plus_core(monkeypatch):
    from utils import embeddings
    tools = toolbox.model_tools(IRIS.allowed_tools)
    names = _names(tools)
    vecs = {n: ("h", _unit(i, len(names))) for i, n in enumerate(names)}
    monkeypatch.setattr(embeddings, "sync", lambda kind, items, limit=500: 0)
    monkeypatch.setattr(embeddings, "load", lambda kind: vecs)
    query = _unit(names.index("food"), len(names))  # "что поесть?" is about food
    offered, deferred, how = tool_select.select("Iris", "что поесть?", tools, query)
    assert how == "vectors"
    assert {"food", "diary", "get_current_time"} <= set(_names(offered))
    assert len(offered) == len(tool_select.CORE["Iris"]) + tool_select.TOP_K
    assert set(_names(offered)) | set(_names(deferred)) == set(names)


def test_without_vectors_keywords_still_select():
    tools = toolbox.model_tools(IRIS.allowed_tools)
    offered, deferred, how = tool_select.select("Iris", "перенеси дедлайн", tools, None)
    assert how == "keywords" and deferred


def test_selection_saves_most_of_the_schema_budget():
    tools = toolbox.model_tools(REDMOND.allowed_tools)
    offered, deferred, _ = tool_select.select("Redmond", "какая погода завтра", tools, None)
    sent = offered + [tool_select.load_tools_schema(deferred)]
    assert _tokens(sent) < 0.6 * _tokens(tools)


def test_load_tools_moves_what_was_asked_for():
    tools = toolbox.model_tools(IRIS.allowed_tools)
    offered, deferred, _ = tool_select.select("Iris", "привет", tools, None)
    wanted = _names(deferred)[0]
    loaded, unknown = tool_select.apply_load([wanted, "nonsense"], deferred, offered)
    assert loaded == [wanted] and unknown == ["nonsense"]
    assert wanted in _names(offered) and wanted not in _names(deferred)


def test_load_tools_catalogue_lists_every_deferred_tool():
    tools = toolbox.model_tools(IRIS.allowed_tools)
    _offered, deferred, _ = tool_select.select("Iris", "привет", tools, None)
    desc = tool_select.load_tools_schema(deferred)["function"]["description"]
    assert all(f"- {n}:" in desc for n in _names(deferred))


def test_keyword_fallback_understands_russian():
    """Real messages from the production log; no vectors (API down)."""
    tools = toolbox.model_tools(IRIS.allowed_tools)
    cases = {
        "перенеси дедлайн по матану на пятницу": "deadlines",
        "Курю кальян с Настей, потом поем": None,
        "что приготовить из того что есть в холодильнике": "food",
        "Мут на 7 дней": "mute_notifications",
        "с 17 до 23 смена в баре": "schedule",
    }
    for text, expected in cases.items():
        offered, _deferred, how = tool_select.select("Iris", text, tools, None)
        assert how == "keywords"
        if expected:
            assert expected in _names(offered), (text, _names(offered))


# ---------- load_tools inside the generator ----------

def test_model_can_load_a_hidden_tool_and_use_it(monkeypatch):
    from logic import response_generator as rgm
    from logic.intent_recognizer import Intent
    from utils import gemini

    def fc(name, args):
        return {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": name, "args": args}, "thoughtSignature": "sig"}]}}]}

    replies = [fc("load_tools", {"names": ["food"]}),
               fc("food", {"action": "pantry"}),
               {"candidates": [{"content": {"role": "model", "parts": [{"text": "Дома есть гречка."}]}}]}]
    offered_per_hop = []

    def fake_contents(contents, **kw):
        offered_per_hop.append([d["name"] for t in (kw.get("tools") or [])
                                for d in t.get("functionDeclarations", [])])
        return replies.pop(0)

    executed = []
    monkeypatch.setattr(gemini, "generate_contents", fake_contents)
    monkeypatch.setattr("logic.tools.execute_tool",
                        lambda name, args, rg=None, session=None: executed.append(name) or "гречка")
    monkeypatch.delenv("REDMOND_GEMINI_API_KEY", raising=False)

    rg = object.__new__(rgm.ResponseGenerator)
    rg.config = SimpleNamespace(gemini_api_key="k", gemini_model="gemini-3.6-flash",
                                groq_api_key="", groq_model="", groq_fallback_model="")
    rg.mem, rg.top_k, rg.max_history = None, 3, 6
    rg.history_by_chat, rg._history_guard, rg._history_loaded = {}, threading.RLock(), set()
    rg._build_system_prompt = lambda ctx: "system"
    rg._build_user_message = lambda ctx: ctx.user_text
    rg._save_interaction = lambda *a, **kw: None
    from logic.agents import IRIS
    iris = copy.copy(IRIS)
    iris.provider_order = ["gemini"]

    reply = rg.generate(Intent(name="chat", slots={}), "привет, как дела?", "owner", iris, 1)
    assert "food" not in offered_per_hop[0], "food should start hidden for a greeting"
    assert "load_tools" in offered_per_hop[0]
    assert "food" in offered_per_hop[1], "loaded tool must be offered on the next step"
    assert executed == ["get_pantry"]
    assert reply.startswith("Дома есть гречка.")
