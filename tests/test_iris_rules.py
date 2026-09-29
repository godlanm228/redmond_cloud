"""logic/iris_rules: Iris's rules travel with the tools they are about.

Sep 30, 2026: every step of every Iris answer carried the whole rulebook
(~3.5K tokens of system prompt) - one message cost ~15K of the 200K tokens a
Groq model gives a day.
"""

from logic import iris_rules, toolbox
from logic.tools import TOOL_SCHEMAS


def _text(tools, scheduled=False):
    return "\n".join(iris_rules.lines(tools, "вт, 29.09 12:00", scheduled=scheduled))


def test_rules_come_only_with_their_tools():
    text = _text({"read_diary", "delete_diary_entry"})
    assert "DIARY FIXES" in text
    for other in ("FOOD & PANTRY", "WEEK PLAN", "SHIFTS", "DEADLINES", "mute_notifications",
                  "add_diary_entry = REAL events"):
        assert other not in text, other


def test_without_knowing_the_tools_every_rule_is_sent():
    text = _text(None)
    for block in ("DIARY:", "DIARY FIXES", "SHIFTS", "DEADLINES", "WEEK PLAN", "FOOD & PANTRY",
                  "mute_notifications", "update_profile", "delegate_research", "read_dossier_section"):
        assert block in text, block


def test_the_base_is_much_smaller_than_the_old_rulebook():
    from utils import llm_gate
    base = llm_gate.estimate_tokens(_text(set()))
    full = llm_gate.estimate_tokens(_text(None))
    assert base < full / 2, (base, full)


def test_scheduled_rules_only_for_scheduled_prompts():
    assert "PINGS:" not in _text(set())
    assert "PINGS:" in _text(set(), scheduled=True)


def test_every_tool_a_rule_is_about_exists():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    for module, belongs, _lines in iris_rules.MODULES:
        missing = belongs - names
        assert not missing, (module, missing)


def test_group_schemas_expand_to_their_actions():
    schemas = toolbox.model_tools(["read_diary", "delete_diary_entry", "log_meal"])
    assert toolbox.legacy_names(schemas) == {"read_diary", "delete_diary_entry", "log_meal"}


def test_a_tool_loaded_mid_answer_brings_its_rules():
    from logic.response_generator import ResponseGenerator
    deferred = toolbox.model_tools(["get_week_plan", "save_week_plan"])
    offered = []
    result = ResponseGenerator._load_tools("Iris", {"names": ["schedule"]}, deferred, offered)
    assert "WEEK PLAN" in result and "save_week_plan with EXACTLY" in result
