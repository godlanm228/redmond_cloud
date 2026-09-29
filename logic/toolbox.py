"""The tool set the model sees: grouped tools with an `action`, filtered per agent.

Why. Every tool schema is part of every request, on every hop of the tool loop.
On Sep 28, 2026 Iris carried 27 schemas (~3.3k tokens) and Redmond all 34
(~5k): more than half of each request, against a Groq limit of 8k tokens a
minute. Twenty of those tools were fine-grained variations of five subjects
(add/read/delete a diary entry, add/list/close/delete/postpone a deadline...),
so a model choosing among them also had more ways to pick the wrong one.

Here they are folded into five grouped tools - diary, goals, deadlines, food,
schedule - each with an `action`. The original tools stay as the executors
behind them: argument validation against their own schemas, the
"change only what you have seen" guard, receipts and output essentials all
keep working on the original names. `resolve()` is the single translation
point from what the model called to what runs.

Standalone tools (mute, profile, dossier, photos, delegation, time, web)
are offered as they are.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from logic.tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

# group -> (what it is, {action: original tool}, {action: one-line summary})
#
# Summaries are short on purpose. The detailed rules (when to log, how to fix
# an entry, postponing vs adding) live in the agent's system prompt already;
# repeating them in every schema on every hop is what made the schemas heavy.
GROUPS: Dict[str, Tuple[str, Dict[str, str]]] = {
    "diary": (
        "The owner's diary of real events, states and decisions.",
        {"add": "add_diary_entry", "read": "read_diary", "delete": "delete_diary_entry"},
    ),
    "goals": (
        "The owner's long-term goals.",
        {"add": "add_goal", "list": "list_goals", "done": "mark_goal_done"},
    ),
    "deadlines": (
        "The owner's dated deadlines (exams, submissions, appointments).",
        {"add": "add_deadline", "list": "list_deadlines", "done": "mark_deadline_done",
         "delete": "delete_deadline", "postpone": "postpone_deadline"},
    ),
    "food": (
        "Meals, nutrition lookup and the pantry.",
        {"log_meal": "log_meal", "lookup": "lookup_food", "pantry": "get_pantry",
         "pantry_update": "update_pantry"},
    ),
    "schedule": (
        "Work shifts, university classes and the week plan.",
        {"view": "get_week_schedule", "save_shift": "save_work_shift",
         "shift_status": "set_work_shift_status", "resolve_conflict": "resolve_shift_conflict",
         "plan": "get_week_plan", "plan_save": "save_week_plan"},
    ),
}

SUMMARIES: Dict[str, str] = {
    "add_diary_entry": "log a real event/state/decision the owner reported, with tags",
    "read_diary": "recent entries (optionally by tag); ids show as #N",
    "delete_diary_entry": "delete entries by #id taken from a read in this turn",
    "add_goal": "create a long-term goal",
    "list_goals": "list goals (optionally by status)",
    "mark_goal_done": "close a goal by #id",
    "add_deadline": "add a NEW dated deadline (never for a postponement)",
    "list_deadlines": "list open deadlines, overdue first",
    "mark_deadline_done": "close a deadline that passed or was submitted",
    "delete_deadline": "remove a deadline created by mistake",
    "postpone_deadline": "move an EXISTING deadline to a new date",
    "log_meal": "log what he ate with honest kcal/protein estimates",
    "lookup_food": "exact nutrition of a packaged product by barcode or name",
    "get_pantry": "what food he has at home",
    "update_pantry": "add/remove pantry items",
    "get_week_schedule": "shifts + classes for the next days",
    "save_work_shift": "save a work shift he reported (date, start, end)",
    "set_work_shift_status": "mark a shift cancelled/uncertain/confirmed",
    "resolve_shift_conflict": "apply his answer to a photo-vs-text shift conflict",
    "get_week_plan": "read the saved week plan",
    "save_week_plan": "save the week plan text exactly as shown to him",
}

_BY_NAME: Dict[str, dict] = {t["function"]["name"]: t for t in TOOL_SCHEMAS}

# original tool -> (group, action)
GROUPED: Dict[str, Tuple[str, str]] = {
    legacy: (group, action)
    for group, (_, actions) in GROUPS.items()
    for action, legacy in actions.items()
}


def call_name(legacy: str) -> str:
    """How the model calls an original tool: 'diary(action=add)' or the name itself."""
    if legacy in GROUPED:
        group, action = GROUPED[legacy]
        return f"{group}(action={action})"
    return legacy


# Longest names first, so that e.g. `read_diary` never matches inside another name.
# Not after `action=`: that is already the new form, so renaming is idempotent
# (food(action=log_meal) stays as it is).
_RENAME_RX = re.compile(
    r"(?<!action=)\b(" + "|".join(sorted(map(re.escape, GROUPED), key=len, reverse=True))
    + r")\b(\()?"
)


def rename_refs(text: str) -> str:
    """Rewrite mentions of original tool names into how the model calls them.

    Prompts and tool descriptions were written against the original names
    ("read_diary (ids show as #N) -> delete_diary_entry"). Translating them in
    one place, from the same GROUPS table, keeps every prompt consistent with
    the tools actually offered:
        read_diary              -> diary(action=read)
        postpone_deadline(id, x) -> deadlines(action=postpone, id, x)
    """
    def sub(m: "re.Match[str]") -> str:
        group, action = GROUPED[m.group(1)]
        return f"{group}(action={action}, " if m.group(2) else f"{group}(action={action})"
    return _RENAME_RX.sub(sub, text or "")


def _group_schema(group: str, actions: List[str]) -> dict:
    intro, mapping = GROUPS[group]
    props: Dict[str, dict] = {}
    lines = [intro, "Pick the action; each lists what it needs."]
    for action in actions:
        f = _BY_NAME[mapping[action]]["function"]
        params = f.get("parameters", {})
        needs = ", ".join(params.get("required", [])) or "nothing"
        summary = SUMMARIES.get(mapping[action]) or rename_refs(f.get("description", ""))
        lines.append(f"- action={action} (needs: {needs}): {summary}")
        for p, spec in params.get("properties", {}).items():
            if spec.get("description"):
                spec = {**spec, "description": rename_refs(spec["description"])}
            if p not in props:
                props[p] = spec
            elif props[p].get("type") != spec.get("type"):
                raise ValueError(f"{group}: parameter {p!r} has different types across actions")
    props = {"action": {"type": "string", "enum": list(actions),
                        "description": "What to do."}, **props}
    return {"type": "function", "function": {
        "name": group,
        "description": "\n".join(lines),
        "parameters": {"type": "object", "properties": props, "required": ["action"]},
    }}


def model_tools(allowed: Optional[Iterable[str]] = None,
                exclude: Iterable[str] = ()) -> List[dict]:
    """Schemas to offer the model. `allowed` lists ORIGINAL tool names (as in
    AgentConfig.allowed_tools); None means everything. `exclude` removes original
    tools (code already did that job, e.g. recorded the message's facts). A
    group is offered with only the actions whose original tools remain."""
    allowed_set = None if allowed is None else set(allowed)
    excluded = set(exclude or ())
    out: List[dict] = []
    for group, (_, mapping) in GROUPS.items():
        actions = [a for a, legacy in mapping.items()
                   if (allowed_set is None or legacy in allowed_set) and legacy not in excluded]
        if actions:
            out.append(_group_schema(group, actions))
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        if name in GROUPED:
            continue
        if (allowed_set is None or name in allowed_set) and name not in excluded:
            f = schema["function"]
            desc = rename_refs(f.get("description", ""))
            out.append(schema if desc == f.get("description", "") else
                       {**schema, "function": {**f, "description": desc}})
    return [_nullable_optionals(s) for s in out]


def _nullable_optionals(schema: dict) -> dict:
    """Every optional parameter also accepts null.

    Models fill optional fields with null instead of leaving them out, and Groq
    validates tool calls against the schema on its side: 29.09.2026 «в больнице
    лежу, мут до первого числа» failed with 400 «/hours: expected number, but
    got null» — mode and scope allowed null, days and hours did not. The code
    already reads None as «not given». Gemini's schema conversion drops null."""
    fn = schema.get("function") or {}
    params = fn.get("parameters")
    if not isinstance(params, dict) or not params.get("properties"):
        return schema
    return {**schema, "function": {**fn, "parameters": _nullable_object(params)}}


def _nullable_object(obj: dict) -> dict:
    required = set(obj.get("required") or [])
    props = {}
    for name, prop in (obj.get("properties") or {}).items():
        prop = dict(prop)
        if prop.get("type") == "object" and prop.get("properties"):
            prop = _nullable_object(prop)
        if name not in required and "type" in prop:
            t = prop["type"]
            types = list(t) if isinstance(t, list) else [t]
            if "null" not in types:
                prop["type"] = types + ["null"]
        props[name] = prop
    return {**obj, "properties": props}


def resolve(name: str, args: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str]:
    """What the model called -> (original tool, its arguments, error).

    Arguments that belong to other actions of the same group are dropped:
    models often send every field of a grouped schema with nulls. An original
    tool name called directly (older prompts, history) passes through as is.
    """
    args = dict(args or {})
    if name not in GROUPS:
        return name, args, ""
    mapping = GROUPS[name][1]
    action = str(args.pop("action", "") or "").strip()
    if action not in mapping:
        return name, args, (f"Неизвестное действие {action!r} для {name}. "
                            f"Доступно: {', '.join(mapping)}.")
    legacy = mapping[action]
    known = _BY_NAME[legacy]["function"].get("parameters", {}).get("properties", {})
    dropped = {k: v for k, v in args.items() if k not in known and v not in (None, "", [])}
    if dropped:
        logger.info("toolbox: %s(action=%s) — чужие поля отброшены: %s", name, action, dropped)
    return legacy, {k: v for k, v in args.items() if k in known}, ""
