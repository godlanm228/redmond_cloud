"""Rule checks for one turn of a scenario run.

Rules catch what never depends on taste: a failed or stubbed reply, internal
text leaking into the chat, a diary entry the owner never said, an expectation
of the scenario not met. Everything that needs judgement (was the reply
relevant, invented, well-toned) is left to the judge.

The checks are written independently of the bot's own guards on purpose: a
check that reused `_ungrounded_write` would share its blind spots.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

# Replies that are not answers. The first one was the Gemini failure of
# Aug 15 - Sep 28, 2026 (14 of 34 owner messages).
_STUBS = [
    re.compile(r"^\s*Готово\s*[.:!]?\s*(записала[^\n]*)?\s*$", re.I),
    re.compile(r"Модели не ответили"),
    re.compile(r"Не могу выполнить эту команду"),
]

# Internal text that must never reach the owner.
_LEAKS = [
    re.compile(r"\(scheduled", re.I),
    re.compile(r"OWNER FACTS|INJECTION DEFENSE|SYSTEM FACTS|TOOLS? AVAILABLE", re.I),
    re.compile(r"\b(?:add_diary_entry|mute_notifications|update_profile|read_dossier(?:_section)?|"
               r"load_tools|delegate_research|get_current_time|web_search)\b"),
    re.compile(r"\baction=\w+"),
]

_WORD = re.compile(r"[a-zа-яё0-9]+", re.I)
_STOP = {"это", "что", "как", "так", "уже", "был", "была", "было", "есть", "надо",
         "меня", "мне", "тебе", "себя", "очень", "просто", "сейчас", "сегодня"}

_SLOW_SECONDS = 90.0


def stems(text: str) -> set:
    out = set()
    for w in _WORD.findall((text or "").lower().replace("ё", "е")):
        if w in _STOP or (len(w) < 3 and not w.isdigit()):
            continue
        out.add(w if w.isdigit() else w[:4])
    return out


def grounded(entry: str, owner_said: Iterable[str]) -> bool:
    """A diary entry is grounded if it shares a word stem or a number with
    something the owner said in this scenario so far."""
    said = set().union(*[stems(t) for t in owner_said] or [set()])
    return bool(stems(entry) & said)


def _provider_kind(line: str) -> str:
    """'Gemini generateContent: HTTP 429: {…} [model=gemini-3.6-flash]' →
    'Gemini 429 gemini-3.6-flash' — one short, countable label."""
    status = re.search(r"HTTP (\d{3})", line)
    model = re.search(r"model=([\w./-]+)", line)
    head = line.split(":", 1)[0].split()[0] if line else "?"
    return " ".join(x for x in (head, status and status.group(1), model and model.group(1)) if x)


def check(turn: Any, res: Any, owner_said: Sequence[str]) -> List[str]:
    """Violations of one turn as 'rule: detail' strings (empty = clean)."""
    out: List[str] = []
    expect: Dict[str, Any] = getattr(turn, "expect", {}) or {}
    replies: List[str] = list(res.replies)
    text = "\n".join(replies)

    for e in res.errors:
        out.append(f"crash: {e[:160]}")
    for p in getattr(res, "provider", []) or []:
        # Not the reply's fault, still a failure the owner pays for (latency,
        # a weaker fallback model, or no answer at all).
        out.append(f"provider: {_provider_kind(p)}")

    silent_ok = expect.get("silent") is True
    if turn.kind == "owner" and not replies and not silent_ok and not expect.get("silent_ok"):
        out.append("no_reply: the owner got no answer")
    if silent_ok and replies:
        out.append(f"not_silent: expected silence, got {text[:80]!r}")

    for r in replies:
        if any(rx.search(r) for rx in _STUBS):
            out.append(f"stub: {r[:80]!r}")
        for rx in _LEAKS:
            m = rx.search(r)
            if m:
                out.append(f"leak: {m.group(0)!r}")

    for name, args in res.tool_calls:
        if name != "add_diary_entry":
            continue
        entry = str((args or {}).get("text", ""))
        if turn.kind == "scheduled":
            out.append(f"diary_on_ping: {entry[:60]!r} written on a scheduled prompt")
        elif not grounded(entry, owner_said):
            out.append(f"diary_ungrounded: {entry[:60]!r}")

    log: List[str] = list(getattr(res, "log", []) or [])
    if turn.kind == "scheduled" and not replies and not getattr(res, "skipped", ""):
        why = next((line for line in log if "Scheduled job" in line), "no reply")
        out.append(f"ping_dropped: {why[:120]}")
    for line in log:
        if line.startswith(("ERROR", "CRITICAL")):
            out.append(f"log_error: {line[:160]}")
        elif "отклонена" in line or "отклонён" in line:
            # The bot's guard stopped the model: the owner saw nothing wrong,
            # but the model tried to do something it shouldn't.
            out.append(f"guard: {line[:160]}")

    if res.seconds > _SLOW_SECONDS:
        out.append(f"slow: {res.seconds:.0f}s")

    out += _expectations(expect, res, text)
    return out


def _expectations(expect: Dict[str, Any], res: Any, text: str) -> List[str]:
    out: List[str] = []
    tools = [n for n, _a in res.tool_calls]

    agent = expect.get("agent")
    if agent:
        allowed = [agent] if isinstance(agent, str) else list(agent)
        if res.agent and res.agent not in allowed:
            out.append(f"agent: {res.agent} answered, expected {'/'.join(allowed)}")

    for name in expect.get("tools", []):
        if name not in tools:
            out.append(f"tool_missing: {name}")
    for name in expect.get("no_tools", []):
        if name in tools:
            out.append(f"tool_unwanted: {name}")

    if "mute" in expect:
        active = bool(res.mute_after)
        if active != bool(expect["mute"]):
            out.append(f"mute: active={active}, expected {bool(expect['mute'])}")

    for pat in expect.get("reply_has", []):
        if not re.search(pat, text, re.I):
            out.append(f"reply_missing: /{pat}/")
    for pat in expect.get("reply_not", []):
        m = re.search(pat, text, re.I)
        if m:
            out.append(f"reply_forbidden: {m.group(0)!r}")

    limit = expect.get("max_chars")
    if limit and len(text) > int(limit):
        out.append(f"too_long: {len(text)} > {limit} chars")
    return out
