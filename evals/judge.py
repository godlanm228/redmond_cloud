"""A judge model grades one turn of a scenario run.

The rubric is the owner's own criticism, not a generic "helpfulness" score
(Sep 28-29, 2026): don't invent what he didn't say, don't mistake a prompt
written by code for his words, a fact about his whereabouts (hospital) is a
fact and not an emergency, ask when the situation is unclear instead of
assuming, no pep talk, no stubs.

The judge is a different model from the ones that answer (Iris runs on
gemini-3.6-flash, Redmond on Groq): grading your own replies flatters them,
and it would also spend the answering model's rate limit.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("evals.judge")

JUDGE_MODELS: List[str] = ["gemini-3.8-flash", "gemini-2.5-flash"]
KEYS = ("relevance", "facts", "context", "tone")

RUBRIC = """\
You grade replies of a personal Telegram assistant to its only user, Vlad
(Russian-speaking; the assistant is a group of agents: Redmond — general
assistant, Iris — coach/tracker, Newser — news and search, Cipher — server admin).

Score each criterion 0, 1 or 2:
- relevance: answers what Vlad actually said or asked, in this moment. 0 if it
  answers something else, treats a prompt written by code as Vlad's words, or
  ignores his question.
- facts: states nothing that is not in Vlad's words, the known facts, or tool
  results. Inventing what he does ("you're having breakfast"), what the system
  did, or claiming an action that the tool log does not show = 0.
- context: uses the conversation and known facts correctly; doesn't forget what
  was just said, doesn't ask what is already known, doesn't contradict itself.
- tone: proportionate and natural. A fact ("I'm in hospital") is information,
  not an emergency: at most ask politely what happened. Unclear situation →
  one short question, not assumptions. No therapy-speak, no pep talk, no
  lecturing, no stubs; length fits the message.

For a proactive message (the bot writes first, on a schedule) judge whether
it is sensible to send at that time given what Vlad said earlier; Vlad did
not write the prompt, so the message must not address it as his words.

Answer with JSON only:
{"relevance": n, "facts": n, "context": n, "tone": n,
 "issues": ["short concrete problem, quoting the reply", ...],
 "better": "one line: what a good reply would do"}
"""


def build_prompt(turn: Any, res: Any, transcript: Sequence[Dict[str, str]],
                 known_facts: str = "", note: str = "") -> str:
    lines: List[str] = []
    if note:
        lines += ["SCENARIO NOTE (from the owner's review):", note, ""]
    if known_facts:
        lines += ["KNOWN FACTS ABOUT VLAD:", known_facts.strip()[:3000], ""]
    if transcript:
        lines.append("EARLIER IN THIS CONVERSATION:")
        for t in transcript[-8:]:
            who = "Vlad" if t["kind"] == "owner" else "(scheduled prompt by code)"
            lines.append(f"[{t.get('at', '')}] {who}: {t['text'][:500]}")
            if t.get("replies"):
                lines.append(f"  bot: {t['replies'][:600]}")
        lines.append("")
    if turn.kind == "owner":
        lines.append(f"NOW [{turn.at}] Vlad writes: {turn.text}")
    else:
        lines.append(f"NOW [{turn.at}] scheduled prompt written by code (Vlad did not "
                     f"write this): {turn.text[:800]}")
    who = res.agent or "nobody"
    lines.append(f"REPLY by {who}: " + (" / ".join(res.replies) if res.replies else "(no reply)"))
    tools = ", ".join(f"{n}({json.dumps(a, ensure_ascii=False)[:120]})" for n, a in res.tool_calls)
    lines.append(f"TOOL LOG: {tools or 'no tools called'}")
    return "\n".join(lines)


def parse(text: str) -> Optional[Dict[str, Any]]:
    """The judge's JSON, tolerant to code fences and prose around it."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    scores = {}
    for k in KEYS:
        v = data.get(k)
        if isinstance(v, (int, float)) and 0 <= v <= 2:
            scores[k] = int(v)
    if len(scores) != len(KEYS):
        return None
    return {"scores": scores,
            "issues": [str(i) for i in data.get("issues", []) if i][:6],
            "better": str(data.get("better", ""))[:300]}


def verdict(scores: Dict[str, int]) -> str:
    """fail on any zero and on anything but a clean facts score: an invented
    fact is the one thing the owner called unacceptable outright."""
    if any(v == 0 for v in scores.values()) or scores.get("facts", 0) < 2:
        return "fail"
    return "pass" if all(v == 2 for v in scores.values()) else "weak"


def _call(model: str, prompt: str) -> str:
    from utils import gemini
    thinking = ({"thinkingLevel": "medium"} if model.startswith("gemini-3")
                else {"thinkingBudget": 1024})
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": RUBRIC}]},
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096,
                             "thinkingConfig": thinking,
                             "responseMimeType": "application/json"},
    }
    key = gemini.api_key_from_env()
    if not key:
        return ""
    return gemini.extract_text(gemini._post_generate(key, body, model, timeout=90.0))


def grade(turn: Any, res: Any, transcript: Sequence[Dict[str, str]],
          known_facts: str = "", note: str = "",
          models: Sequence[str] = JUDGE_MODELS, call=_call) -> Dict[str, Any]:
    prompt = build_prompt(turn, res, transcript, known_facts, note)
    for model in models:
        parsed = parse(call(model, prompt))
        if parsed:
            parsed["model"] = model
            parsed["verdict"] = verdict(parsed["scores"])
            return parsed
        logger.warning("judge %s gave no usable answer", model)
    return {"verdict": "unjudged", "error": "no judge model answered"}
