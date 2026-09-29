"""Understanding: one structured reading of each owner message, checked by code.

Why. Three classifiers used to decide what the owner meant, and all three
were wrong in the same way - by guessing:
  • the router (keywords, then a small model) chose the agent; on Sep 29, 2026
    it answered the real «Суисайд» of Sep 4 with silence, and sent «курю кальян
    с Настей, чилю» to Redmond, who retold it to Iris with invented details;
  • the distress detector (regular expressions) treated «в больнице лежу» as an
    emergency - a fact about where he is, the owner said, not a crisis;
  • nobody decided what counts as a fact about him, so the answering model
    wrote its own guesses to the diary: «Позавтракал» on a mute request,
    «Довёл UI до презентабельного вида» on a plan to do it.

Here one model call reads the message in its conversation and returns: who it
is for, what it refers to, the facts about his life it contains, the commands
it gives, and how urgent it is. Every fact, command and urgency must carry a
verbatim quote from the message, and code checks that the quote is really
there: what he did not write cannot become a fact. Code then acts on it -
records the facts, executes the commands - and the answering agent is told
what was understood and done instead of guessing again.

If no model can read the message, `understand()` returns None and the old
path (keywords, distress detector) takes over: an answer must not depend on
the understanding step being available.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

AGENTS = ("Iris", "Redmond", "Newser", "Cipher")
NOBODY = "Никто"
WHEN = ("done", "now", "plan", "habit")
URGENCY = ("none", "attention", "crisis")
COMMANDS = ("mute", "unmute")

PROMPT = """\
Ты — шаг понимания в личном Telegram-хабе Влада. Четыре агента:
- Iris — коуч и трекер: дневник, еда, сон, спорт, учёба, работа, цели, дедлайны, настроение, самочувствие, тишина уведомлений.
- Redmond — общий ассистент: вопросы, объяснения, советы, всё остальное.
- Newser — новости и поиск свежей информации в интернете.
- Cipher — администратор сервера бота (код, сбои, деплой). Сообщение, где о Cipher или о подписке Claude только ГОВОРЯТ, адресовано не ему.

Прочитай НОВОЕ сообщение Влада в контексте разговора и верни ТОЛЬКО JSON:
{
 "addressee": "Iris" | "Redmond" | "Newser" | "Cipher" | "Никто",
 "research": true/false,
 "about": "о чём сообщение, одной фразой",
 "refers_to": "если это ответ или вопрос к предыдущей реплике — к какой и о чём; иначе пусто",
 "facts": [{"quote": "...", "fact": "...", "when": "done|now|plan|habit", "topic": "..."}],
 "commands": [{"quote": "...", "type": "mute|unmute", "until": "", "hours": null, "scope": "all|pings"}],
 "urgency": "none|attention|crisis",
 "urgency_quote": ""
}

Правила:
- quote — ДОСЛОВНЫЙ фрагмент нового сообщения, скопированный символ в символ. Код проверит, что он там есть. Не перефразируй в quote.
- facts — только то, что Влад сам сообщил о своей жизни: что сделал, делает, где находится, как себя чувствует, что съел, что планирует, чем занят. fact — коротко, от третьего лица, по-русски, НЕ добавляя ничего сверх сказанного. Не факты: вопросы, просьбы, команды, мнения о боте, пересказ прошлых реплик бота.
- when: done — уже сделал/случилось; now — происходит сейчас; plan — намерение, план, «хочу/буду/надо/довести»; habit — регулярно.
- Нет фактов — пустой список. Никогда не додумывай факт, которого нет в тексте (еду, сон, занятия).
- commands — только явные просьбы выключить/включить уведомления бота («мут на 7 дней», «не пиши до понедельника», «можешь писать»). until — дата окончания, если названа, в виде YYYY-MM-DD или YYYY-MM-DDTHH:MM по его времени; hours — если назван срок в часах/днях (дни × 24). scope: all — если просит полную тишину («фул мут», «ничего не присылай»), иначе pings.
- urgency: crisis — только явная угроза жизни или острая ситуация прямо сейчас (суицид, «не могу дышать», «вызови скорую», сильная боль сейчас). attention — серьёзное, но не острое: больница, болезнь, обследования, плохие новости. none — всё остальное. Нахождение в больнице само по себе — attention, не crisis. Для attention и crisis urgency_quote — дословный фрагмент.
- addressee: если агент назван в начале или обращение к нему — он. Иначе — чья это зона по смыслу. Ответ на реплику агента или жалоба на его ответ — к тому агенту. Сообщение о тяжёлом состоянии, боли, больнице — Iris. «Никто» — только если Влад явно думает вслух и не ждёт ответа; вопрос, просьба или рассказ о себе — всегда агенту.
- research: true — если нужен свежий веб-поиск по нескольким источникам (новости, рынки, обзоры); одиночный факт — false.
"""


@dataclass
class Fact:
    quote: str
    fact: str
    when: str = "now"
    topic: str = ""


@dataclass
class Command:
    quote: str
    type: str
    until: str = ""
    hours: Optional[float] = None
    scope: str = "pings"


@dataclass
class Understanding:
    addressee: str
    research: bool = False
    about: str = ""
    refers_to: str = ""
    facts: List[Fact] = field(default_factory=list)
    commands: List[Command] = field(default_factory=list)
    urgency: str = "none"
    urgency_quote: str = ""
    model: str = ""
    dropped: List[str] = field(default_factory=list)  # what code refused: unquoted claims
    # Set once code has acted on it (facts recorded, commands executed): a
    # handoff to Iris carries the same Understanding and must not act twice.
    applied: bool = False
    done: List[tuple] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Quote check
# ---------------------------------------------------------------------------

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, ё→е, punctuation and line breaks as single spaces."""
    return " ".join(_NON_WORD.sub(" ", (text or "").lower().replace("ё", "е")).split())


def quoted(quote: str, message: str) -> bool:
    """The quote really is in the message (after normalisation)."""
    q = normalize(quote)
    return bool(q) and f" {q} " in f" {normalize(message)} "


# ---------------------------------------------------------------------------
# Reading the model's answer
# ---------------------------------------------------------------------------

def parse(raw: str, message: str) -> Optional[Understanding]:
    """The model's JSON → Understanding with every unquoted claim dropped.
    None if there is no usable JSON at all."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    addressee = str(data.get("addressee") or "").strip()
    if addressee.lower() == NOBODY.lower():
        addressee = NOBODY
    elif addressee not in AGENTS:
        addressee = ""
    u = Understanding(addressee=addressee, research=bool(data.get("research")),
                      about=str(data.get("about") or "")[:300],
                      refers_to=str(data.get("refers_to") or "")[:300])

    for f in data.get("facts") or []:
        if not isinstance(f, dict):
            continue
        quote, text = str(f.get("quote") or ""), str(f.get("fact") or "").strip()
        if not text or not quoted(quote, message):
            u.dropped.append(f"fact without its quote: {text[:80]!r}")
            continue
        when = str(f.get("when") or "now").lower()
        u.facts.append(Fact(quote=quote, fact=text[:300], when=when if when in WHEN else "now",
                            topic=str(f.get("topic") or "")[:40]))

    for c in data.get("commands") or []:
        if not isinstance(c, dict):
            continue
        kind = str(c.get("type") or "").lower()
        quote = str(c.get("quote") or "")
        if kind not in COMMANDS or not quoted(quote, message):
            u.dropped.append(f"command without its quote: {kind} {quote[:60]!r}")
            continue
        hours = c.get("hours")
        try:
            hours = float(hours) if hours not in (None, "") else None
        except (TypeError, ValueError):
            hours = None
        scope = "all" if str(c.get("scope") or "").lower() == "all" else "pings"
        u.commands.append(Command(quote=quote, type=kind, until=str(c.get("until") or "")[:16],
                                  hours=hours, scope=scope))

    urgency = str(data.get("urgency") or "none").lower()
    u.urgency_quote = str(data.get("urgency_quote") or "")
    if urgency not in URGENCY:
        urgency = "none"
    if urgency != "none" and not quoted(u.urgency_quote, message):
        u.dropped.append(f"urgency {urgency} without its quote")
        urgency = "attention" if urgency == "crisis" else "none"
    u.urgency = urgency
    return u


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def build_input(message: str, history: Sequence[Dict[str, str]], now: str,
                state: str = "") -> str:
    """history: [{"who": "Влад"|agent|"плановое сообщение", "text": ...}], oldest first."""
    lines = [f"Сейчас: {now}."]
    if state:
        lines.append(f"Состояние: {state}")
    if history:
        lines.append("Разговор до этого (старые сверху):")
        for h in history[-8:]:
            lines.append(f"- {h.get('who', '?')}: {' '.join(str(h.get('text', '')).split())[:400]}")
    lines.append(f"НОВОЕ сообщение Влада: «{message}»")
    return "\n".join(lines)


def understand(message: str, history: Sequence[Dict[str, str]] = (), now: str = "",
               state: str = "", complete=None) -> Optional[Understanding]:
    """Read `message`. None if no model of the "understand" pool answered with
    usable JSON - the caller falls back to the old path."""
    from utils import llm, llm_gate
    if not (message or "").strip():
        return None
    if not now:
        from utils.time import now_local
        n = now_local()
        now = n.strftime("%Y-%m-%d %H:%M, ") + ["понедельник", "вторник", "среда", "четверг",
                                                 "пятница", "суббота", "воскресенье"][n.weekday()]
    prompt = build_input(message, history, now, state)
    tokens = llm_gate.estimate_tokens(PROMPT, prompt)
    call = complete or llm.complete
    tried: List[str] = []
    while True:
        model = llm_gate.acquire([m for m in llm_gate.pool("understand") if m not in tried],
                                 tokens, llm_gate.OWNER, max_wait=15.0)
        if model is None:
            if tried:
                logger.warning("Понимание: ни одна модель не дала разбор (%s)", ", ".join(tried))
            return None
        tried.append(model)
        raw = call(model, prompt, system=PROMPT, max_tokens=700, temperature=0.0)
        u = parse(raw, message)
        if u is not None and u.addressee:
            u.model = model
            if u.dropped:
                logger.warning("Понимание: отброшено без цитаты — %s", "; ".join(u.dropped))
            return u
        logger.warning("Понимание: %s дала непригодный ответ %r", model, (raw or "")[:120])


# ---------------------------------------------------------------------------
# Acting on it (code, not the answering model)
# ---------------------------------------------------------------------------

WHEN_TAG = {"done": "сделано", "now": "сейчас", "plan": "план", "habit": "привычка"}


def _diary_text(quote: str) -> str:
    text = " ".join((quote or "").split()).strip(" ,.;")
    return text[:1].upper() + text[1:] if text else ""


def apply(u: "Understanding", execute=None) -> List[tuple]:
    """Carry out what the message says: its commands, and its facts into the
    diary in the owner's own words (the quote), tagged plan / done / now.

    The diary gets his words, not a retelling: 29.09.2026 his plan «довести до
    ума…» was written by the model as «Довёл UI до презентабельного вида».
    Returns (tool, args, result) for the receipt under the answer."""
    from logic.tools import execute_tool
    run = execute or execute_tool
    done: List[tuple] = []

    for c in u.commands:
        if c.type == "unmute":
            args = {"mode": "off"}
        else:
            args = {"scope": c.scope}
            if c.until:
                args["until"] = c.until
            elif c.hours:
                args["hours"] = c.hours
        done.append(("mute_notifications", args, run("mute_notifications", args)))

    recent = set()
    try:
        from logic import coach_storage
        recent = {normalize(e.get("text", "")) for e in coach_storage.read_diary(last_n=15)}
    except Exception:  # noqa: BLE001 — without the check a repeat is written twice, not lost
        logger.debug("diary read for repeats failed", exc_info=True)
    for f in u.facts:
        text = _diary_text(f.quote)
        if not text or normalize(text) in recent:
            continue
        tags = [t for t in (WHEN_TAG.get(f.when, ""), f.topic.strip().lower()) if t]
        args = {"text": text, "tags": tags}
        done.append(("add_diary_entry", args, run("add_diary_entry", args)))
        recent.add(normalize(text))
    return done


ATTENTION = (
    "SERIOUS BUT NOT ACUTE. The owner mentioned something serious about himself "
    "(hospital, illness, an examination, bad news). Take it as a fact, not an alarm: "
    "acknowledge it plainly and, if it fits, ask politely what happened or how he is. "
    "No drama, no advice lists, no hotlines, no guessing the cause. Anything else he "
    "asked for in the same message - do it."
)


def prompt_block(u: "Understanding", done: Sequence[tuple] = ()) -> str:
    """What the answering agent is told about the message: what code read and did."""
    lines = ["[Как код понял сообщение владельца]"]
    if u.about:
        lines.append(f"  о чём: {u.about}")
    if u.refers_to:
        lines.append(f"  относится к: {u.refers_to}")
    for f in u.facts:
        lines.append(f"  факт ({WHEN_TAG.get(f.when, f.when)}): {f.fact}")
    for name, _args, result in done:
        lines.append(f"  сделано кодом ({name}): {' '.join(str(result).split())[:200]}")
    lines.append("  Правила: факты из этого сообщения код уже записал в дневник — сам их не "
                 "пиши. Сроки, даты и номера записей бери только из строк «сделано кодом», "
                 "своими словами не пересказывай. План — это план, а не сделанное.")
    return "\n".join(lines)
