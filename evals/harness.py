"""Replay real conversations through the production pipeline and grade the replies.

Why this exists. On Sep 29, 2026 the hub had 570 unit tests and still answered
the owner badly. The tests checked mechanics against invented model responses;
no test ever ran a real model on a real conversation. All four failures of
that day lived exactly there: a scheduled prompt read as the owner's words, a
failed turn that lost his message, invented diary entries, a word classifier
treating "I'm in hospital" as an emergency.

A scenario is a conversation as it happened: owner messages and scheduled
prompts, each at its original time. Every owner message goes to all four
agents' handlers as a Telegram update, as in production (they run as four
Applications on one update stream); scheduled prompts go through the
scheduler's own send path. Real models answer. Replies are recorded instead
of sent, Cipher is never called (it would spend the owner's Claude
subscription and act on the real server).

Isolation. Everything runs in a temporary directory with copies of config/,
the dossier and the database; the bot's relative paths resolve there. Each
scenario starts from a fresh copy cut to the moment the conversation began:
memory, diary and chat history after that moment are deleted, otherwise the
bot would find its own original replies through recall.

Each turn is graded twice: rules (evals/checks) and a judge model
(evals/judge). Scenarios with real messages are private: they live in
data/evals/ (gitignored) and may point to memory rows by id instead of
quoting the text.

    ./venv/bin/python -m evals.harness --scenarios data/evals/scenarios.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("evals")


@dataclass
class Turn:
    kind: str                      # "owner" | "scheduled"
    text: str
    at: str                        # local time the turn happened, ISO
    expect: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    turns: List[Turn]
    note: str = ""
    setup: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    scenario: str
    index: int
    kind: str
    text: str
    at: str
    agent: str = ""
    replies: List[str] = field(default_factory=list)
    tool_calls: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    diary_added: List[str] = field(default_factory=list)
    mute_after: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    skipped: str = ""
    seconds: float = 0.0
    violations: List[str] = field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def load_scenarios(path: Path, source_db: Path) -> List[Scenario]:
    """A turn carries text and at, or a memory_id: then the owner's words and
    the time come from the database, and the scenario file holds no private text.

    A memory row stores the owner's message as `user`; scheduled prompts were
    stored the same way, and start with "(" like every prompt written by code.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        out = []
        for sc in raw:
            turns = []
            for t in sc["turns"]:
                if "memory_id" in t:
                    row = conn.execute("SELECT user, timestamp FROM memory WHERE id=?",
                                       (t["memory_id"],)).fetchone()
                    if row is None:
                        raise ValueError(f"{sc['name']}: memory #{t['memory_id']} not found")
                    text = row[0]
                    at = t.get("at") or _local(datetime.fromtimestamp(row[1], timezone.utc)
                                               ).isoformat(timespec="seconds")
                else:
                    text, at = t["text"], t["at"]
                kind = t.get("kind") or ("scheduled" if text.lstrip().startswith("(") else "owner")
                turns.append(Turn(kind=kind, text=text, at=at, expect=t.get("expect", {})))
            out.append(Scenario(name=sc["name"], turns=turns, note=sc.get("note", ""),
                                setup=sc.get("setup", {})))
        return out
    finally:
        conn.close()


def _local(dt: datetime) -> datetime:
    from utils.time import OWNER_TZ
    if OWNER_TZ is None:
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=OWNER_TZ)
    return dt.astimezone(OWNER_TZ)


def _parse_ts(value: Any) -> Optional[datetime]:
    """Timestamps in the database come in three shapes: local with offset
    (diary, db.history_add), naive UTC (response_generator on the VM, whose
    clock is UTC) and plain dates (goals). None if unreadable."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    s = str(value).strip()
    try:
        if len(s) == 10:
            d = date.fromisoformat(s)
            return _local(datetime(d.year, d.month, d.day))
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

class Sandbox:
    """A temporary working directory with copies of everything the bot reads
    or writes by relative path: config/, data/owner_dossier.md, the database."""

    def __init__(self, hub_dir: Path):
        self.hub = Path(hub_dir).resolve()
        self.dir = Path(tempfile.mkdtemp(prefix="redmond-eval-"))
        self.pristine = self.dir / "_pristine"
        self.pristine.mkdir()
        shutil.copytree(self.hub / "config", self.dir / "config")
        (self.dir / "data").mkdir()
        (self.dir / "logs").mkdir()
        dossier = self.hub / "data" / "owner_dossier.md"
        if dossier.exists():
            shutil.copy2(dossier, self.dir / "data" / "owner_dossier.md")
        profile = self.dir / "config" / "owner_profile.json"
        if profile.exists():
            shutil.copy2(profile, self.pristine / "owner_profile.json")
        # backup API: a consistent snapshot even while the live bot writes (WAL).
        src = sqlite3.connect(f"file:{self.hub / 'data' / 'memory.sqlite'}?mode=ro", uri=True)
        dst = sqlite3.connect(self.pristine / "memory.sqlite")
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        self.db = self.dir / "data" / "memory.sqlite"

    def enter(self) -> None:
        os.chdir(self.dir)

    def fresh(self, start: datetime, kv: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """Working copies for one scenario, cut to its first moment."""
        from utils import db
        db.close_all()
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(f"{self.db}{suffix}")
        shutil.copy2(self.pristine / "memory.sqlite", self.db)
        if (self.pristine / "owner_profile.json").exists():
            shutil.copy2(self.pristine / "owner_profile.json",
                         self.dir / "config" / "owner_profile.json")
        return cut_to(self.db, start, kv or {})

    def cleanup(self) -> None:
        with contextlib.suppress(Exception):
            os.chdir(self.hub)
        shutil.rmtree(self.dir, ignore_errors=True)


# State that describes "today" and would leak the real day into the replayed one.
_DAY_KEYS = ("mute", "presence", "day_state")


def cut_to(path: Path, start: datetime, kv: Dict[str, Any]) -> Dict[str, int]:
    """Delete what happened at or after `start`; reopen what was closed after it.
    Returns rows removed per table (for the report and tests)."""
    start = start if start.tzinfo else _local(start)
    removed: Dict[str, int] = {}
    conn = sqlite3.connect(path)
    try:
        def gone(table: str, col: str = "ts") -> None:
            try:
                rows = conn.execute(f"SELECT id, {col} FROM {table}").fetchall()
            except sqlite3.OperationalError:
                return  # older schema without this table
            ids = [(rid,) for rid, ts in rows if (_parse_ts(ts) or start) >= start]
            conn.executemany(f"DELETE FROM {table} WHERE id=?", ids)
            removed[table] = len(ids)

        # memory: epoch seconds; FTS follows through triggers.
        cur = conn.execute("DELETE FROM memory WHERE timestamp >= ?", (start.timestamp(),))
        removed["memory"] = cur.rowcount
        for table in ("diary", "chat_history", "shift_events", "vision_results"):
            gone(table)
        day = start.date().isoformat()
        for table in ("goals", "deadlines"):
            try:
                rows = conn.execute(f"SELECT id, created FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue
            ids = [(rid,) for rid, created in rows if str(created)[:10] > day]
            conn.executemany(f"DELETE FROM {table} WHERE id=?", ids)
            removed[table] = len(ids)
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("UPDATE goals SET status='active', closed=NULL "
                         "WHERE closed IS NOT NULL AND substr(closed,1,10) > ?", (day,))
            conn.execute("UPDATE deadlines SET status='pending', closed=NULL "
                         "WHERE closed IS NOT NULL AND substr(closed,1,10) > ?", (day,))
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("DELETE FROM embeddings WHERE kind='memory' AND CAST(ref AS INTEGER) "
                         "NOT IN (SELECT id FROM memory)")
            conn.execute("DELETE FROM embeddings WHERE kind='diary' AND CAST(ref AS INTEGER) "
                         "NOT IN (SELECT id FROM diary)")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.executemany("DELETE FROM kv WHERE key=?", [(k,) for k in _DAY_KEYS])
            for key, value in kv.items():
                conn.execute("INSERT OR REPLACE INTO kv(key, value, updated) VALUES(?,?,?)",
                             (key, json.dumps(value, ensure_ascii=False),
                              start.isoformat(timespec="minutes")))
        conn.commit()
    finally:
        conn.close()
    return removed


# ---------------------------------------------------------------------------
# Telegram stand-ins
# ---------------------------------------------------------------------------

class RecordingCoordinator:
    """Stands in for core.coordinator.Coordinator: records instead of sending."""

    def __init__(self):
        self.sent: List[Dict[str, str]] = []

    def bot_for(self, agent_name: str):
        return None  # no live status messages

    @contextlib.asynccontextmanager
    async def typing(self, agent_name: str, chat_id: int):
        yield

    async def respond_as(self, agent_name, chat_id, text, emoji="", output_format="plain"):
        self.sent.append({"agent": agent_name, "text": str(text)})
        return [len(self.sent)]


def owner_ids() -> Tuple[int, int]:
    """(chat, user) the gate accepts: the real ones from .env."""
    from handlers import multi_bot
    chat = multi_bot._main_chat_id()
    users = sorted(multi_bot._allowed_user_ids())
    if chat is None or not users:
        raise RuntimeError("MAIN_CHAT_ID / ALLOWED_USER_IDS are not set (.env not loaded?)")
    return chat, users[0]


def fake_update(text: str, chat_id: int, user_id: int, n: int) -> Any:
    user = SimpleNamespace(id=user_id, is_bot=False, first_name="Влад", username="owner")
    chat = SimpleNamespace(id=chat_id, type="supergroup")

    async def reply_text(*_a, **_kw):
        return None

    message = SimpleNamespace(text=text, message_id=n, reply_to_message=None, voice=None,
                              photo=None, from_user=user, chat=chat, reply_text=reply_text)
    return SimpleNamespace(effective_user=user, effective_chat=chat, message=message)


class LogTap(logging.Handler):
    """Warnings of the bot during a turn. A scheduled reply that failed is
    dropped silently by design (the owner didn't ask for it); the log is the
    only trace, so the run has to read it."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("evals"):
            return
        with contextlib.suppress(Exception):
            self.lines.append(f"{record.levelname} {record.name}: {record.getMessage()[:240]}")


def app_context(agent: Any, shared: Dict[str, Any]) -> Any:
    bot_data = dict(shared)
    bot_data["agent"] = agent
    return SimpleNamespace(application=SimpleNamespace(bot_data=bot_data))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, sandbox: Sandbox, pause: float = 30.0, use_judge: bool = True):
        self.sandbox = sandbox
        self.pause = pause
        self.use_judge = use_judge
        self.chat_id, self.user_id = owner_ids()
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.cut: Dict[str, Dict[str, int]] = {}
        self.dispatcher = None
        self._n = 1000
        self.tap = LogTap()
        logging.getLogger().addHandler(self.tap)
        self._patch()

    def _patch(self) -> None:
        from handlers import multi_bot
        from logic import tools

        original = tools.execute_tool
        calls = self.calls

        def recording(name, args, rg=None, session=None):
            calls.append((name, dict(args or {})))
            return original(name, args, rg=rg, session=session)

        tools.execute_tool = recording

        async def no_cipher(user_text, context, chat_id, meta=None):
            return "(Cipher в прогоне не вызывается)"

        multi_bot._generate_cipher = no_cipher

    def start(self, sc: Scenario) -> None:
        """Fresh data and a fresh bot for a scenario (history, sticky, caches)."""
        from config.config_loader import load_app_config
        from core.dispatcher import Dispatcher
        from utils import db

        if self.dispatcher is not None:
            with contextlib.suppress(Exception):
                self.dispatcher.response_generator.mem.conn.close()
        first = _local(datetime.fromisoformat(sc.turns[0].at))
        self.cut[sc.name] = self.sandbox.fresh(first, sc.setup.get("kv"))
        db.set_db_path(self.sandbox.db)
        config = load_app_config(self.sandbox.dir / "config" / "config.json")
        self.dispatcher = Dispatcher(config)
        self.coordinator = RecordingCoordinator()
        self.shared = {"dispatcher": self.dispatcher, "coordinator": self.coordinator,
                       "router_states": {}}

    async def run_turn(self, sc: Scenario, i: int, turn: Turn) -> TurnResult:
        from core import scheduler
        from logic import coach_storage
        from utils import db, failures
        from utils.time import set_clock

        at = _local(datetime.fromisoformat(turn.at))
        started = time.time()
        set_clock(lambda: at + timedelta(seconds=time.time() - started))
        res = TurnResult(scenario=sc.name, index=i, kind=turn.kind, text=turn.text, at=turn.at)
        diary_before = {r["id"] for r in db.query("SELECT id FROM diary")}
        sent_before = len(self.coordinator.sent)
        log_before = len(self.tap.lines)
        self.calls.clear()
        try:
            if turn.kind == "scheduled":
                if coach_storage.muted_now():
                    res.skipped = "muted"  # production jobs check this before sending
                else:
                    await scheduler._generate_and_send(
                        self.dispatcher, self.coordinator, self.chat_id,
                        scheduler.IRIS_TICKER, turn.text, self.shared["router_states"])
            else:
                await self._owner_message(turn.text)
        except Exception as e:  # noqa: BLE001 — a crash is a finding, not the end of the run
            logger.exception("turn crashed")
            res.errors.append(f"crash: {e.__class__.__name__}: {e}")
        finally:
            set_clock(None)

        res.seconds = round(time.time() - started, 1)
        new = self.coordinator.sent[sent_before:]
        res.replies = [m["text"] for m in new]
        res.agent = ", ".join(dict.fromkeys(m["agent"] for m in new))
        res.tool_calls = list(self.calls)
        res.diary_added = [r["text"] for r in db.query("SELECT id, text FROM diary ORDER BY id")
                           if r["id"] not in diary_before]
        res.mute_after = coach_storage.mute_info()
        res.errors += [f"{w}: {t}" for ts, w, t in failures.recent(hours=1, limit=20)
                       if ts >= started]
        res.log = self.tap.lines[log_before:]
        return res

    async def _owner_message(self, text: str) -> None:
        """Like production: every Application receives the update; each
        handler decides whether the message is its own."""
        from handlers import multi_bot
        from logic.agents import AGENTS

        self._n += 1
        update = fake_update(text, self.chat_id, self.user_id, self._n)
        jobs = []
        for agent in AGENTS:
            handler = (multi_bot.redmond_handler if agent.name == "Redmond"
                       else multi_bot.slim_agent_handler)
            jobs.append(handler(update, app_context(agent, self.shared)))
        await asyncio.gather(*jobs)

    def close(self) -> None:
        logging.getLogger().removeHandler(self.tap)

    def known_facts(self) -> str:
        rg = self.dispatcher.response_generator
        with contextlib.suppress(Exception):
            return rg._compact_owner_facts()
        return ""

    async def run(self, scenarios: List[Scenario]) -> List[TurnResult]:
        from evals import checks, judge

        results: List[TurnResult] = []
        for sc in scenarios:
            self.start(sc)
            owner_said: List[str] = []
            transcript: List[Dict[str, str]] = []
            for i, turn in enumerate(sc.turns):
                if turn.kind == "owner":
                    owner_said.append(turn.text)
                res = await self.run_turn(sc, i, turn)
                res.violations = checks.check(turn, res, owner_said)
                if self.use_judge and not res.skipped and (res.replies or turn.kind == "owner"):
                    res.judge = await asyncio.to_thread(
                        judge.grade, turn, res, transcript, self.known_facts(), sc.note)
                transcript.append({"kind": turn.kind, "at": turn.at, "text": turn.text,
                                   "replies": " / ".join(res.replies)})
                results.append(res)
                logger.info("%s #%d %s → %s | %s | judge=%s", sc.name, i, turn.kind,
                            res.agent or "—", "; ".join(res.violations) or "ok",
                            (res.judge or {}).get("verdict", "—"))
                if self.pause:
                    await asyncio.sleep(self.pause)
        return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def summarize(results: List[TurnResult]) -> Dict[str, Any]:
    by_rule: Dict[str, int] = {}
    for r in results:
        for v in r.violations:
            rule = v.split(":", 1)[0]
            by_rule[rule] = by_rule.get(rule, 0) + 1
    judged = [r.judge for r in results if r.judge and "scores" in r.judge]
    avg = {}
    for key in ("relevance", "facts", "context", "tone"):
        vals = [j["scores"][key] for j in judged]
        if vals:
            avg[key] = round(sum(vals) / len(vals), 2)
    verdicts: Dict[str, int] = {}
    for j in judged:
        verdicts[j["verdict"]] = verdicts.get(j["verdict"], 0) + 1
    graded = [r for r in results if not r.skipped]
    clean = sum(1 for r in graded if not r.violations
                and (r.judge or {}).get("verdict", "pass") == "pass")
    return {"turns": len(results), "graded": len(graded), "clean": clean,
            "violations_by_rule": by_rule, "judge_verdicts": verdicts,
            "judge_avg": avg, "judged": len(judged)}


def render_markdown(results: List[TurnResult], summary: Dict[str, Any]) -> str:
    lines = ["# Scenario run", "", "```", json.dumps(summary, ensure_ascii=False, indent=1),
             "```", ""]
    current = None
    for r in results:
        if r.scenario != current:
            current = r.scenario
            lines += ["", f"## {current}", ""]
        who = "Влад" if r.kind == "owner" else "пинг"
        lines.append(f"**#{r.index} [{r.at}] {who}:** {r.text[:300]}")
        if r.skipped:
            lines.append(f"- пропущено: {r.skipped}")
        for reply in r.replies:
            lines.append(f"- **{r.agent}:** {reply[:1200]}")
        if r.tool_calls:
            lines.append("- tools: " + ", ".join(n for n, _a in r.tool_calls))
        if r.violations:
            lines.append("- ⚠ " + "; ".join(r.violations))
        if r.judge:
            j = r.judge
            lines.append(f"- судья ({j.get('model', '—')}): {j.get('verdict')} {j.get('scores', {})}")
            for issue in j.get("issues", []):
                lines.append(f"  - {issue}")
            if j.get("better"):
                lines.append(f"  - лучше: {j['better']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Replay real conversations and grade the replies.")
    ap.add_argument("--scenarios", default="data/evals/scenarios.json")
    ap.add_argument("--out", default="data/evals/results")
    ap.add_argument("--pause", type=float, default=30.0,
                    help="seconds between turns (Groq: 8K tokens/min per model)")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    args = ap.parse_args(argv)

    hub = Path.cwd()
    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv
        load_dotenv(hub / ".env")
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    scenarios = load_scenarios(hub / args.scenarios, hub / "data" / "memory.sqlite")
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.name in wanted]
    out_dir = hub / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    sandbox = Sandbox(hub)
    sandbox.enter()
    try:
        runner = Runner(sandbox, pause=args.pause, use_judge=not args.no_judge)
        try:
            results = asyncio.run(runner.run(scenarios))
        finally:
            runner.close()
    finally:
        sandbox.cleanup()

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    summary = summarize(results)
    (out_dir / f"run-{stamp}.json").write_text(json.dumps(
        {"summary": summary, "cut": runner.cut, "results": [asdict(r) for r in results]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"run-{stamp}.md").write_text(render_markdown(results, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"report: {out_dir / f'run-{stamp}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
