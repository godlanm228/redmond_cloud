"""The scenario runner itself (evals/): isolation, time travel, plumbing, grading.

The runner exists to measure the bot on real conversations with real models;
these tests only make sure the measurement is honest: that a run cannot touch
production data, that the bot sees the database as it was when the
conversation began (not its own original replies), that messages reach the
same handlers as in production, and that the rules and the judge read the
result correctly. Models are faked here on purpose — the real ones are the
subject of the run, not of the unit tests.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import checks, harness, judge
from utils.time import OWNER_TZ


def _t(s):
    return datetime.fromisoformat(s).replace(tzinfo=OWNER_TZ)


def _make_db(path: Path):
    """A database with the production schema and rows on both sides of 14:00."""
    from utils import db
    from utils.memory import MemoryStore
    db.set_db_path(path)
    db.connect()
    store = MemoryStore(str(path), vector_search=False)
    before, after = _t("2026-09-29T13:30").timestamp(), _t("2026-09-29T14:30").timestamp()
    store.conn.execute("INSERT INTO memory(user, bot, timestamp) VALUES('завтрак был', 'ок', ?)", (before,))
    store.conn.execute("INSERT INTO memory(user, bot, timestamp) VALUES('я в больнице', 'ок', ?)", (after,))
    store.conn.commit()
    store.conn.close()
    db.execute("INSERT INTO diary(ts, text) VALUES('2026-09-29T13:59+02:00', 'до')")
    db.execute("INSERT INTO diary(ts, text) VALUES('2026-09-29T14:00+02:00', 'после')")
    # naive UTC: 11:30 UTC = 13:30 local (before), 12:30 UTC = 14:30 local (after)
    db.execute("INSERT INTO chat_history(chat_id, ts, user, bot) VALUES(1, '2026-09-29T11:30:00', 'a', 'b')")
    db.execute("INSERT INTO chat_history(chat_id, ts, user, bot) VALUES(1, '2026-09-29T12:30:00', 'c', 'd')")
    db.execute("INSERT INTO goals(id, title, status, created, closed) VALUES(1, 'old', 'done', '2026-09-01', '2026-09-30')")
    db.execute("INSERT INTO goals(id, title, status, created) VALUES(2, 'new', 'active', '2026-09-30')")
    db.execute("INSERT INTO embeddings(kind, ref, model, hash, dim, vec) VALUES('memory', '2', 'm', 'h', 1, x'00')")
    db.kv_set("mute", {"scope": "all", "until": "2026-10-01T16:45"})
    db.close_all()


def test_cut_removes_the_future_in_every_timestamp_format(tmp_path):
    path = tmp_path / "m.sqlite"
    _make_db(path)
    removed = harness.cut_to(path, _t("2026-09-29T14:00"), {})
    conn = sqlite3.connect(path)
    assert [r[0] for r in conn.execute("SELECT user FROM memory")] == ["завтрак был"]
    assert [r[0] for r in conn.execute("SELECT text FROM diary")] == ["до"]
    assert [r[0] for r in conn.execute("SELECT user FROM chat_history")] == ["a"]
    assert [r[0] for r in conn.execute("SELECT title FROM goals")] == ["old"]
    assert conn.execute("SELECT status, closed FROM goals").fetchone() == ("active", None)
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM kv WHERE key='mute'").fetchone()[0] == 0
    # full-text search must not find the future either: recall goes through it
    assert conn.execute("SELECT COUNT(*) FROM memory_fts WHERE memory_fts MATCH 'больнице'").fetchone()[0] == 0
    assert removed["memory"] == 1 and removed["chat_history"] == 1


def test_cut_applies_the_scenario_state(tmp_path):
    path = tmp_path / "m.sqlite"
    _make_db(path)
    harness.cut_to(path, _t("2026-09-29T14:00"), {"mute": {"scope": "pings", "until": "x"}})
    conn = sqlite3.connect(path)
    assert json.loads(conn.execute("SELECT value FROM kv WHERE key='mute'").fetchone()[0])["scope"] == "pings"


def _hub(tmp_path) -> Path:
    hub = tmp_path / "hub"
    (hub / "config").mkdir(parents=True)
    (hub / "data").mkdir()
    (hub / "config" / "config.json").write_text("{}", encoding="utf-8")
    (hub / "config" / "owner_profile.json").write_text('{"current": {"city": "Essen"}}', encoding="utf-8")
    (hub / "data" / "owner_dossier.md").write_text("# dossier", encoding="utf-8")
    _make_db(hub / "data" / "memory.sqlite")
    return hub


def test_a_run_cannot_write_to_production_files(tmp_path, monkeypatch):
    from logic import tools
    hub = _hub(tmp_path)
    monkeypatch.chdir(hub)
    source_profile = (hub / "config" / "owner_profile.json").read_text(encoding="utf-8")
    sandbox = harness.Sandbox(hub)
    try:
        sandbox.enter()
        sandbox.fresh(_t("2026-09-29T14:00"))
        tools._tool_update_profile({"category": "current", "field": "city", "action": "set",
                                    "value": "Bochum"}, None)
        assert "Bochum" in (sandbox.dir / "config" / "owner_profile.json").read_text(encoding="utf-8")
        assert (hub / "config" / "owner_profile.json").read_text(encoding="utf-8") == source_profile

        # the next scenario starts from the original, not from the previous one's writes
        sandbox.fresh(_t("2026-09-29T14:00"))
        assert "Bochum" not in (sandbox.dir / "config" / "owner_profile.json").read_text(encoding="utf-8")
    finally:
        sandbox.cleanup()
    conn = sqlite3.connect(hub / "data" / "memory.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0] == 2, "source database changed"


def test_scenarios_can_point_to_memory_rows(tmp_path):
    hub = _hub(tmp_path)
    (tmp_path / "s.json").write_text(json.dumps([{"name": "x", "turns": [
        {"memory_id": 2},
        {"text": "(scheduled) обед", "at": "2026-09-29T15:00"},
    ]}]), encoding="utf-8")
    sc = harness.load_scenarios(tmp_path / "s.json", hub / "data" / "memory.sqlite")[0]
    assert sc.turns[0].text == "я в больнице" and sc.turns[0].kind == "owner"
    assert sc.turns[0].at.startswith("2026-09-29T14:30")
    assert sc.turns[1].kind == "scheduled"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A Runner on a sandboxed hub, with generation faked at the handler seam."""
    from handlers import multi_bot
    from logic import tools
    hub = _hub(tmp_path)
    monkeypatch.chdir(hub)
    monkeypatch.setenv("MAIN_CHAT_ID", "-100")
    monkeypatch.setenv("ALLOWED_USER_IDS", "7")
    monkeypatch.setattr(multi_bot, "_generate_cipher", multi_bot._generate_cipher)
    monkeypatch.setattr(tools, "execute_tool", tools.execute_tool)
    seen = []
    real_generate = multi_bot._generate

    async def fake_generate(agent, user_text, context, chat_id=0, **kw):
        if agent.executor == "cipher_subprocess":
            # the real path, so that the runner's own Cipher stub is what's tested
            return await real_generate(agent, user_text, context, chat_id, **kw)
        from utils.time import now_local
        seen.append((agent.name, user_text, now_local()))
        return f"ответ {agent.name}"

    monkeypatch.setattr(multi_bot, "_generate", fake_generate)
    from logic.agents import REDMOND
    monkeypatch.setattr(multi_bot, "route", lambda text, state, **kw: (REDMOND, False))
    sandbox = harness.Sandbox(hub)
    sandbox.enter()
    r = harness.Runner(sandbox, pause=0, use_judge=False)
    r.seen = seen
    yield r
    r.close()
    sandbox.cleanup()


def _scenario(*turns):
    return harness.Scenario(name="s", turns=[harness.Turn(kind=k, text=t, at=a) for k, t, a in turns])


def test_owner_message_goes_through_the_production_handlers_in_replayed_time(runner):
    sc = _scenario(("owner", "как дела?", "2026-09-29T14:05:00"))
    results = asyncio.run(runner.run([sc]))
    r = results[0]
    assert r.replies == ["ответ Redmond"] and r.agent == "Redmond"
    name, text, now = runner.seen[0]
    assert (name, text) == ("Redmond", "как дела?")
    assert now.strftime("%Y-%m-%d %H:%M") == "2026-09-29 14:05", "the turn ran in real time, not replayed"


def test_cipher_is_never_called(runner):
    sc = _scenario(("owner", "Шифр, перезапусти сервис", "2026-09-29T14:05:00"))
    r = asyncio.run(runner.run([sc]))[0]
    assert all("Cipher" != name for name, _t2, _n in runner.seen)
    assert "(Cipher в прогоне не вызывается)" in r.replies


def test_scheduled_prompt_is_skipped_while_muted(runner, monkeypatch):
    from logic import coach_storage
    monkeypatch.setattr(coach_storage, "muted_now", lambda: True)
    sc = _scenario(("scheduled", "(scheduled) обед", "2026-09-29T15:00:00"))
    r = asyncio.run(runner.run([sc]))[0]
    assert r.skipped == "muted" and not r.replies
    assert not any(v.startswith("ping_dropped") for v in r.violations)


# ---------- rules ----------

def _res(**kw):
    base = dict(replies=["Ок"], agent="Iris", tool_calls=[], errors=[], log=[], seconds=1.0,
                mute_after=None, skipped="")
    base.update(kw)
    return SimpleNamespace(**base)


def _turn(kind="owner", text="поел борща", expect=None):
    return harness.Turn(kind=kind, text=text, at="2026-09-29T14:00", expect=expect or {})


def test_rules_catch_stubs_and_leaks():
    v = checks.check(_turn(), _res(replies=["Готово: записала в дневник."]), ["поел борща"])
    assert any(x.startswith("stub") for x in v)
    v = checks.check(_turn(), _res(replies=["Вызову mute_notifications(action=set)"]), ["x"])
    assert any(x.startswith("leak") for x in v)


def test_rules_catch_diary_entries_the_owner_never_said():
    ok = checks.check(_turn(), _res(tool_calls=[("add_diary_entry", {"text": "Поел борщ"})]), ["поел борща"])
    assert not [x for x in ok if x.startswith("diary")]
    bad = checks.check(_turn(text="мут на 7 дней"),
                       _res(tool_calls=[("add_diary_entry", {"text": "Завтрак"})]), ["мут на 7 дней"])
    assert any(x.startswith("diary_ungrounded") for x in bad)
    ping = checks.check(_turn(kind="scheduled", text="(scheduled) обед"),
                        _res(tool_calls=[("add_diary_entry", {"text": "обед"})]), [])
    assert any(x.startswith("diary_on_ping") for x in ping)


def test_rules_report_a_dropped_ping_and_blocked_attempts():
    v = checks.check(_turn(kind="scheduled", text="(scheduled) обед"),
                     _res(replies=[], log=["WARNING core.scheduler: Scheduled job for Iris: модели не ответили"]), [])
    assert any(x.startswith("ping_dropped") for x in v)
    v = checks.check(_turn(), _res(log=["WARNING logic.response_generator: Запись в дневник отклонена: …"]), ["x"])
    assert any(x.startswith("guard") for x in v)


def test_rules_check_scenario_expectations():
    t = _turn(expect={"agent": "Iris", "tools": ["mute_notifications"], "mute": True,
                      "reply_not": ["завтрак"]})
    v = checks.check(t, _res(agent="Redmond", replies=["Приятного завтрака"]), ["x"])
    rules = {x.split(":")[0] for x in v}
    assert {"agent", "tool_missing", "mute", "reply_forbidden"} <= rules


def test_no_reply_to_the_owner_is_a_violation():
    v = checks.check(_turn(), _res(replies=[]), ["x"])
    assert any(x.startswith("no_reply") for x in v)


# ---------- judge ----------

def test_judge_output_is_parsed_through_fences_and_prose():
    text = 'вот:\n```json\n{"relevance": 2, "facts": 1, "context": 2, "tone": 2, "issues": ["a"]}\n```'
    parsed = judge.parse(text)
    assert parsed["scores"]["facts"] == 1 and parsed["issues"] == ["a"]
    assert judge.parse('{"relevance": 3}') is None


def test_an_invented_fact_fails_the_turn():
    assert judge.verdict({"relevance": 2, "facts": 1, "context": 2, "tone": 2}) == "fail"
    assert judge.verdict({"relevance": 2, "facts": 2, "context": 1, "tone": 2}) == "weak"
    assert judge.verdict({"relevance": 2, "facts": 2, "context": 2, "tone": 2}) == "pass"


def test_judge_falls_back_to_the_next_model():
    answers = {"a": "", "b": '{"relevance": 2, "facts": 2, "context": 2, "tone": 2}'}
    out = judge.grade(_turn(), _res(), [], models=["a", "b"], call=lambda m, p: answers[m])
    assert out["model"] == "b" and out["verdict"] == "pass"


def test_judge_is_told_a_scheduled_prompt_is_not_the_owner():
    prompt = judge.build_prompt(_turn(kind="scheduled", text="(scheduled) обед"), _res(), [])
    assert "Vlad did not write this" in prompt


def test_a_crashing_judge_does_not_end_the_run(runner, monkeypatch):
    """Sep 29 smoke run: the judge crashed on its input and took the whole run,
    report included, down with it."""
    runner.use_judge = True

    def broken(*a, **kw):
        raise AttributeError("'list' object has no attribute 'strip'")

    monkeypatch.setattr(judge, "grade", broken)
    sc = _scenario(("owner", "как дела?", "2026-09-29T14:05:00"),
                   ("owner", "а сейчас?", "2026-09-29T14:06:00"))
    results = asyncio.run(runner.run([sc]))
    assert len(results) == 2
    assert all(r.judge["verdict"] == "unjudged" for r in results)


def test_owner_facts_reach_the_judge_as_text(runner, monkeypatch):
    runner.start(_scenario(("owner", "x", "2026-09-29T14:05:00")))
    rg = runner.dispatcher.response_generator
    monkeypatch.setattr(rg, "_compact_owner_facts", lambda: ["учёба: WI", "английский: B1–B2"])
    assert runner.known_facts() == "учёба: WI\nанглийский: B1–B2"
