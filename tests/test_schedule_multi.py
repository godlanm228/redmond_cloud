"""Several shifts a day, events next to them, and what the bot says about them.

01.10.2026: six screenshots of the university timetable were stored as bar
shifts. The table was keyed by date, so each slot of a day silently replaced
the previous one — 10 slots in, 3 left — and the receipt still said «смен
сохранено: 4». Here: a different time is a different shift; the same shift
(overlapping hours: plan 17:00, fact 17:14) is updated; the receipt lists
what is in the table.
"""

import sqlite3
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from logic import tools
from logic import week_schedule as ws
from utils import db
from utils.time import OWNER_TZ

D = "2026-10-05"
DAY = date(2026, 10, 5)


def _shift(start, end, source="photo", status="planned", **kw):
    return {"date": D, "start": start, "end": end, "status": status, "source": source,
            "confidence": "high", **kw}


def test_the_oct_1_screenshots_no_longer_overwrite_each_other():
    """The exact Monday slots from 01.10 (as they were taken for shifts)."""
    slots = [("08:00", "11:20"), ("12:20", "14:00"), ("14:05", "15:45"), ("20:00", "21:30")]
    outcome = ws.apply_shifts([_shift(s, e) for s, e in slots])
    assert outcome.saved == 4
    assert [(s["start"], s["end"]) for s in ws.get_shifts(DAY)] == slots
    assert [(a["start"], a["end"]) for a in outcome.applied] == slots


def test_plan_then_fact_is_the_same_shift():
    ws.apply_shifts([_shift("17:00", "23:00")])
    ws.apply_shifts([_shift("17:14", "22:59", source="photo")])
    assert [(s["start"], s["end"]) for s in ws.get_shifts(DAY)] == [("17:14", "22:59")]


def test_an_overnight_shift_overlaps_its_late_correction():
    ws.apply_shifts([_shift("22:00", "02:00")])
    ws.apply_shifts([_shift("23:00", "03:00")])
    assert len(ws.get_shifts(DAY)) == 1


def test_a_moved_shift_replaces_only_when_said_so():
    ws.apply_shifts([_shift("18:00", "23:00", source="text")])
    ws.apply_shifts([_shift("10:00", "14:00", source="text")])
    assert len(ws.get_shifts(DAY)) == 2, "непересекающиеся часы — вторая смена"

    db.execute("DELETE FROM shifts")
    ws.apply_shifts([_shift("18:00", "23:00", source="text")])
    ws.apply_shifts([_shift("10:00", "14:00", source="text", replaces=True)])
    assert [(s["start"], s["end"]) for s in ws.get_shifts(DAY)] == [("10:00", "14:00")]


def test_a_photo_does_not_override_his_words_but_may_add_another_shift():
    ws.apply_shifts([_shift("17:00", "23:00", source="text")])
    out = ws.apply_shifts([_shift("16:00", "23:00", source="photo")])
    assert out.saved == 0 and len(out.conflicts) == 1
    out = ws.apply_shifts([_shift("09:00", "12:00", source="photo")])
    assert out.saved == 1 and not out.conflicts
    assert [(s["start"], s["source"]) for s in ws.get_shifts(DAY)] == [
        ("09:00", "photo"), ("17:00", "text")]


def test_cancel_without_hours_cancels_every_shift_of_the_day():
    ws.apply_shifts([_shift("09:00", "12:00", source="text"),
                     _shift("18:00", "23:00", source="text")])
    ws.apply_shifts([{"date": D, "status": "cancelled", "source": "text"}])
    assert ws.get_shifts(DAY) == []
    assert all(r["status"] == "cancelled" for r in ws.get_shift_records(DAY))


def test_the_relevant_shift_is_the_running_or_the_next_one():
    ws.apply_shifts([_shift("09:00", "12:00"), _shift("18:00", "23:00")])
    at = lambda h, m=0: datetime(2026, 10, 5, h, m, tzinfo=OWNER_TZ)  # noqa: E731
    assert ws.get_shift(DAY, at(10))["start"] == "09:00"
    assert ws.get_shift(DAY, at(13))["start"] == "18:00"
    assert ws.get_shift(DAY, at(23, 30))["start"] == "18:00"


def test_old_date_keyed_table_is_rebuilt_with_its_rows(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE shifts (date TEXT PRIMARY KEY, start TEXT, end TEXT,"
                 " status TEXT NOT NULL DEFAULT 'planned', source TEXT NOT NULL DEFAULT 'unknown',"
                 " confidence TEXT NOT NULL DEFAULT 'medium', updated TEXT NOT NULL,"
                 " last_confirmed_at TEXT, note TEXT)")
    conn.execute("INSERT INTO shifts VALUES('2026-09-10','16:00','23:00','planned','text',"
                 "'high','2026-09-09T10:00',NULL,'Бар')")
    conn.commit()
    conn.close()

    db.set_db_path(path)
    assert ws.get_shifts(date(2026, 9, 10))[0]["note"] == "Бар"
    ws.apply_shifts([{"date": "2026-09-10", "start": "09:00", "end": "12:00", "source": "text"}])
    assert len(ws.get_shifts(date(2026, 9, 10))) == 2


# ---------------------------------------------------------------------------
# What the agents see and can do
# ---------------------------------------------------------------------------

@pytest.fixture
def oct5(monkeypatch):
    fixed = datetime(2026, 10, 5, 8, 0, tzinfo=OWNER_TZ)
    monkeypatch.setattr(ws, "now_local", lambda: fixed)
    import utils.time
    monkeypatch.setattr(utils.time, "now_local", lambda: fixed)
    return fixed


def test_the_week_view_lists_all_shifts_and_events_with_ids(oct5):
    ws.apply_shifts([_shift("09:00", "12:00"), _shift("18:00", "23:00")])
    tennis = ws.add_event("Tennis", "sport", "20:00", "21:30", weekly_from=DAY)
    ws.add_event("Diskrete Mathematik", "lecture", "12:20", "14:00", on=DAY)
    first = ws.format_week(1)
    assert "смена 09:00–12:00" in first and "смена 18:00–23:00" in first
    assert f"Tennis (спорт, еженедельно) #{tennis}" in first
    assert "12:20–14:00 Diskrete Mathematik #" in first
    assert "Tennis" in ws.format_week(8).splitlines()[7], "через неделю — снова понедельник"


def test_study_slots_are_only_classes(oct5):
    ws.add_event("Tennis", "sport", "20:00", "21:30", weekly_from=DAY)
    ws.add_event("Diskrete Mathematik", "lecture", "12:20", "14:00", on=DAY)
    assert [s[2] for s in ws.study_slots(DAY)] == ["Diskrete Mathematik"]


def test_save_work_shift_says_when_the_day_now_has_two(oct5):
    tools.execute_tool("save_work_shift", {"date": D, "start": "18:00", "end": "23:00"})
    out = tools.execute_tool("save_work_shift", {"date": D, "start": "10:00", "end": "14:00"})
    assert "смен: 2" in out and "перенос" in out
    out = tools.execute_tool("save_work_shift",
                             {"date": "2026-10-06", "start": "10:00", "end": "14:00"})
    assert "смен:" not in out


def test_weekly_tennis_via_the_tool(oct5):
    out = tools.execute_tool("add_schedule_event", {
        "title": "Tennis", "kind": "sport", "date": D, "start": "20:00", "end": "21:30",
        "weekly": True})
    assert "каждый понедельник" in out
    for d in (5, 12, 19, 26):
        assert [r["title"] for r in ws.day_events(date(2026, 10, d))] == ["Tennis"]
    assert ws.day_events(date(2026, 10, 6)) == []


def test_removing_an_event_needs_its_id_from_a_view(oct5):
    row = ws.add_event("Tennis", "sport", "20:00", "21:30", weekly_from=DAY)
    session = tools.ToolSession()
    blind = tools.execute_tool("remove_schedule_event", {"event_id": row}, None, session=session)
    assert "Убрала" not in blind
    tools.execute_tool("get_week_schedule", {"days": 8}, None, session=session)
    out = tools.execute_tool("remove_schedule_event", {"event_id": row, "date": "2026-10-19"},
                             None, session=session)
    assert "серия закончена 2026-10-18" in out
    assert ws.day_events(date(2026, 10, 12)) and not ws.day_events(date(2026, 10, 19))


def test_an_event_without_a_kind_or_with_half_a_time_is_refused(oct5):
    out = tools.execute_tool("add_schedule_event",
                             {"title": "X", "kind": "party", "date": D})
    assert "Не добавила" in out
    out = tools.execute_tool("add_schedule_event",
                             {"title": "X", "kind": "other", "date": D, "start": "10:00"})
    assert "Не добавила" in out


# ---------------------------------------------------------------------------
# Code-written prompts carry only the tools they need
# ---------------------------------------------------------------------------

def test_a_photo_prompt_gets_only_the_tools_it_names():
    from logic import response_generator as rgm
    from logic import toolbox
    g = object.__new__(rgm.ResponseGenerator)
    all_tools = toolbox.model_tools(None)
    ctx = SimpleNamespace(understanding=None, needs=["schedule"], query_vec=None,
                          user_text="(scheduled week-plan) Влад загрузил новый график смен.")
    offered, deferred = g._select_tools(ctx, "Iris", all_tools)
    names = {s["function"]["name"] for s in offered}
    assert names == {"schedule", "get_current_time"}
    assert deferred, "остальное — через load_tools, а не в каждом запросе"

    ctx.needs = None
    offered, _ = g._select_tools(ctx, "Iris", all_tools)
    assert len(offered) == len(all_tools), "без needs код-промпт по-прежнему получает всё"


# ---------------------------------------------------------------------------
# Who answered goes into the history
# ---------------------------------------------------------------------------

def test_the_answering_agent_is_written_to_the_history(monkeypatch):
    import threading

    from logic import response_generator as rgm
    from logic.intent_recognizer import Intent
    from utils import gemini, llm_gate

    rg = object.__new__(rgm.ResponseGenerator)
    rg.config = SimpleNamespace(gemini_api_key="k", gemini_model="gemini-3.6-flash",
                                gemini_fallback_models=[], groq_api_key="", groq_model="",
                                groq_fallback_model="")
    rg.mem, rg.top_k, rg.max_history = None, 3, 6
    rg.history_by_chat, rg._history_guard, rg._history_loaded = {}, threading.RLock(), set()
    llm_gate.configure_from(rg.config)
    rg._build_system_prompt = lambda ctx: "system"
    rg._build_user_message = lambda ctx: ctx.user_text
    monkeypatch.setattr(gemini, "generate_contents", lambda contents, **kw: {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Привет!"}]}}]})
    monkeypatch.delenv("REDMOND_GEMINI_API_KEY", raising=False)
    iris = SimpleNamespace(name="Iris", provider_order=["gemini"], allowed_tools=None,
                           temperature=0.5, max_tokens=800, emoji="🎯")

    rg.generate(Intent(name="chat", slots={}), "привет", "owner", iris, 42)
    assert db.history_load(42, 5)[-1]["agent"] == "Iris"


# ---------------------------------------------------------------------------
# The morning greeting belongs to the morning
# ---------------------------------------------------------------------------

def _situation(now, wake):
    from logic.situation_engine import DaySituation, ShiftSituation
    shift = ShiftSituation(record=None, active_record=None, start_at=None, end_at=None,
                           status="planned", source="legacy", confidence="medium",
                           updated_at=None, confirmed_at=None)
    return DaySituation(now=now, day_state={}, pings={}, owner_seen=True, muted=False,
                        tags=set(), entries_today=1, wake_time=wake, shift=shift,
                        in_study_block=False, last_msg=now.strftime("%H:%M"))


def test_greeting_does_not_wait_out_a_mute_until_the_evening():
    """01.10.2026: woke 11:59, mute until 16:45, greeting at 17:00 «проснулся недавно»."""
    from logic.pings import _slot_decision
    late = datetime(2026, 10, 1, 17, 0, tzinfo=OWNER_TZ)
    decision = _slot_decision(_situation(late, "11:59"), late)
    assert decision is None or decision[0] != "greeting"

    on_time = datetime(2026, 10, 1, 12, 20, tzinfo=OWNER_TZ)
    assert _slot_decision(_situation(on_time, "11:59"), on_time)[0] == "greeting"


# ---------------------------------------------------------------------------
# An album is one vision request
# ---------------------------------------------------------------------------

def test_an_album_is_read_with_one_request(monkeypatch):
    from logic import vision
    calls = []

    def fake(prompt, images, max_tokens=700):
        calls.append(len(images))
        return ('[{"type": "calendar", "events": [{"date": "2026-10-05", "start": "12:20",'
                ' "end": "14:00", "title": "Diskrete Mathematik"}], "description": "пн"},'
                ' {"type": "food", "food_kind": "meal", "dish": "паста", "description": "еда"}]')

    monkeypatch.setattr(vision, "_call_gemini_vision", fake)
    out = vision.analyze_images(["a", "b"])
    assert calls == [2]
    assert [r["type"] for r in out] == ["calendar", "food"]
    assert out[0]["events"][0]["title"] == "Diskrete Mathematik"


def test_a_malformed_album_answer_falls_back_to_one_by_one(monkeypatch):
    from logic import vision
    calls = []

    def fake(prompt, images, max_tokens=700):
        calls.append(len(images))
        if len(images) > 1:
            return '[{"type": "other", "description": "одно на двоих"}]'
        return '{"type": "other", "description": "фото"}'

    monkeypatch.setattr(vision, "_call_gemini_vision", fake)
    out = vision.analyze_images(["a", "b"])
    assert calls == [2, 1, 1] and len(out) == 2
