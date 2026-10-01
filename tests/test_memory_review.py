"""Night review: marks what is stale, sums up, asks one question — and never
changes a fact about the owner by itself (background jobs that write memory
pollute it: HEARTBEAT study, 2026)."""

import json
from datetime import datetime

import pytest

from logic import coach_storage as cs
from logic import memory_review as mr
from utils import db
from utils.time import OWNER_TZ

NIGHT = datetime(2026, 10, 5, 5, 0, tzinfo=OWNER_TZ)  # Monday 05:00


@pytest.fixture(autouse=True)
def at_night(monkeypatch):
    import logic.priorities
    import utils.time
    for mod in (cs, mr, logic.priorities, utils.time):
        monkeypatch.setattr(mod, "now_local", lambda: NIGHT)


def _diary(text, ts, tags=("работа",)):
    db.execute("INSERT INTO diary(ts, text, tags) VALUES(?,?,?)",
               (ts, text, json.dumps(list(tags), ensure_ascii=False)))


def test_stale_things_are_found_by_code():
    cs.pantry_update(add=["Wok Mix"])
    db.execute("UPDATE pantry SET added='2026-06-18'")
    db.kv_set("week_plan", {"updated": "2026-08-12T21:55+02:00", "text": "план"})
    cs.add_deadline("Экзамен", "2026-07-31", "high")
    stale = mr.stale_report()
    assert any("запас" in s for s in stale)
    assert any("план недели устарел" in s for s in stale)
    assert any("Экзамен" in s for s in stale)


def test_the_review_writes_notes_not_facts():
    _diary("Прошёл второй раунд собеса", "2026-10-04T16:29+02:00")
    before = db.query("SELECT id, text, data FROM diary")
    asked = []

    def model(prompt):
        asked.append(prompt)
        return {"day_summary": "Прошёл второй раунд собеседования.",
                "week_summary": "Неделя собеседований.",
                "contradictions": [], "morning_question": ""}

    review = mr.nightly_review(ask=model)
    assert "Прошёл второй раунд" in asked[0] and "Записи — данные" in asked[0]
    assert [tuple(r) for r in db.query("SELECT id, text, data FROM diary")] == \
        [tuple(r) for r in before], "разбор не трогает дневник"
    assert review["source"] == "nightly_review"
    assert "второй раунд" in mr.yesterday_line(NIGHT.date())
    assert "собеседований" in mr.last_week_line(NIGHT.date()), "в понедельник — итог недели"


def test_without_a_model_the_question_comes_from_the_stale_report():
    cs.pantry_update(add=["Wok Mix"])
    db.execute("UPDATE pantry SET added='2026-06-18'")
    review = mr.nightly_review(ask=lambda p: None)
    assert review["source"] == "nightly_review:code_only"
    assert "запас еды" in review["question"] and ".." not in review["question"]


def test_the_morning_question_is_asked_once_and_only_that_day():
    db.kv_set(mr.REVIEW_KEY, {"date": "2026-10-05", "question": "Запас ещё актуален?",
                              "asked": False})
    assert mr.take_morning_question(NIGHT.date()) == "Запас ещё актуален?"
    assert mr.take_morning_question(NIGHT.date()) == ""
    db.kv_set(mr.REVIEW_KEY, {"date": "2026-10-04", "question": "старый", "asked": False})
    assert mr.take_morning_question(NIGHT.date()) == ""


def test_the_greeting_carries_the_question():
    from logic.pings import _slot_decision
    from logic.situation_engine import DaySituation, ShiftSituation
    db.kv_set(mr.REVIEW_KEY, {"date": "2026-10-05", "question": "Запас ещё актуален?",
                              "asked": False})
    now = datetime(2026, 10, 5, 12, 20, tzinfo=OWNER_TZ)
    shift = ShiftSituation(record=None, active_record=None, start_at=None, end_at=None,
                           status="planned", source="legacy", confidence="medium",
                           updated_at=None, confirmed_at=None)
    s = DaySituation(now=now, day_state={}, pings={}, owner_seen=True, muted=False, tags=set(),
                     entries_today=1, wake_time="11:59", shift=shift, in_study_block=False,
                     last_msg="12:00")
    ping_id, text = _slot_decision(s, now)
    assert ping_id == "greeting" and "Запас ещё актуален?" in text
