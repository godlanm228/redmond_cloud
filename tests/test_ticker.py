"""The day ticker after the redesign of Oct 1, 2026.

Data 10.06–01.10: 138 pings; the «как ты, какие планы» cold-start was 75 of
them with a 25% answer rate, and on 46 days the bot wrote while the owner did
not write once. Pings about something concrete (food before a shift, training,
a deadline) were answered about twice as often. So: concrete facts only,
time-critical reminders even when he has not written, the rest only when he is
around and not in the middle of a conversation, and types he ignores switch off.
"""

from datetime import datetime, timedelta

import pytest

from logic import coach_storage as cs
from logic import pings
from logic import week_schedule as ws
from logic.situation_engine import DaySituation, ShiftSituation
from utils import db
from utils.time import OWNER_TZ

MON = datetime(2026, 10, 5, 0, 0, tzinfo=OWNER_TZ)


def at(h, m=0, day=MON):
    return day.replace(hour=h, minute=m)


def situation(now, *, seen=True, last_msg_ago=120, pings_today=None, tags=(), wake=None):
    shift = ShiftSituation(record=None, active_record=None, start_at=None, end_at=None,
                           status="planned", source="", confidence="", updated_at=None,
                           confirmed_at=None)
    last = (now - timedelta(minutes=last_msg_ago)).strftime("%H:%M") if seen else None
    return DaySituation(now=now, day_state={}, pings=dict(pings_today or {}), owner_seen=seen,
                        muted=False, tags=set(tags), entries_today=0, wake_time=wake,
                        shift=shift, in_study_block=False, last_msg=last)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    holder = {"now": at(12)}
    for mod in (cs, pings, ws):
        monkeypatch.setattr(mod, "now_local", lambda: holder["now"])
    import utils.time
    monkeypatch.setattr(utils.time, "now_local", lambda: holder["now"])
    monkeypatch.setattr(pings, "crunch_deadline", lambda: None)
    monkeypatch.setattr(pings, "radar_deadline", lambda: None)
    return holder


def decide(clock, now, **kw):
    clock["now"] = now
    return pings._slot_decision(situation(now, **kw), now)


def test_no_more_how_are_you_when_he_has_not_written(clock):
    for h in (12, 13, 15, 18):
        assert decide(clock, at(h), seen=False) is None


def test_leave_for_a_class_even_if_he_has_not_written(clock):
    ws.add_event("Diskrete Mathematik - Vorlesung", "lecture", "12:20", "14:00", on=MON.date())
    d = decide(clock, at(10, 50), seen=False)
    assert d and d[0] == "leave:12:20" and "11:20" in d[1]


def test_early_start_tomorrow_is_said_the_evening_before(clock):
    ws.add_event("Englisch Seminar", "lecture", "08:00", "11:20", on=(MON + timedelta(days=1)).date())
    d = decide(clock, at(21, 30), seen=False)
    assert d and d[0] == "tomorrow_early" and "07:00" in d[1]


def test_food_before_a_long_block_of_classes(clock):
    for s, e in (("12:20", "14:00"), ("14:05", "15:45"), ("15:50", "17:20")):
        ws.add_event("Vorlesung", "lecture", s, e, on=MON.date())
    d = decide(clock, at(10, 30))
    assert d and d[0] == "meal" and "пары с 12:20" in d[1]


def test_not_while_he_is_in_a_conversation(clock):
    assert decide(clock, at(15), last_msg_ago=20) is None
    assert decide(clock, at(15), last_msg_ago=90)[0] == "meal"


def test_how_did_it_go_after_his_own_plan(clock):
    clock["now"] = at(11, 59)
    cs.add_diary_entry("Ich hab heute bewerbungsgepräch um 15", tags=["план", "работа"],
                       source="owner")
    d = decide(clock, at(16, 30), tags=("питание",))
    assert d and d[0] == "followup" and "bewerbungsgepräch" in d[1]


def test_no_training_question_when_tennis_is_planned(clock):
    ws.add_event("Tennis", "sport", "20:00", "21:30", weekly_from=MON.date())
    assert decide(clock, at(18), tags=("питание",)) is None


def test_a_ping_he_keeps_ignoring_switches_off(clock):
    log = [{"type": "meal", "at": (at(14) - timedelta(days=i)).isoformat(timespec="minutes"),
            "answered": False} for i in range(1, 7)]
    db.kv_set(cs.PING_LOG_KEY, log)
    assert decide(clock, at(15)) is None
    from logic.memory_review import stale_report
    assert any("пинг «meal» выключен" in s for s in stale_report(MON.date()))


def test_an_answer_within_the_hour_counts(clock):
    clock["now"] = at(15)
    cs.mark_ping("meal")
    clock["now"] = at(15, 40)
    cs.mark_owner_seen()
    assert cs.ping_reply_rate("meal") == (1, 1)


def test_one_message_after_two_silent_days(clock):
    clock["now"] = at(12) - timedelta(days=3)
    cs.mark_owner_seen()
    d = decide(clock, at(14), seen=False)
    assert d and d[0] == "absence" and "3 дн." in d[1]
    clock["now"] = at(14)
    pings._on_send("absence", at(14))
    assert decide(clock, at(15), seen=False) is None, "одно сообщение на одно отсутствие"


def test_three_a_day_at_most(clock):
    full = {"greeting": "12:00", "meal": "14:00", "followup": "16:00"}
    assert decide(clock, at(18), pings_today=full) is None
