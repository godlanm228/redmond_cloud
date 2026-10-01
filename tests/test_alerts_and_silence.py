"""Аварии доходят, тишина соблюдает названный срок, пинги знают время суток.

Все три случая — из боевого лога и базы 26.08–28.09.2026:
  • Cipher лежал с 10.09, ERROR в логе каждый день, в чат — ничего: полная
    тишина глушила и аварии;
  • «не разговаривай со мной 7 дней, фул мут» → тишина навсегда, простояла
    две недели, а бот пообещал «до 21 сентября»;
  • первое сообщение за день в 20:21 → в 20:30 «Время к обеду».
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from core import alerts
from logic import coach_storage
from logic.tools import execute_tool
from utils.time import now_local


# ---------- аварии ----------

def test_first_alert_goes_out():
    assert alerts.should_send("cipher_auth", "Cipher не отвечает", muted=True)


def test_same_alert_is_not_repeated_the_same_day():
    alerts.mark_sent("cipher_auth", "Cipher не отвечает")
    assert not alerts.should_send("cipher_auth", "Cipher не отвечает", muted=False)


def test_changed_alert_goes_out_at_once():
    alerts.mark_sent("cipher_auth", "Cipher скоро отвалится")
    assert alerts.should_send("cipher_auth", "Cipher не отвечает", muted=True)


def test_under_full_silence_the_same_alert_repeats_rarely(monkeypatch):
    alerts.mark_sent("cipher_auth", "Cipher не отвечает")
    later = now_local() + timedelta(days=2)
    monkeypatch.setattr(alerts, "now_local", lambda: later)
    assert alerts.should_send("cipher_auth", "Cipher не отвечает", muted=False)
    assert not alerts.should_send("cipher_auth", "Cipher не отвечает", muted=True)


def test_alert_passes_through_full_silence():
    """Раньше: «полная тишина, сообщение не отправляю» — и две недели молчания."""
    from core import scheduler
    coach_storage.set_mute(mode="forever", scope="all")
    sent = []

    class Coord:
        async def respond_as(self, agent, chat_id, text, emoji, fmt):
            sent.append(text)

    asyncio.run(scheduler._send_alert(Coord(), 1, "cipher_auth", "Cipher", "🧠",
                                      "🧠 Cipher не отвечает"))
    assert sent == ["🧠 Cipher не отвечает"]


def test_failed_generation_is_not_sent_proactively(monkeypatch):
    """Скедулер не шлёт отказ и не пишет «Scheduled job done» на провал."""
    from core import scheduler
    from logic.response_generator import Reply
    sent = []

    class Coord:
        async def respond_as(self, *a, **kw):
            sent.append(a)

        def typing(self, *a, **kw):
            class _Ctx:
                async def __aenter__(self_inner):
                    return None

                async def __aexit__(self_inner, *exc):
                    return False
            return _Ctx()

    dispatcher = SimpleNamespace(
        intent_recognizer=SimpleNamespace(recognize=lambda text: None),
        response_generator=SimpleNamespace(
            generate=lambda *a, **kw: Reply("Модели не ответили", failed=True)),
    )
    asyncio.run(scheduler._generate_and_send(dispatcher, Coord(), 1, scheduler.IRIS_TICKER, "пинг"))
    assert sent == []


def test_ticker_cannot_write_to_the_diary():
    """10.09.2026 пинг записал в дневник сам себя."""
    from core import scheduler
    assert "add_diary_entry" not in (scheduler.IRIS_TICKER.allowed_tools or [])


# ---------- тишина ----------

def test_named_duration_beats_forever():
    """14.09.2026: «на 7 дней, фул мут» стало тишиной навсегда."""
    msg = execute_tool("mute_notifications", {"mode": "forever", "days": 7, "scope": "all"})
    info = coach_storage.mute_info()
    assert info["until"] != "forever", msg
    until = datetime.fromisoformat(info["until"])
    assert timedelta(days=6, hours=23) <= until - now_local() <= timedelta(days=7, minutes=1)
    assert until.strftime("%d.%m") in msg, f"в результате нет даты окончания: {msg}"


def test_ten_days_are_not_silently_cut_to_seven():
    coach_storage.set_mute(mode="hours", hours=240)
    until = datetime.fromisoformat(coach_storage.mute_info()["until"])
    assert until - now_local() > timedelta(days=9)


def test_forever_still_works_when_no_duration_is_named():
    execute_tool("mute_notifications", {"mode": "forever", "scope": "all"})
    assert coach_storage.mute_info()["until"] == "forever"


# ---------- пинг «обед» ----------

def _situation(now):
    from logic.situation_engine import DaySituation, ShiftSituation
    shift = ShiftSituation(record=None, active_record=None, start_at=None, end_at=None,
                           status="planned", source="", confidence="", updated_at=None,
                           confirmed_at=None)
    # Писал два часа назад: пинг не перебивает разговор (QUIET_AFTER_MESSAGE_MIN).
    return DaySituation(now=now, day_state={}, pings={"checkin": "12:00"}, owner_seen=True,
                        muted=False, tags=set(), entries_today=1, wake_time=None,
                        shift=shift, in_study_block=False,
                        last_msg=(now - timedelta(hours=2)).strftime("%H:%M"))


def _decision_at(monkeypatch, hour, minute=0):
    from logic import pings
    monkeypatch.setattr(pings, "crunch_deadline", lambda: None)
    now = now_local().replace(hour=hour, minute=minute, second=0, microsecond=0)
    return pings._slot_decision(_situation(now), now)


def test_lunch_ping_in_the_afternoon(monkeypatch):
    decision = _decision_at(monkeypatch, 14, 30)
    assert decision is not None and decision[0] == "meal"


def test_no_lunch_ping_at_half_past_eight_in_the_evening(monkeypatch):
    decision = _decision_at(monkeypatch, 20, 30)
    assert decision is None or decision[0] != "meal", decision
