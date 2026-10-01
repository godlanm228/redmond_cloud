"""Calendar files and calendar screenshots become the owner's schedule.

01.10.2026 the owner sent his university timetable twice: as six screenshots
of CampusNet and as the CampusNet export (.ics). The screenshots were taken for
bar shifts (the photo prompt called any calendar app a shift planner) and,
with one shift per date, ten slots collapsed into three. The file reached no
handler at all. The fixture below has the exact shape of that export:
UTF-16 LE without a BOM, a home-made VTIMEZONE «CampusNetZeit», umlauts,
explicit dates instead of a weekly rule, a week without classes (project
week) and the switch to winter time on Oct 25.
"""

from datetime import date

import pytest

from logic import calendar_import as ci
from logic import documents
from logic import week_schedule as ws
from utils import db

TODAY = date(2026, 10, 1)


def _event(day, start, end, summary, location, uid):
    return (
        "BEGIN:VEVENT\r\n"
        f"DTSTART;TZID=CampusNetZeit:{day}T{start}00\r\n"
        f"DTEND;TZID=CampusNetZeit:{day}T{end}00\r\n"
        f"LOCATION:{location}\r\n"
        f"UID:{uid}\r\n"
        "DTSTAMP:20261001T143545Z\r\n"
        f"SUMMARY:{summary}\r\n"
        "END:VEVENT\r\n"
    )


def campusnet_ics(events) -> bytes:
    body = (
        "BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        "PRODID:-//Datenlotsen Informationssysteme GmbH//CampusNet//DE\r\n"
        "METHOD:PUBLISH\r\n"
        "BEGIN:VTIMEZONE\r\nTZID:CampusNetZeit\r\n"
        "BEGIN:STANDARD\r\nDTSTART:16011028T030000\r\n"
        "RRULE:FREQ=YEARLY;BYDAY=-1SU;BYMONTH=10\r\n"
        "TZOFFSETFROM:+0200\r\nTZOFFSETTO:+0100\r\nEND:STANDARD\r\n"
        "BEGIN:DAYLIGHT\r\nDTSTART:16010325T020000\r\n"
        "RRULE:FREQ=YEARLY;BYDAY=-1SU;BYMONTH=3\r\n"
        "TZOFFSETFROM:+0100\r\nTZOFFSETTO:+0200\r\nEND:DAYLIGHT\r\n"
        "END:VTIMEZONE\r\n"
        + "".join(events) + "END:VCALENDAR\r\n"
    )
    return body.encode("utf-16-le")  # CampusNet: UTF-16 LE, no BOM


# Two weeks of classes, a project week without any, then one more week.
TIMETABLE = [
    _event("20261005", "1220", "1400", "Diskrete Mathematik - Vorlesung", "Hörsaal 2", "a1"),
    _event("20261005", "1405", "1545", "Grundlagen der Informatik - Praktikum", "PC-Raum", "a2"),
    _event("20261006", "1500", "1635", "Diskrete Mathematik - Übung", "Seminarraum 1", "a3"),
    _event("20261009", "0800", "1120", "Einführung in akademisches Arbeiten - Gruppenprojekt",
           "Gruppentreffen", "a4"),
    _event("20261012", "1220", "1400", "Diskrete Mathematik - Vorlesung", "Hörsaal 2", "b1"),
    _event("20261026", "1220", "1400", "Diskrete Mathematik - Vorlesung", "Hörsaal 2", "c1"),
    _event("20261030", "1405", "1820", "Wirtschaft I - Vorlesung/Übung", "Hörsaal 3", "c2"),
]


@pytest.fixture
def today(monkeypatch):
    from datetime import datetime
    from utils.time import OWNER_TZ
    fixed = datetime(2026, 10, 1, 16, 0, tzinfo=OWNER_TZ)
    monkeypatch.setattr(ci, "now_local", lambda: fixed)
    monkeypatch.setattr(ws, "now_local", lambda: fixed)
    return fixed


# ---------------------------------------------------------------------------
# What the file is
# ---------------------------------------------------------------------------

def test_campusnet_export_is_recognised_as_a_calendar_from_its_bytes():
    verdict = documents.inspect(campusnet_ics(TIMETABLE), "832849531700533.ics")
    assert verdict.kind == "calendar" and verdict.ok
    assert "Hörsaal" in verdict.text, "UTF-16 без BOM должен декодироваться с умлаутами"


def test_the_name_does_not_decide_what_a_file_is():
    exe = b"MZ\x90\x00" + b"\x00" * 200
    verdict = documents.inspect(exe, "stundenplan.ics")
    assert not verdict.ok and verdict.kind == "executable"

    zipped = b"PK\x03\x04" + b"\x00" * 200
    assert documents.inspect(zipped, "kalender.ics").kind == "archive"

    not_a_calendar = documents.inspect("Привет, это просто текст".encode(), "plan.ics")
    assert not not_a_calendar.ok and "не календарь" in not_a_calendar.reason


def test_a_calendar_under_a_different_name_is_still_read():
    assert documents.inspect(campusnet_ics(TIMETABLE), "export.txt").kind == "calendar"


def test_scripts_are_refused_even_though_they_are_text():
    verdict = documents.inspect(b"@echo off\r\ndel /q *.*\r\n", "update.bat")
    assert not verdict.ok and verdict.kind == "executable"


def test_unreadable_formats_get_an_honest_answer_not_silence():
    pdf = documents.inspect(b"%PDF-1.7\n...", "vertrag.pdf")
    assert not pdf.ok and "PDF" in pdf.reason
    big = documents.inspect(b"x" * (documents.MAX_BYTES + 1), "huge.ics")
    assert not big.ok and "МБ" in big.reason


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_times_are_the_owners_local_time_across_the_dst_switch():
    text = documents.inspect(campusnet_ics(TIMETABLE), "x.ics").text
    parsed = ci.parse(text, today=TODAY)
    assert parsed.name == "CampusNet" and parsed.university
    by_uid = {e.uid.split("@")[0]: e for e in parsed.events}
    assert (by_uid["a1"].date, by_uid["a1"].start, by_uid["a1"].end) == ("2026-10-05", "12:20", "14:00")
    # 30.10 is already winter time (+01:00) — still 14:05 on the wall clock.
    assert (by_uid["c2"].start, by_uid["c2"].end) == ("14:05", "18:20")


def test_weekly_rules_are_expanded_and_exceptions_respected():
    ev = (
        "BEGIN:VEVENT\r\nDTSTART;TZID=Europe/Berlin:20261005T200000\r\n"
        "DTEND;TZID=Europe/Berlin:20261005T213000\r\n"
        "RRULE:FREQ=WEEKLY;COUNT=4\r\nEXDATE;TZID=Europe/Berlin:20261012T200000\r\n"
        "UID:tennis\r\nSUMMARY:Tennis\r\nEND:VEVENT\r\n"
    )
    text = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//Google Inc//Google Calendar//EN\r\n"
            + ev + "END:VCALENDAR\r\n")
    parsed = ci.parse(text, today=TODAY)
    assert [e.date for e in parsed.events] == ["2026-10-05", "2026-10-19", "2026-10-26"]
    assert not parsed.university


def test_past_events_are_counted_but_not_imported():
    old = _event("20260915", "1000", "1100", "Altes Seminar", "", "old")
    parsed = ci.parse(documents.decode_text(campusnet_ics(TIMETABLE + [old])), today=TODAY)
    assert parsed.past == 1
    assert all(e.date >= "2026-10-01" for e in parsed.events)


def test_a_multi_day_all_day_event_is_stored_day_by_day():
    ev = ("BEGIN:VEVENT\r\nDTSTART;VALUE=DATE:20261019\r\nDTEND;VALUE=DATE:20261024\r\n"
          "UID:pw\r\nSUMMARY:Projektwoche\r\nEND:VEVENT\r\n")
    text = "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//x//y//DE\r\n" + ev + "END:VCALENDAR\r\n"
    parsed = ci.parse(text, today=TODAY)
    assert [e.date for e in parsed.events] == [f"2026-10-{d}" for d in range(19, 24)]
    assert all(e.start == "" for e in parsed.events)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _ev(title, location=""):
    return ci.CalendarEvent(date="2026-10-05", start="10:00", end="11:00", title=title,
                            location=location, uid=title)


def test_rules_place_the_obvious():
    cases = {
        "Diskrete Mathematik - Vorlesung": "lecture",
        "Englisch Seminar": "lecture",
        "Spätschicht Bar": "shift",
        "Tennis": "sport",
        "Urlaub": "rest",
        "Vorstellungsgespräch Bosch": "work",
    }
    events = [_ev(t) for t in cases]
    unplaced = ci.classify(events, ask_model=lambda titles: pytest.fail("модель не нужна"))
    assert unplaced == []
    assert {e.title: e.kind for e in events} == cases


def test_a_university_export_makes_an_unknown_title_a_class():
    event = _ev("Einführung in akademisches Arbeiten - Gruppenprojekt", "Gruppentreffen")
    ci.classify([event], university=True, ask_model=lambda t: pytest.fail("не нужна"))
    assert (event.kind, event.how) == ("lecture", "origin")


def test_the_model_places_the_rest_and_sees_each_title_once():
    asked = []

    def model(titles):
        asked.append(list(titles))
        return {"Zahnarzt": "other", "Geburtstag Mama": "other"}

    events = [_ev("Zahnarzt"), _ev("Zahnarzt"), _ev("Geburtstag Mama")]
    assert ci.classify(events, ask_model=model) == []
    assert asked == [["Zahnarzt", "Geburtstag Mama"]]


def test_without_a_model_unknown_events_are_kept_and_named_not_guessed():
    events = [_ev("Treffen mit Lukas")]
    unplaced = ci.classify(events, ask_model=lambda t: {})
    assert events[0].kind == "other" and unplaced == events
    report = ci.ImportReport(unplaced=["Treffen mit Lukas"])
    text = ci.describe("x", events, report)
    assert "Treffen mit Lukas" in text and "не понял" in text.lower()


def test_a_crashing_model_does_not_lose_the_import():
    def boom(titles):
        raise RuntimeError("503")
    events = [_ev("Treffen mit Lukas")]
    assert ci.classify(events, ask_model=boom) == events


# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------

def test_the_owners_file_lands_as_classes_with_the_project_week_left_empty(today):
    text = documents.inspect(campusnet_ics(TIMETABLE), "x.ics").text
    receipt = ci.import_calendar_text(text)

    monday = ws.study_slots(date(2026, 10, 5))
    assert [s[:2] for s in monday] == [("12:20", "14:00"), ("14:05", "15:45")], \
        "две пары в один день — две записи, а не одна затёртая"
    assert ws.get_shifts(date(2026, 10, 5)) == [], "пары — не смены"
    for d in range(19, 24):
        assert ws.day_events(date(2026, 10, d)) == [], "проектная неделя должна остаться пустой"
    assert "19.10–25.10" in receipt, "пустую неделю надо назвать, а не молча пропустить"
    assert "учёба" in receipt


def test_sending_the_same_export_again_does_not_duplicate(today):
    text = documents.inspect(campusnet_ics(TIMETABLE), "x.ics").text
    ci.import_calendar_text(text)
    ci.import_calendar_text(text)
    rows = db.query("SELECT COUNT(*) c FROM timetable WHERE date IS NOT NULL")
    assert rows[0]["c"] == len(TIMETABLE)


def test_a_newer_export_removes_a_cancelled_class_but_nothing_from_other_calendars(today):
    ci.import_calendar_text(documents.decode_text(campusnet_ics(TIMETABLE)))
    ws.add_event("Tennis", "sport", "20:00", "21:30", weekly_from=date(2026, 10, 5))
    ws.add_event("Zahnarzt", "other", "09:00", "10:00", on=date(2026, 10, 12))

    without_b1 = [e for e in TIMETABLE if "UID:b1" not in e]
    ci.import_calendar_text(documents.decode_text(campusnet_ics(without_b1)))

    titles = [r["title"] for r in ws.day_events(date(2026, 10, 12))]
    assert "Diskrete Mathematik - Vorlesung" not in titles, "отменённая пара должна уйти"
    assert "Zahnarzt" in titles and "Tennis" in titles, "чужие события не трогаем"


def test_shift_events_from_a_calendar_go_to_shifts(today):
    events = [_event("20261007", "1700", "2300", "Spätschicht", "Bar", "s1")]
    text = documents.decode_text(campusnet_ics(events)).replace("CampusNet", "Dienstplan")
    ci.import_calendar_text(text)
    shifts = ws.get_shifts(date(2026, 10, 7))
    assert [(s["start"], s["end"], s["source"]) for s in shifts] == [("17:00", "23:00", "calendar")]
    assert ws.day_events(date(2026, 10, 7)) == []


def test_screenshot_events_take_the_same_path_and_a_resent_day_replaces_itself(today):
    shot = [
        {"date": "2026-10-05", "start": "12:20", "end": "14:00",
         "title": "Diskrete Mathematik - Vorlesung", "location": "Hörsaal 2"},
        {"date": "2026-10-05", "start": "20:00", "end": "21:30", "title": "Tischtennis"},
    ]
    receipt, n = ci.import_screenshot_events(shot)
    assert n == 2 and receipt
    kinds = {r["title"]: r["kind"] for r in ws.day_events(date(2026, 10, 5))}
    assert kinds == {"Diskrete Mathematik - Vorlesung": "lecture", "Tischtennis": "sport"}

    ci.import_screenshot_events(shot)
    assert len(ws.day_events(date(2026, 10, 5))) == 2


def test_the_receipt_shows_the_weekly_pattern_not_every_event(today):
    text = documents.decode_text(campusnet_ics(TIMETABLE))
    receipt = ci.import_calendar_text(text)
    assert "пн 12:20–14:00 Diskrete Mathematik - Vorlesung ×3" in receipt
