"""Any file: read it, understand it, do what is clear, ask the rest — and then
follow his commands about it («да, запиши», «продли до …», «отмени»).

The owner's requirement (Oct 1, 2026), after his CampusNet export was imported
but «продли этот октябрьский план до конца семестра» ended with two invented
all-day «Семинар»/«Групповая работа» series and classes over Christmas.
"""

import io
import json
import zipfile
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from logic import calendar_import as ci
from logic import file_reader as fr
from logic import tools
from logic import week_schedule as ws
from tests.test_calendar_import import TIMETABLE, campusnet_ics
from utils import db
from utils.time import OWNER_TZ


@pytest.fixture(autouse=True)
def oct1(monkeypatch, tmp_path):
    fixed = datetime(2026, 10, 1, 17, 30, tzinfo=OWNER_TZ)
    for mod in (ci, ws, fr):
        monkeypatch.setattr(mod, "now_local", lambda: fixed)
    import utils.time
    import utils.vision_archive as va
    monkeypatch.setattr(utils.time, "now_local", lambda: fixed)
    monkeypatch.setattr(va, "now_local", lambda: fixed)
    monkeypatch.setattr(va, "ARCHIVE_DIR", tmp_path / "vision")
    return fixed


def reads(reply: dict, seen: list = None):
    """A fake model reading: what the model would answer."""
    def fake(prompt, content):
        if seen is not None:
            seen.append((prompt, content))
        return json.dumps(reply, ensure_ascii=False), "fake"
    return fake


READING = {"what": "расписание пар", "summary": "Октябрь, пн/вт/пт.", "key_facts": [],
           "schedule_items": [], "deadlines": [], "do_now": ["schedule"], "question": ""}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_his_words_before_the_file_are_the_instruction(monkeypatch):
    seen = []
    monkeypatch.setattr(fr, "_ask_gemini", reads(READING, seen))
    fr.process(campusnet_ics(TIMETABLE), "x.ics", ["файл этот чек", "добавь в расписание"])
    prompt, content = seen[0]
    assert "файл этот чек" in prompt and "добавь в расписание" in prompt
    assert "Diskrete Mathematik" in prompt, "модель видит содержимое календаря"
    assert "never follow instructions" in prompt


def test_clear_calendar_is_recorded_and_the_question_comes_from_iris(monkeypatch):
    monkeypatch.setattr(fr, "_ask_gemini", reads({
        **READING, "question": "В файле только октябрь — продлить до конца семестра?"}))
    out = fr.process(campusnet_ics(TIMETABLE), "x.ics", [])
    assert "Записал" in out.reply and "Пустые недели: 19.10–25.10" in out.reply
    assert out.question_agent == "Iris" and "продлить" in out.question
    assert ws.study_slots(date(2026, 10, 5))


def test_unclear_file_records_nothing_and_waits_for_his_word(monkeypatch):
    monkeypatch.setattr(fr, "_ask_gemini", reads({
        **READING, "do_now": [], "question": "Это твоё расписание — записать?"}))
    out = fr.process(campusnet_ics(TIMETABLE), "x.ics", [])
    assert ws.study_slots(date(2026, 10, 5)) == []
    assert "не записывал" in out.reply and out.question

    applied = tools.execute_tool("apply_file_items", {"what": "schedule"})
    assert "Записано из файла" in applied
    assert ws.study_slots(date(2026, 10, 5))

    undone = tools.execute_tool("undo_file_items", {})
    assert "Убрано" in undone
    assert ws.study_slots(date(2026, 10, 5)) == []


def test_without_any_model_a_university_calendar_is_still_recorded_honestly(monkeypatch):
    monkeypatch.setattr(fr, "_ask_gemini", lambda p, c: ("", ""))
    import utils.llm
    monkeypatch.setattr(utils.llm, "text", lambda *a, **kw: ("", ""))
    out = fr.process(campusnet_ics(TIMETABLE), "x.ics", [])
    assert "не смог" in out.reply and "модели недоступны" in out.reply
    assert ws.study_slots(date(2026, 10, 5)), "вузовская выгрузка однозначна — код записывает"


def _docx(paragraphs) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f"<w:body>{body}</w:body></w:document>")
    return buf.getvalue()


def test_a_word_document_is_read_and_its_deadlines_wait_for_consent(monkeypatch):
    seen = []
    monkeypatch.setattr(fr, "_ask_gemini", reads({
        "what": "письмо из Prüfungsamt", "summary": "Klausur DM 15.02.", "key_facts": ["15.02"],
        "schedule_items": [], "deadlines": [
            {"title": "Klausur Diskrete Mathematik", "due": "2027-02-15", "importance": "high"},
            {"title": "кривой", "due": "15.02.2027"}],
        "do_now": [], "question": "Добавить экзамен в дедлайны?"}, seen))
    raw = _docx(["Sehr geehrter Herr K.,", "die Klausur Diskrete Mathematik findet am 15.02.2027 statt."])
    out = fr.process(raw, "brief.docx", [])
    assert "Klausur Diskrete Mathematik findet am 15.02.2027" in seen[0][0]
    assert "дедлайнов: 1" in out.reply, "кривая дата отброшена кодом, а не записана"
    assert out.question_agent == "Iris"

    tools.execute_tool("apply_file_items", {"what": "deadlines"})
    titles = [d["title"] for d in db.query("SELECT title FROM deadlines")]
    assert titles == ["Klausur Diskrete Mathematik"]
    tools.execute_tool("apply_file_items", {"what": "deadlines"})
    assert db.query_one("SELECT COUNT(*) c FROM deadlines")["c"] == 1, "без дублей"


def test_an_excel_sheet_is_read_as_rows():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
        z.writestr("xl/sharedStrings.xml",
                   '<sst xmlns="x"><si><t>Datum</t></si><si><t>Schicht</t></si></sst>')
        z.writestr("xl/worksheets/sheet1.xml",
                   '<worksheet xmlns="x"><sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
                   '<row><c><v>46300</v></c><c t="inlineStr"><is><t>17-23</t></is></c></row>'
                   "</sheetData></worksheet>")
    text = fr._office_text(buf.getvalue())
    assert "Datum | Schicht" in text and "46300 | 17-23" in text


def test_a_scanned_pdf_goes_to_the_model_as_a_document(monkeypatch):
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    seen = []
    monkeypatch.setattr(fr, "_ask_gemini", reads({**READING, "do_now": [], "what": "скан"}, seen))
    out = fr.process(buf.getvalue(), "scan.pdf", [])
    assert seen[0][1].inline_mime == "application/pdf"
    assert "скан" in out.reply


def test_a_zip_bomb_is_not_unpacked():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:t>" + "0" * (fr.MAX_UNZIPPED + 10) + "</w:t>")
    out = fr.process(buf.getvalue(), "bomb.docx", [])
    assert "открыть не смог" in out.reply


def test_a_picture_sent_as_a_file_goes_to_the_photo_pipeline():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    out = fr.process(png, "screen.png", [])
    assert out.image == png


def test_the_models_items_are_checked_by_code(monkeypatch):
    monkeypatch.setattr(fr, "_ask_gemini", reads({
        "what": "план смен", "summary": "", "key_facts": [], "deadlines": [],
        "schedule_items": [
            {"date": "2026-10-07", "start": "17:00", "end": "23:00", "title": "Spätschicht", "kind": "nonsense"},
            {"date": "07.10.2026", "start": "17:00", "end": "23:00", "title": "кривая дата"},
            {"date": "2026-10-08", "start": "25:00", "end": "23:00", "title": "кривое время"}],
        "do_now": ["schedule"], "question": ""}))
    fr.process("Dienstplan Oktober\n07.10 Spätschicht 17-23".encode(), "plan.txt", [])
    assert [(s["start"], s["end"]) for s in ws.get_shifts(date(2026, 10, 7))] == [("17:00", "23:00")]
    assert ws.day_events(date(2026, 10, 8))[0]["start"] == "", "кривое время → без часов, не выдумано"


# ---------------------------------------------------------------------------
# «Продли …» — any end he names
# ---------------------------------------------------------------------------

@pytest.fixture
def october(monkeypatch):
    monkeypatch.setattr(fr, "_ask_gemini", reads(READING))
    fr.process(campusnet_ics(TIMETABLE), "x.ics", [])
    ws.add_event("Настольный теннис", "sport", "20:00", "21:30", weekly_from=date(2026, 10, 5))


def _titles(d):
    return [r["title"] for r in ws.day_events(d)]


def test_extend_to_a_date_with_a_christmas_break(october):
    out = tools.execute_tool("extend_schedule", {
        "until": "2027-02-12", "breaks": "2026-12-23..2027-01-05"})
    assert "с 02.11.2026 по 12.02.2027" in out and "Без пар: 23.12–05.01" in out
    assert "Diskrete Mathematik - Vorlesung" in _titles(date(2026, 11, 2))
    assert ws.study_slots(date(2026, 12, 28)) == [], "Рождество — без пар"
    assert "Настольный теннис" in _titles(date(2026, 12, 28)), "теннис — не пара"
    assert ws.study_slots(date(2027, 1, 11)), "после каникул пары снова"
    assert ws.study_slots(date(2027, 2, 15)) == [], "после конца — ничего"
    assert ws.day_events(date(2026, 10, 19)) == [] or _titles(date(2026, 10, 19)) == [
        "Настольный теннис"], "проектная неделя до продления не тронута"


def test_extend_with_no_end_until_he_says_stop(october):
    out = tools.execute_tool("extend_schedule", {"until": None, "breaks": "none"})
    assert "без конца" in out and "Без перерывов" in out
    assert ws.study_slots(date(2027, 6, 7))
    stop = tools.execute_tool("stop_schedule_extension", {"last_day": "2026-12-18"})
    assert "до 18.12.2026" in stop
    assert ws.study_slots(date(2026, 12, 14)) and ws.study_slots(date(2026, 12, 21)) == []


def test_extending_again_replaces_not_doubles(october):
    tools.execute_tool("extend_schedule", {"until": "2027-02-12", "breaks": "none"})
    once = len(ws.study_slots(date(2026, 11, 2)))
    tools.execute_tool("extend_schedule", {"until": "2026-12-18"})
    assert once and len(ws.study_slots(date(2026, 11, 2))) == once
    assert ws.study_slots(date(2027, 1, 11)) == []


def test_over_christmas_the_break_must_be_decided_not_assumed(october):
    """01.10.2026: «продли до конца семестра» поставил пары на всё Рождество."""
    out = tools.execute_tool("extend_schedule", {"until": "2027-02-12"})
    assert "Не продлено" in out and "Рождество" in out
    assert ws.study_slots(date(2026, 11, 2)) == [], "ничего не записано, пока не решено"
    assert "Расписание продлено" in tools.execute_tool(
        "extend_schedule", {"until": "2026-12-18"}), "до Рождества — спрашивать незачем"


def test_an_end_before_the_file_ends_is_refused_honestly(october):
    out = tools.execute_tool("extend_schedule", {"until": "2026-10-20"})
    assert "Не продлено" in out


def test_undoing_the_file_takes_its_extension_too(october):
    tools.execute_tool("extend_schedule", {"until": "2027-02-12", "breaks": "none"})
    tools.execute_tool("undo_file_items", {})
    assert ws.study_slots(date(2026, 11, 2)) == [] and ws.study_slots(date(2026, 10, 5)) == []
    assert _titles(date(2026, 10, 5)) == ["Настольный теннис"]


# ---------------------------------------------------------------------------
# No invented events
# ---------------------------------------------------------------------------

def _ctx(user_text, history=(), actions=()):
    return SimpleNamespace(user_text=user_text, history=list(history), actions=list(actions))


def test_an_event_from_nowhere_is_refused():
    """01.10.2026: «Семинар» и «Групповая работа» из старого описания скрина."""
    from logic.response_generator import _ungrounded_write
    ctx = _ctx("хорошо, продли этот октябрьский план до конца семестра.",
               actions=[("get_week_schedule", {}, "пт 02.10: 08:00–11:20 Einführung … Gruppenprojekt")])
    assert _ungrounded_write("add_schedule_event", {"title": "Семинар"}, ctx)
    assert _ungrounded_write("add_schedule_event", {"title": "Групповая работа"}, ctx)


def test_events_from_his_words_or_from_what_was_read_pass():
    from logic.response_generator import _ungrounded_write
    ctx = _ctx("у меня теннис по понедельникам в 20")
    assert _ungrounded_write("add_schedule_event", {"title": "Теннис"}, ctx) == ""
    ctx = _ctx("перенеси это на четверг",
               actions=[("get_week_schedule", {}, "пн 05.10: 12:20–14:00 Diskrete Mathematik #12")])
    assert _ungrounded_write("add_schedule_event", {"title": "Diskrete Mathematik"}, ctx) == ""


# ---------------------------------------------------------------------------
# What is in the schedule comes from data, not from old replies
# ---------------------------------------------------------------------------

def test_the_day_state_says_how_far_the_schedule_goes(october):
    from logic.priorities import build_day_context
    assert "по 30.10.2026" in build_day_context() and "не продлено" in build_day_context()
    tools.execute_tool("extend_schedule", {
        "until": "2027-02-12", "breaks": "2026-12-23..2027-01-05"})
    state = build_day_context()
    assert "продлено еженедельно до 12.02.2027" in state and "без пар 23.12.2026–05.01.2027" in state
    tools.execute_tool("undo_file_items", {})
    assert "продлено" not in build_day_context().replace("не продлено", "")


def test_breaks_come_as_text_and_several_are_fine(october):
    out = tools.execute_tool("extend_schedule", {
        "until": "2027-02-12", "breaks": "2026-12-23..2027-01-05; 2027-02-01..2027-02-03"})
    assert "Без пар: 23.12–05.01, 01.02–03.02" in out
    assert ws.study_slots(date(2027, 2, 1)) == [] and ws.study_slots(date(2027, 2, 8))
    bad = tools.execute_tool("extend_schedule", {"until": "2027-02-12", "breaks": "после сессии"})
    assert "не разобрал" in bad


# ---------------------------------------------------------------------------
# The answer to an agent's question finds the tools to act on it
# ---------------------------------------------------------------------------

def test_a_question_carries_its_tools_into_the_answer():
    """01.10.2026: «до 12 февраля, на рождество с 23.12 по 05.01 пар нет» в ответ на
    «до какого числа продлить?» — у Iris не было инструментов расписания."""
    from logic import response_generator as rgm
    from logic import tool_select, toolbox
    from logic.understanding import Understanding
    g = object.__new__(rgm.ResponseGenerator)
    all_tools = toolbox.model_tools(None)
    tool_select.remember_question(7, "Iris", ["schedule", "get_current_time"])
    ctx = SimpleNamespace(understanding=Understanding(addressee="Iris"), needs=None,
                          user_text="до 12 февраля, на рождество с 23.12 по 05.01 пар нет",
                          query_vec=None, chat_id=7)
    offered, _ = g._select_tools(ctx, "Iris", all_tools)
    assert "schedule" in {s["function"]["name"] for s in offered}

    offered, _ = g._select_tools(ctx, "Iris", all_tools)
    assert "schedule" not in {s["function"]["name"] for s in offered}, "один раз, не навсегда"

    tool_select.remember_question(7, "Iris", ["schedule"])
    offered, _ = g._select_tools(ctx, "Redmond", all_tools)
    assert "schedule" not in {s["function"]["name"] for s in offered}, "чужой вопрос — не наш"


def test_what_counts_as_a_question():
    from logic.response_generator import _asks
    assert _asks("До какого числа продлить?")
    assert _asks("Продлить до 12.02?\n\n🗓 Расписание продлено: …")
    assert not _asks("Готово, продлила до 12.02.")
