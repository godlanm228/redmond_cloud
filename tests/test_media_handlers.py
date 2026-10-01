"""Files and albums reach the bot, and get one answer each.

01.10.2026: a calendar file (.ics) found no handler — no reply, no log line;
six screenshots sent as one album were handled one by one: six vision
requests, three week plans in a row, burnt model limits.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from handlers import multi_bot
from logic import calendar_import as ci
from logic import week_schedule as ws
from tests.test_calendar_import import TIMETABLE, campusnet_ics
from utils.time import OWNER_TZ

OWNER, CHAT = 605, -100


class FakeCoordinator:
    def __init__(self):
        self.sent = []

    async def respond_as(self, agent, chat_id, text, emoji="", fmt="plain"):
        self.sent.append((agent, text))
        return [1]

    @asynccontextmanager
    async def typing(self, name, chat_id):
        yield

    def bot_for(self, name):
        return None


class FakeFile:
    def __init__(self, raw):
        self.raw = raw

    async def download_as_bytearray(self):
        return bytearray(self.raw)


class FakeMedia:
    def __init__(self, raw, **kw):
        self.raw = raw
        self.__dict__.update(kw)

    async def get_file(self):
        return FakeFile(self.raw)


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", str(OWNER))
    monkeypatch.setenv("MAIN_CHAT_ID", str(CHAT))
    fixed = datetime(2026, 10, 1, 16, 0, tzinfo=OWNER_TZ)
    monkeypatch.setattr(ci, "now_local", lambda: fixed)
    monkeypatch.setattr(ws, "now_local", lambda: fixed)
    monkeypatch.setattr(multi_bot, "ALBUM_WAIT_SEC", 0.05)
    coordinator = FakeCoordinator()
    notes = []
    rg = SimpleNamespace(note_to_history=lambda chat, user, note, agent="": notes.append(note),
                         chat_history=lambda chat_id: [])
    context = SimpleNamespace(application=SimpleNamespace(bot_data={
        "coordinator": coordinator,
        "dispatcher": SimpleNamespace(response_generator=rg, config=SimpleNamespace()),
        "router_states": {},
    }))
    return SimpleNamespace(context=context, coordinator=coordinator, notes=notes)


def _update(**message):
    fields = dict(photo=None, document=None, caption=None, media_group_id=None, sticker=None,
                  animation=None, video=None, video_note=None, audio=None)
    msg = SimpleNamespace(**{**fields, **message})
    return SimpleNamespace(effective_user=SimpleNamespace(id=OWNER, is_bot=False),
                           effective_chat=SimpleNamespace(id=CHAT), message=msg)


def test_a_calendar_file_is_read_answered_and_stored(bot, monkeypatch):
    from logic import file_reader
    monkeypatch.setattr(file_reader, "_ask_gemini", lambda prompt, content: (
        '{"what": "расписание пар CampusNet на октябрь", "summary": "Пары пн, вт, пт.",'
        ' "key_facts": [], "schedule_items": [], "deadlines": [], "do_now": ["schedule"],'
        ' "question": "В файле только октябрь — продлить до конца семестра?"}', "fake"))
    raw = campusnet_ics(TIMETABLE)
    doc = FakeMedia(raw, file_name="832849531700533.ics", mime_type="text/calendar",
                    file_size=len(raw))
    asyncio.run(multi_bot.redmond_document_handler(_update(document=doc), bot.context))

    (agent, text), (asker, question) = bot.coordinator.sent
    assert agent == "Redmond" and "расписание пар CampusNet" in text and "Записал" in text
    assert asker == "Iris" and "продлить" in question, "вопрос задаёт тот, кто выполнит ответ"
    assert bot.context.application.bot_data["router_states"][CHAT].last_agent_name == "Iris"
    assert ws.study_slots(date(2026, 10, 5)), "пары должны лечь в расписание"
    assert bot.notes and "CampusNet" in bot.notes[0], "чтение — в историю, для «что это за пункты»"


def test_an_executable_named_like_a_calendar_is_refused_out_loud(bot):
    raw = b"MZ" + b"\x00" * 100
    doc = FakeMedia(raw, file_name="stundenplan.ics", mime_type="text/calendar",
                    file_size=len(raw))
    asyncio.run(multi_bot.redmond_document_handler(_update(document=doc), bot.context))
    assert "не открываю" in bot.coordinator.sent[0][1]


def test_an_album_is_one_vision_request_and_one_receipt(bot, monkeypatch):
    seen = []

    def analyze(images):
        seen.append(len(images))
        return [{"type": "calendar", "error": "", "description": f"скрин {i}",
                 "events": [{"date": "2026-10-0%d" % (5 + i), "start": "12:20", "end": "14:00",
                             "title": "Diskrete Mathematik - Vorlesung"}]}
                for i in range(len(images))]

    import logic.vision
    import utils.vision_archive
    monkeypatch.setattr(logic.vision, "analyze_images", analyze)
    monkeypatch.setattr(utils.vision_archive, "save", lambda *a, **kw: None)

    async def send_album():
        for i in range(3):
            photo = [FakeMedia(b"jpeg%d" % i)]
            await multi_bot.redmond_photo_handler(
                _update(photo=photo, media_group_id="album-1"), bot.context)
        await asyncio.sleep(0.3)

    asyncio.run(send_album())
    assert seen == [3], "альбом — один запрос к зрению на все фото"
    receipts = [t for a, t in bot.coordinator.sent if "📅" in t]
    assert len(receipts) == 1
    for d in (5, 6, 7):
        assert ws.study_slots(date(2026, 10, d)), f"пара {d}.10 должна быть в расписании"
        assert ws.get_shifts(date(2026, 10, d)) == [], "и не должна быть сменой"


def test_a_sticker_is_logged_not_answered_and_a_video_gets_an_honest_reply(bot, caplog):
    import logging
    caplog.set_level(logging.INFO)
    asyncio.run(multi_bot.redmond_other_media_handler(_update(sticker=object()), bot.context))
    assert bot.coordinator.sent == []
    assert "стикер" in caplog.text
    asyncio.run(multi_bot.redmond_other_media_handler(_update(video=object()), bot.context))
    assert "не разбираю" in bot.coordinator.sent[0][1]
