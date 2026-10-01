"""Data carries its markers: where it came from, when it was true, whether it
still is. Nothing is deleted to make it look current — it is marked.

The owner, Oct 1, 2026: «данные можно не сносить, а актуализировать либо
пометить; данные должны иметь пометки, иначе как понимать, актуальны они или
нет». What the audit found that day, each case below:
  * six deadlines closed in June–August were listed as «ПРОСРОЧЕН на 111 дн»;
  * the week plan for Aug 13–19 was handed out as the current one;
  * OpenFoodFacts answered 503 and the bot said «продукт не найден»;
  * the dossier section «05 Актуально» could not be read by section;
  * a cancelled shift left «Смена 13.10» in the diary.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from logic import coach_storage as cs
from logic import tools
from utils import db
from utils.time import OWNER_TZ

NOW = datetime(2026, 10, 1, 18, 0, tzinfo=OWNER_TZ)


@pytest.fixture(autouse=True)
def oct1(monkeypatch):
    import logic.priorities
    import utils.time
    for mod in (cs, logic.priorities, utils.time):
        monkeypatch.setattr(mod, "now_local", lambda: NOW)
    return NOW


# ---------------------------------------------------------------------------
# Deadlines
# ---------------------------------------------------------------------------

def test_closed_deadlines_are_not_shown_as_overdue():
    d = cs.add_deadline("Тест Projektmanagement", "2026-06-12", "high")
    cs.mark_deadline_done(d["id"])
    out = tools.execute_tool("list_deadlines", {})
    assert "ПРОСРОЧЕН" not in out and "Открытых дедлайнов нет" in out
    closed = tools.execute_tool("list_deadlines", {"include_done": True})
    assert "✓ закрыт" in closed and "ПРОСРОЧЕН" not in closed


def test_long_overdue_open_deadline_is_marked_stale_and_leaves_priorities():
    from logic.priorities import build_priorities_block, top_priorities
    old = cs.add_deadline("Экзамен", "2026-07-31", "high")
    fresh = cs.add_deadline("Klausur Diskrete Mathematik", "2026-10-05", "high")
    out = tools.execute_tool("list_deadlines", {})
    assert "УСТАРЕЛ?" in out and "осталось 4 дн" in out
    assert [d["id"] for d in top_priorities()] == [fresh["id"]]
    block = build_priorities_block()
    assert "УСТАРЕВШИЕ ДЕДЛАЙНЫ" in block and f"#{old['id']}" in block
    assert "УСТАРЕВШИЕ" not in build_priorities_block(), "спросить раз в неделю, не каждый ответ"


def test_the_same_deadline_twice_is_one_deadline():
    first = tools.execute_tool("add_deadline", {"title": "Матан: 6 открытых тестов", "due": "2026-10-21"})
    again = tools.execute_tool("add_deadline", {"title": "матан — 6 открытых тестов!", "due": "2026-10-22"})
    assert "уже есть" in again
    assert db.query_one("SELECT COUNT(*) c FROM deadlines")["c"] == 1
    assert "#1" in first


# ---------------------------------------------------------------------------
# Week plan and pantry
# ---------------------------------------------------------------------------

def test_an_old_week_plan_is_not_handed_out_as_current():
    db.kv_set("week_plan", {"updated": "2026-08-12T21:55+02:00", "text": "Четверг, 13 августа: смена"})
    out = tools.execute_tool("get_week_plan", {})
    assert "Плана на ЭТУ неделю нет" in out and "2026-08-10" in out
    assert "13 августа" not in out, "протухший текст не отдаётся"
    tools.execute_tool("save_week_plan", {"text": "пн: пары, вечер теннис"})
    assert "пн: пары" in tools.execute_tool("get_week_plan", {})


def test_a_pantry_untouched_for_weeks_is_marked_stale():
    cs.pantry_update(add=["Wok Mix"])
    db.execute("UPDATE pantry SET added='2026-06-18'")
    out = tools.execute_tool("get_pantry", {})
    assert "УСТАРЕЛ" in out and "105 дн." in out


# ---------------------------------------------------------------------------
# Diary
# ---------------------------------------------------------------------------

def test_a_retracted_entry_stays_with_its_reason_but_counts_nowhere():
    e = cs.add_diary_entry("Смена 2026-10-13 с 15:00 до 16:35", tags=["работа"])
    cs.retract_diary_entry(e["id"], "смены не было: учебная пара, ошибка импорта 01.10")
    rows = cs.read_diary(last_n=5)
    assert rows and cs.is_retracted(rows[-1]) and "ошибка импорта" in rows[-1]["data"]["reason"]
    assert "работа" not in cs.today_tags() and cs.entries_today() == 0
    assert "работа" not in cs.last_entry_per_tag(["работа"])
    out = tools.execute_tool("read_diary", {"last_n": 5})
    assert "отозвана" in out and "ошибка импорта" in out


def test_who_wrote_an_entry_is_recorded():
    tools.execute_tool("add_diary_entry", {"text": "Сходил в зал", "tags": ["спорт"]})
    assert cs.read_diary(last_n=1)[0]["data"]["source"] == "agent"


# ---------------------------------------------------------------------------
# Food lookup: not found ≠ service down
# ---------------------------------------------------------------------------

def test_a_dead_service_is_not_reported_as_not_found(monkeypatch):
    from utils import openfoodfacts

    class Down(Exception):
        response = SimpleNamespace(status_code=503)

    def boom(*a, **kw):
        raise Down()

    monkeypatch.setattr(openfoodfacts.requests, "get", boom, raising=False)
    out = tools.execute_tool("lookup_food", {"name": "Skyr natur"})
    assert "недоступен (HTTP 503)" in out and "не найдено" not in out

    def empty(*a, **kw):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"products": []})

    monkeypatch.setattr(openfoodfacts.requests, "get", empty, raising=False)
    assert "не найдено" in tools.execute_tool("lookup_food", {"name": "Skyr natur"})


# ---------------------------------------------------------------------------
# Dossier
# ---------------------------------------------------------------------------

def test_every_dossier_section_is_readable(tmp_path, monkeypatch):
    dossier = tmp_path / "data" / "owner_dossier.md"
    dossier.parent.mkdir()
    dossier.write_text("# Профиль\n\n## 01 Профиль\nимя\n\n## 05 Актуально\nучится на WI, ищет Werkstudent\n",
                       encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert "Werkstudent" in tools.execute_tool("read_dossier_section", {"section": "current"})
    assert "Werkstudent" in tools.execute_tool("read_dossier_section", {"section": "05"})
    assert "05" in tools.execute_tool("read_dossier_section", {"section": "nope"}) or \
        "Актуально" in tools.execute_tool("read_dossier_section", {"section": "nope"})


# ---------------------------------------------------------------------------
# Search: the real source, not Google's redirect
# ---------------------------------------------------------------------------

def test_search_sources_show_where_the_fact_comes_from(monkeypatch):
    from utils import gemini
    monkeypatch.setattr(gemini.requests, "head", lambda url, **kw: SimpleNamespace(
        headers={"Location": "https://www.hochschule-ruhr-west.de/die-hrw/jahresplan"}),
        raising=False)
    url = gemini._real_url("https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ")
    assert url == "https://www.hochschule-ruhr-west.de/die-hrw/jahresplan"

    def down(url, **kw):
        raise TimeoutError("timeout")
    monkeypatch.setattr(gemini.requests, "head", down, raising=False)
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZ"
    assert gemini._real_url(redirect) == redirect, "не раскрылась — остаётся рабочая ссылка"


def test_the_readings_own_writes_are_marked_as_his_words():
    from logic import understanding
    u = understanding.Understanding(addressee="Iris", facts=[understanding.Fact(
        quote="сходил в зал", fact="Сходил в зал", when="done", topic="спорт")])
    understanding.apply(u)
    assert cs.read_diary(last_n=1)[0]["data"]["source"] == "owner"
