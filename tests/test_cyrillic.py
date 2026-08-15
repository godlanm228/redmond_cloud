"""Кириллица должна работать везде, где работает латиница.

Повод. В архиве фото поиск «оливье» не находил «Оливье»: встроенный SQLite
LOWER() опускает только ASCII. То есть по-русски поиск не работал вообще —
а вся переписка, дневник и описания фото здесь русские.

Возник вопрос, не хранить ли данные по-английски с переводом на входе и
выходе. Эти тесты отвечают на него фактами: показывают, что кириллица
проходит весь путь без потерь, и фиксируют каждое место, где регистр или
кодировка могли бы её сломать. Перевод на хранении добавил бы к каждой записи
две лишние LLM-операции и ровно тот риск искажения смысла, которого мы
избегаем.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic import coach_storage
from logic.tools import execute_tool
from logic.week_schedule import apply_shifts, get_shift, shift_history
from utils import db, vision_archive
from utils.memory import MemoryStore

# Реальные строки из переписки и данных Влада.
REAL_TEXTS = [
    "Поел: Салат Оливье (дом)",
    "Работаю с 12 до 16 в экстраблате",
    "Замороженные овощи Rewe Beste Wahl Gemüspfanne Französische Art",
    "Проснулся — первое сообщение в 13:37",
    "Матан: 6 открытых тестов (жёсткий предел 28.06)",
    "Смена 2026-07-29 с 15:00 до 23:30",
]


class DiaryCyrillicTests(unittest.TestCase):
    def test_text_survives_roundtrip_exactly(self):
        for text in REAL_TEXTS:
            coach_storage.add_diary_entry(text, tags=["питание"])
        stored = [e["text"] for e in coach_storage.read_diary(last_n=50)]
        self.assertEqual(stored, REAL_TEXTS)

    def test_german_umlauts_and_eszett_survive(self):
        coach_storage.add_diary_entry("Gemüspfanne Französische Art, Spüle, groß")
        self.assertEqual(coach_storage.read_diary(last_n=1)[0]["text"],
                         "Gemüspfanne Französische Art, Spüle, groß")

    def test_yo_is_not_normalised_away(self):
        """«ё» и «е» — разные буквы, подмена меняет смысл («всё» / «все»)."""
        coach_storage.add_diary_entry("Всё сделал, все довольны")
        self.assertEqual(coach_storage.read_diary(last_n=1)[0]["text"],
                         "Всё сделал, все довольны")

    def test_cyrillic_tags_filter_correctly(self):
        coach_storage.add_diary_entry("поел борщ", tags=["питание"])
        coach_storage.add_diary_entry("сходил в зал", tags=["спорт"])
        found = coach_storage.read_diary(tag="питание")
        self.assertEqual(len(found), 1)
        self.assertIn("борщ", found[0]["text"])

    def test_today_tags_are_cyrillic(self):
        coach_storage.add_diary_entry("поел", tags=["питание", "работа"])
        self.assertEqual(coach_storage.today_tags(), {"питание", "работа"})

    def test_case_insensitive_duplicate_detection_works_in_russian(self):
        """Дедуп сравнивает через .lower() Python — он Unicode-aware."""
        first = coach_storage.add_diary_entry("Поел Салат Оливье")
        again = coach_storage.add_diary_entry("поел салат оливье")
        self.assertEqual(again["id"], first["id"])

    def test_structured_data_keeps_cyrillic(self):
        coach_storage.add_diary_entry(
            "Поел", tags=["питание"],
            data={"dish": "Салат Оливье", "place": "дом"})
        self.assertEqual(coach_storage.read_diary(last_n=1)[0]["data"]["dish"],
                         "Салат Оливье")


class GoalsAndDeadlinesCyrillicTests(unittest.TestCase):
    def test_goal_title_and_why(self):
        g = coach_storage.add_goal("Лечь спать до 23:59",
                                   why="Обеспечить достаточный сон")
        self.assertEqual(coach_storage.list_goals()[0]["title"], g["title"])
        self.assertEqual(coach_storage.list_goals()[0]["why"], g["why"])

    def test_progress_note_in_russian(self):
        g = coach_storage.add_goal("Цель")
        done = coach_storage.mark_goal_done(g["id"], note="Закрыл досрочно, всё ок")
        self.assertEqual(done["progress_log"][0]["note"], "Закрыл досрочно, всё ок")

    def test_deadline_title_and_note(self):
        d = coach_storage.add_deadline("Матан: тест-допуск к Klausur", "2026-09-01")
        self.assertEqual(coach_storage.list_deadlines()[0]["title"], d["title"])

    def test_pantry_keeps_long_mixed_names(self):
        item = "Замороженные овощи Rewe Beste Wahl Gemüspfanne Französische Art"
        coach_storage.pantry_update(add=[item])
        self.assertEqual(coach_storage.get_pantry()["items"], [item])

    def test_pantry_dedup_is_case_insensitive_in_russian(self):
        coach_storage.pantry_update(add=["Салат Оливье"])
        coach_storage.pantry_update(add=["САЛАТ ОЛИВЬЕ"])
        self.assertEqual(len(coach_storage.get_pantry()["items"]), 1)

    def test_pantry_remove_matches_different_case(self):
        coach_storage.pantry_update(add=["Овощи"])
        coach_storage.pantry_update(remove=["ОВОЩИ"])
        self.assertEqual(coach_storage.get_pantry()["items"], [])


class ShiftCyrillicTests(unittest.TestCase):
    def test_note_and_reason_are_readable(self):
        apply_shifts([{"date": "2026-09-10", "start": "17:00", "end": "23:00",
                       "source": "text", "note": "не иду, заболел"}])
        note = db.query_one("SELECT note FROM shifts WHERE date='2026-09-10'")["note"]
        self.assertEqual(note, "не иду, заболел")

    def test_conflict_reason_is_russian_and_intact(self):
        apply_shifts([{"date": "2026-09-11", "start": "16:00", "end": "23:00",
                       "source": "text"}])
        apply_shifts([{"date": "2026-09-11", "start": "18:00", "end": "23:30",
                       "source": "photo"}])
        reasons = [e["reason"] for e in shift_history("2026-09-11") if e["reason"]]
        self.assertTrue(any("расходится" in r for r in reasons))


class VisionArchiveCyrillicTests(unittest.TestCase):
    """Именно здесь и сломалось: LOWER() не опускал «Оливье»."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()
        vision_archive.save(b"one", {
            "type": "food", "food_kind": "meal", "dish": "Салат Оливье",
            "description": "Остатки салата Оливье в металлической миске"})
        vision_archive.save(b"two", {
            "type": "shift_schedule", "shifts": [{"date": "2026-08-13"}],
            "description": "График смен на неделю, бар и Spüle"})

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_lowercase_query_finds_capitalised_word(self):
        self.assertTrue(vision_archive.search("оливье"))

    def test_uppercase_query_finds_it_too(self):
        self.assertTrue(vision_archive.search("ОЛИВЬЕ"))

    def test_mixed_case_query_works(self):
        self.assertTrue(vision_archive.search("ОлИвЬе"))

    def test_german_word_in_description_is_searchable(self):
        self.assertTrue(vision_archive.search("spüle"))

    def test_russian_label_is_searchable_in_any_case(self):
        rec = vision_archive.recent(1)[0]
        vision_archive.set_label(rec["id"], "График Августа")
        self.assertTrue(vision_archive.search("график августа"))

    def test_sqlite_builtin_lower_is_indeed_ascii_only(self):
        """Фиксируем ПРИЧИНУ бага, чтобы её не «починили» обратно."""
        row = db.query_one("SELECT LOWER('Оливье') a, LOWER('OLIVIER') b")
        self.assertEqual(row["a"], "Оливье")   # не опустилось
        self.assertEqual(row["b"], "olivier")  # ASCII опустилось

    def test_our_normalisation_handles_it(self):
        self.assertEqual(vision_archive._search_text("Оливье", ["График"], "Метка"),
                         "оливье график метка")


class MemorySearchCyrillicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = MemoryStore(os.path.join(self.tmp.name, "m.sqlite"),
                               vector_search=False)
        for text in REAL_TEXTS:
            self.mem.add(text, "Поняла, записала")

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_fts_finds_capitalised_word_by_lowercase_query(self):
        self.assertTrue(self.mem.search("оливье"))

    def test_fts_prefix_matches_inflected_russian(self):
        self.assertTrue(self.mem.search("экстраблат"))

    def test_like_fallback_is_case_insensitive_in_russian(self):
        """Третий эшелон тоже обязан работать: он включается на сборках без FTS5."""
        hits = self.mem._search_like("оливье", 5)
        self.assertTrue(hits)
        self.assertIn("Оливье", hits[0]["user"])

    def test_like_fallback_finds_uppercase_query(self):
        self.assertTrue(self.mem._search_like("ОЛИВЬЕ", 5))

    def test_german_text_is_searchable(self):
        self.assertTrue(self.mem.search("Gemüspfanne"))


class ToolReplyCyrillicTests(unittest.TestCase):
    """Ответы tools уходят прямо в чат — там кириллица должна быть читаемой."""

    def test_find_photo_reply_is_readable(self):
        tmp = tempfile.TemporaryDirectory()
        old = os.getcwd()
        os.chdir(tmp.name)
        Path("data").mkdir()
        try:
            vision_archive.save(b"one", {"type": "food", "dish": "Салат Оливье",
                                         "description": "Салат Оливье в миске"})
            reply = execute_tool("find_photo", {"query": "оливье"})
            self.assertIn("Оливье", reply)
            self.assertNotIn("\\u", reply)  # не escape-последовательности
        finally:
            os.chdir(old)
            tmp.cleanup()

    def test_json_dumps_never_escapes_cyrillic(self):
        """ensure_ascii=False всюду: иначе в базе окажется \\u0421\\u0430..."""
        coach_storage.add_diary_entry("Поел борщ", tags=["питание"])
        raw = db.query_one("SELECT tags FROM diary ORDER BY id DESC LIMIT 1")["tags"]
        self.assertEqual(raw, '["питание"]')
        self.assertNotIn("\\u", raw)

    def test_shift_payload_json_is_readable(self):
        apply_shifts([{"date": "2026-09-12", "start": "17:00", "end": "23:00",
                       "source": "text", "note": "бар"}])
        raw = db.query_one(
            "SELECT payload FROM shift_events WHERE date='2026-09-12'")["payload"]
        self.assertIn("бар", raw)
        self.assertNotIn("\\u", raw)


if __name__ == "__main__":
    unittest.main()
