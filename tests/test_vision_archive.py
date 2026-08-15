"""Архив фото и разборов зрения.

12.08.2026 на вопрос «почему скрин распознался неверно» ответить было нечем:
картинка не сохранялась, в логе была одна строка «type=shift_schedule,
shifts=5». Эти тесты фиксируют, что теперь сохраняется и файл, и полный ответ
модели, и что из него в итоге записали.
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

from logic.tools import execute_tool
from utils import db, vision_archive

SCHEDULE = {
    "type": "shift_schedule",
    "shifts": [{"date": "2026-08-13", "start": "17:00", "end": "23:00"}],
    "description": "Скриншот приложения с графиком смен на неделю, бар и Spüle",
}
FOOD = {
    "type": "food", "food_kind": "meal", "dish": "Салат Оливье",
    "kcal_low": 250, "kcal_high": 350,
    "description": "Остатки салата Оливье в металлической миске",
}
OTHER = {
    "type": "other",
    "description": "Скриншот профиля: рабочее время 24 ч 23 мин, 338,93 евро",
}


class ArchiveBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()


class SaveTests(ArchiveBase):
    def test_file_and_record_are_created(self):
        rec_id = vision_archive.save(b"jpeg-bytes", SCHEDULE, chat_id=1, model="gemini")
        self.assertIsNotNone(rec_id)
        rec = vision_archive.get(rec_id)
        self.assertTrue(rec["exists"])
        self.assertEqual(rec["kind"], "shift_schedule")
        self.assertEqual(rec["model"], "gemini")

    def test_raw_answer_is_kept_whole(self):
        """Главное, чего не хватало: полный ответ модели, а не тип одной строкой."""
        rec_id = vision_archive.save(b"bytes", SCHEDULE)
        raw = vision_archive.get(rec_id)["raw"]
        self.assertEqual(raw["shifts"], SCHEDULE["shifts"])
        self.assertEqual(raw["description"], SCHEDULE["description"])

    def test_same_photo_twice_is_not_duplicated(self):
        first = vision_archive.save(b"same", FOOD)
        second = vision_archive.save(b"same", FOOD)
        self.assertEqual(first, second)
        self.assertEqual(vision_archive.stats()["records"], 1)

    def test_different_photos_are_separate(self):
        vision_archive.save(b"one", FOOD)
        vision_archive.save(b"two", SCHEDULE)
        self.assertEqual(vision_archive.stats()["records"], 2)

    def test_empty_bytes_are_ignored(self):
        self.assertIsNone(vision_archive.save(b"", FOOD))

    def test_failure_never_raises(self):
        """Архив вспомогательный — разбор фото из-за него падать не должен."""
        original = vision_archive._path_for
        vision_archive._path_for = lambda *a: (_ for _ in ()).throw(OSError("диск"))
        try:
            self.assertIsNone(vision_archive.save(b"bytes", FOOD))
        finally:
            vision_archive._path_for = original

    def test_applied_is_recorded(self):
        rec_id = vision_archive.save(b"bytes", SCHEDULE)
        vision_archive.set_applied(rec_id, "смен сохранено: 5, спорных: 0")
        self.assertIn("смен сохранено: 5", vision_archive.get(rec_id)["applied"])

    def test_applied_on_missing_id_is_safe(self):
        vision_archive.set_applied(None, "что-то")
        vision_archive.set_applied(999999, "что-то")


class TagTests(ArchiveBase):
    def test_schedule_gets_work_tags(self):
        rec = vision_archive.get(vision_archive.save(b"a", SCHEDULE))
        self.assertIn("график", rec["tags"])
        self.assertIn("смены", rec["tags"])

    def test_food_tags_include_dish(self):
        rec = vision_archive.get(vision_archive.save(b"b", FOOD))
        self.assertIn("Салат Оливье", rec["tags"])
        self.assertIn("еда", rec["tags"])

    def test_groceries_items_become_tags(self):
        result = {"type": "food", "food_kind": "groceries",
                  "items": ["Wok Mix", "Овощи"], "description": "пакет продуктов"}
        rec = vision_archive.get(vision_archive.save(b"c", result))
        self.assertIn("Wok Mix", rec["tags"])

    def test_tags_are_deduplicated(self):
        rec = vision_archive.get(vision_archive.save(b"d", SCHEDULE))
        lowered = [t.lower() for t in rec["tags"]]
        self.assertEqual(len(lowered), len(set(lowered)))


class SearchTests(ArchiveBase):
    def setUp(self):
        super().setUp()
        vision_archive.save(b"one", SCHEDULE)
        vision_archive.save(b"two", FOOD)
        vision_archive.save(b"three", OTHER)

    def test_find_schedule_by_word(self):
        found = vision_archive.search("график")
        self.assertTrue(found)
        self.assertEqual(found[0]["kind"], "shift_schedule")

    def test_find_food_by_dish(self):
        found = vision_archive.search("оливье")
        self.assertTrue(found)
        self.assertEqual(found[0]["kind"], "food")

    def test_find_by_description_words(self):
        found = vision_archive.search("евро заработок 338")
        self.assertTrue(found)
        self.assertEqual(found[0]["kind"], "other")

    def test_label_is_searchable(self):
        rec_id = vision_archive.save(b"four", OTHER)
        vision_archive.set_label(rec_id, "зарплата за август")
        found = vision_archive.search("зарплата")
        self.assertTrue(any(r["id"] == rec_id for r in found))

    def test_nothing_found_gives_empty(self):
        self.assertEqual(vision_archive.search("зубоврачебный кабинет"), [])

    def test_empty_query_returns_recent(self):
        self.assertEqual(len(vision_archive.search("")), 3)

    def test_limit_is_respected(self):
        self.assertLessEqual(len(vision_archive.search("скриншот", limit=1)), 1)


class RetentionTests(ArchiveBase):
    def test_old_original_is_dropped_but_record_stays(self):
        rec_id = vision_archive.save(b"old", SCHEDULE)
        db.execute("UPDATE vision_results SET ts='2020-01-01T10:00' WHERE id=?", (rec_id,))
        vision_archive.enforce_limits()
        rec = vision_archive.get(rec_id)
        self.assertFalse(rec["exists"])
        self.assertEqual(rec["raw"]["shifts"], SCHEDULE["shifts"])

    def test_fresh_original_is_kept(self):
        rec_id = vision_archive.save(b"fresh", FOOD)
        vision_archive.enforce_limits()
        self.assertTrue(vision_archive.get(rec_id)["exists"])

    def test_directory_cap_evicts_oldest(self):
        original = vision_archive.MAX_DIR_BYTES
        vision_archive.MAX_DIR_BYTES = 10
        try:
            first = vision_archive.save(b"x" * 50, FOOD)
            vision_archive.save(b"y" * 50, SCHEDULE)
            self.assertFalse(vision_archive.get(first)["exists"])
        finally:
            vision_archive.MAX_DIR_BYTES = original

    def test_stats_report_files_and_bytes(self):
        vision_archive.save(b"12345", FOOD)
        stats = vision_archive.stats()
        self.assertEqual(stats["records"], 1)
        self.assertEqual(stats["files"], 1)
        self.assertGreater(stats["bytes"], 0)


class FindPhotoToolTests(ArchiveBase):
    def test_tool_reports_what_was_recognised(self):
        rec_id = vision_archive.save(b"one", SCHEDULE)
        vision_archive.set_applied(rec_id, "смен сохранено: 5")
        reply = execute_tool("find_photo", {"query": "график смен"})
        self.assertIn("график", reply.lower())
        self.assertIn("смен сохранено: 5", reply)

    def test_tool_says_when_nothing_found(self):
        reply = execute_tool("find_photo", {"query": "квантовая физика"})
        self.assertIn("ничего нет", reply)

    def test_tool_marks_deleted_originals(self):
        rec_id = vision_archive.save(b"old", FOOD)
        db.execute("UPDATE vision_results SET ts='2020-01-01T10:00' WHERE id=?", (rec_id,))
        vision_archive.enforce_limits()
        reply = execute_tool("find_photo", {"query": "оливье"})
        self.assertIn("файл уже удалён", reply)

    def test_tool_limit_is_clamped(self):
        for i in range(6):
            vision_archive.save(f"photo{i}".encode(), FOOD)
        reply = execute_tool("find_photo", {"query": "оливье", "limit": 99})
        self.assertLessEqual(len(reply.strip().split("\n")) - 1, 5)

    def test_tool_survives_bad_limit(self):
        vision_archive.save(b"one", FOOD)
        reply = execute_tool("find_photo", {"query": "оливье", "limit": "много"})
        self.assertIn("Нашла", reply)


if __name__ == "__main__":
    unittest.main()
