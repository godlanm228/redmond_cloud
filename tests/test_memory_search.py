"""Поиск по памяти: FTS5 вместо мёртвого LIKE.

Регрессия, которую ловим: `_search_like` возвращал score ровно 0.5, а
вызывающий фильтрует `score > 0.5` — память не отдавала НИЧЕГО с июня.
Векторный поиск на VM недоступен (нет faiss/sentence_transformers,
эмбеддингов 0 из 479), так что этот путь и есть боевой.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.memory import MemoryStore

DIALOGS = [
    ("Работаю с 12 до 16 в экстраблате", "Поняла, записала про работу"),
    ("Что ты зафиксировала", "Ты поел салат Оливье дома"),
    ("Скинул график смен на неделю", "Принял график, смен сохранено: 5"),
    ("Какая погода в Эссене", "Сейчас 18 градусов, облачно"),
    ("Почему гемини упал", "Не знаю, у меня нет данных о сбоях"),
]


class MemorySearchBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "mem.sqlite")
        self.mem = MemoryStore(self.path, vector_search=False)
        for u, b in DIALOGS:
            self.mem.add(u, b)

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()


class FtsAvailabilityTests(MemorySearchBase):
    def test_fts_is_enabled(self):
        self.assertTrue(self.mem.fts)

    def test_index_covers_all_rows(self):
        n = self.mem.conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
        self.assertEqual(n, len(DIALOGS))

    def test_backfill_indexes_preexisting_rows(self):
        """Записи, созданные до появления индекса, обязаны в него попасть."""
        self.mem.conn.executescript("DROP TABLE memory_fts;")
        self.mem.conn.commit()
        again = MemoryStore(self.path, vector_search=False)
        try:
            self.assertTrue(again.fts)
            hits = again.search("оливье")
            self.assertTrue(hits)
        finally:
            again.close()


class SearchQualityTests(MemorySearchBase):
    def test_finds_by_word_from_bot_reply(self):
        hits = self.mem.search("оливье")
        self.assertTrue(hits)
        self.assertIn("Оливье", hits[0]["bot"])

    def test_prefix_match_finds_inflected_form(self):
        """Русский не стеммится — «график» обязан находить «графика/графике»."""
        hits = self.mem.search("график")
        self.assertTrue(hits)
        self.assertTrue(any("график" in h["user"].lower() for h in hits))

    def test_scores_pass_the_consumer_filter(self):
        """Вызывающий берёт только score > 0.5. Иначе поиск бессмыслен."""
        for query in ("оливье", "погода", "гемини"):
            for hit in self.mem.search(query):
                self.assertGreater(hit["score"], 0.5, f"{query}: {hit}")

    def test_ranking_puts_best_first(self):
        hits = self.mem.search("график смен")
        self.assertTrue(hits)
        self.assertIn("график", hits[0]["user"].lower())

    def test_top_k_is_respected(self):
        self.assertLessEqual(len(self.mem.search("а", top_k=2)), 2)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.mem.search("зубоврачебный кабинет в мурманске"), [])


class QuerySanitizationTests(MemorySearchBase):
    """Свободный текст — не синтаксис FTS5. Спецсимволы не должны ронять поиск."""

    DANGEROUS = [
        'что-то (важное)', 'кавычки "внутри" строки', 'звёздочка * и OR AND NEAR',
        'скобки ) ( и минус -', 'запрос^с^кареткой', ':колон: и ;точка;',
    ]

    def test_special_characters_do_not_raise(self):
        for q in self.DANGEROUS:
            try:
                self.mem.search(q)
            except Exception as e:  # noqa: BLE001
                self.fail(f"{q!r} уронил поиск: {e}")

    def test_short_words_are_dropped(self):
        from utils.memory import MemoryStore as MS
        self.assertEqual(MS._fts_query("а на до"), "")

    def test_query_is_prefix_matched_and_quoted(self):
        from utils.memory import MemoryStore as MS
        self.assertEqual(MS._fts_query("график смен"), '"график"* OR "смен"*')

    def test_empty_query_is_safe(self):
        self.assertEqual(self.mem.search(""), [])
        self.assertEqual(self.mem.search("   "), [])


class ScoreMappingTests(unittest.TestCase):
    def test_stronger_match_scores_higher(self):
        weak = MemoryStore._bm25_to_score(-0.5)
        strong = MemoryStore._bm25_to_score(-12.0)
        self.assertGreater(strong, weak)

    def test_always_above_consumer_threshold(self):
        for rank in (0.0, -0.01, -1.0, -50.0):
            self.assertGreater(MemoryStore._bm25_to_score(rank), 0.5)

    def test_never_reaches_one(self):
        self.assertLess(MemoryStore._bm25_to_score(-1e6), 1.0)


class IndexSyncTests(MemorySearchBase):
    def test_new_record_is_searchable_immediately(self):
        self.mem.add("Сдал экзамен по матану", "Поздравляю, записала")
        hits = self.mem.search("матану")
        self.assertTrue(hits)

    def test_deleted_record_leaves_the_index(self):
        row_id = self.mem.add("Временная запись про уникальнейшее", "ок")
        self.assertTrue(self.mem.search("уникальнейшее"))
        self.mem.conn.execute("DELETE FROM memory WHERE id=?", (row_id,))
        self.mem.conn.commit()
        self.assertEqual(self.mem.search("уникальнейшее"), [])


if __name__ == "__main__":
    unittest.main()
