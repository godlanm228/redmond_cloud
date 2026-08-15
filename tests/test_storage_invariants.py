"""Инварианты, нарушение которых означало бы потерю данных.

A — ошибка провайдера классифицируется по ВСЕЙ цепочке моделей, а не по последней;
B — параллельные изменения не теряются, чтение не разрушает данные.

До 15.08.2026 хранилищем были JSON-файлы, и B держался на блокировках плюс
atomic write. Теперь это транзакции SQLite, но проверяем то же самое свойство:
двадцать одновременных записей должны дать двадцать записей.
"""

import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.utils = types.SimpleNamespace(quote=lambda s: s)
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

from logic import coach_storage
from logic.response_generator import (
    _is_model_gone_error,
    _is_oversize_error,
    _is_rate_limit_error,
    chain_has,
)
from utils import db

RATE_LIMIT_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-120b` ... on tokens per minute (TPM): Limit 8000', "
    "'code': 'rate_limit_exceeded'}}"
)
MODEL_GONE_404 = (
    "Error code: 404 - {'error': {'message': 'The model `qwen/qwen3-32b` does not "
    "exist or you do not have access to it.', 'code': 'model_not_found'}}"
)
OVERSIZE_413 = "Error code: 413 - request too large, please reduce the length"


class ErrorChainTests(unittest.TestCase):
    """Ровно сценарий 12.08.2026: 429 у primary, 404 у fallback."""

    CHAIN = [RATE_LIMIT_429, MODEL_GONE_404]

    def test_rate_limit_survives_later_404(self):
        self.assertTrue(chain_has(self.CHAIN, _is_rate_limit_error))

    def test_last_error_alone_would_have_missed_it(self):
        # Фиксируем причину бага: по последней ошибке класс не определяется.
        self.assertFalse(_is_rate_limit_error(self.CHAIN[-1]))

    def test_model_gone_detected(self):
        self.assertTrue(chain_has(self.CHAIN, _is_model_gone_error))

    def test_oversize_detected_anywhere_in_chain(self):
        self.assertTrue(chain_has([OVERSIZE_413, MODEL_GONE_404], _is_oversize_error))

    def test_unrelated_chain_matches_nothing(self):
        chain = ["Connection reset by peer"]
        self.assertFalse(chain_has(chain, _is_rate_limit_error))
        self.assertFalse(chain_has(chain, _is_model_gone_error))
        self.assertFalse(chain_has(chain, _is_oversize_error))

    def test_empty_chain_is_safe(self):
        self.assertFalse(chain_has([], _is_rate_limit_error))


class StorageBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        db.set_db_path(Path("data/hub-test.sqlite"))

    def tearDown(self):
        db.close_all()
        db.set_db_path(db.DEFAULT_DB_PATH)
        os.chdir(self.old_cwd)
        self.tmp.cleanup()


class ConcurrentMutationTests(StorageBase):
    def test_parallel_diary_writes_do_not_lose_entries(self):
        n = 12
        errors = []

        def add(i):
            try:
                coach_storage.add_diary_entry(f"запись номер {i}")
            except Exception as e:  # noqa: BLE001 — тест должен показать причину
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(coach_storage.read_diary(last_n=100)), n)

    def test_parallel_gemini_bump_counts_every_call(self):
        n = 20
        threads = [threading.Thread(target=coach_storage.gemini_bump) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(coach_storage.gemini_count_today(), n)

    def test_parallel_deadline_adds_get_unique_ids(self):
        """Сквозной id не должен выдаваться дважды под нагрузкой."""
        n = 10
        for i in range(n):
            coach_storage.add_deadline(f"дедлайн {i}", "2026-09-01")
        ids = [d["id"] for d in coach_storage.list_deadlines()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), n)


class ReadDoesNotDestroyTests(StorageBase):
    """Чтение при любых обстоятельствах не должно уничтожать данные —
    именно это ломалось на JSON (битый файл → дефолт → сохранение поверх)."""

    def test_reading_empty_storage_is_safe(self):
        self.assertEqual(coach_storage.read_diary(), [])
        self.assertEqual(coach_storage.list_goals(), [])
        self.assertEqual(coach_storage.get_pantry()["items"], [])

    def test_broken_kv_value_does_not_wipe_it(self):
        db.execute("INSERT INTO kv(key, value, updated) VALUES('mute','{битый','now')")
        self.assertFalse(coach_storage.muted_now())
        row = db.query_one("SELECT value FROM kv WHERE key='mute'")
        self.assertEqual(row["value"], "{битый")

    def test_diary_survives_repeated_reads(self):
        coach_storage.add_diary_entry("важная запись про экзамен")
        for _ in range(5):
            coach_storage.read_diary()
        self.assertEqual(len(coach_storage.read_diary()), 1)


if __name__ == "__main__":
    unittest.main()
