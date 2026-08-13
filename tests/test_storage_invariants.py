"""Инварианты, которые вернули бы потерю данных, если их сломать.

A — ошибка провайдера классифицируется по ВСЕЙ цепочке моделей, а не по последней;
C — «файла нет» и «файл битый» это разные исходы, поверх битого не пишем;
D — запись атомарна, а read-modify-write не теряет параллельных изменений.
"""

import json
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
from logic.coach_storage import StorageCorrupt
from logic.response_generator import (
    _is_model_gone_error,
    _is_oversize_error,
    _is_rate_limit_error,
    chain_has,
)

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


class StorageBaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def _path(self, name):
        return Path(coach_storage._coach_dir()) / name


class CorruptFileTests(StorageBaseTest):
    def test_missing_file_gives_default(self):
        self.assertEqual(coach_storage._load_json("нет-такого.json", []), [])

    def test_corrupt_file_raises_instead_of_default(self):
        self._path("diary.json").write_text("{битый", encoding="utf-8")
        with self.assertRaises(StorageCorrupt):
            coach_storage._load_json("diary.json", [])

    def test_corrupt_file_is_copied_aside(self):
        self._path("diary.json").write_text("{битый", encoding="utf-8")
        with self.assertRaises(StorageCorrupt):
            coach_storage._load_json("diary.json", [])
        self.assertTrue(self._path("diary.json.corrupt").exists())

    def test_mutator_does_not_overwrite_corrupt_data(self):
        """Главный сценарий потери: битый дневник + новая запись = один элемент."""
        self._path("diary.json").write_text("[{битый", encoding="utf-8")
        with self.assertRaises(StorageCorrupt):
            coach_storage.add_diary_entry("новая запись")
        self.assertEqual(self._path("diary.json").read_text(encoding="utf-8"), "[{битый")

    def test_safe_reader_degrades_without_touching_file(self):
        self._path("mute.json").write_text("{битый", encoding="utf-8")
        self.assertEqual(coach_storage._load_json_safe("mute.json", {}), {})
        self.assertFalse(coach_storage.muted_now())
        self.assertEqual(self._path("mute.json").read_text(encoding="utf-8"), "{битый")


class AtomicWriteTests(StorageBaseTest):
    def test_write_is_atomic_and_leaves_no_tmp(self):
        coach_storage._save_json("goals.json", [{"id": 1}])
        self.assertEqual(json.loads(self._path("goals.json").read_text(encoding="utf-8")),
                         [{"id": 1}])
        self.assertFalse(self._path("goals.json.tmp").exists())

    def test_failed_serialization_leaves_old_content_intact(self):
        coach_storage._save_json("goals.json", [{"id": 1}])
        with self.assertRaises(TypeError):
            coach_storage._save_json("goals.json", {"плохое": {1, 2, 3}})  # set не сериализуем
        self.assertEqual(json.loads(self._path("goals.json").read_text(encoding="utf-8")),
                         [{"id": 1}])


class ConcurrentMutationTests(StorageBaseTest):
    def test_parallel_diary_writes_do_not_lose_entries(self):
        """Без блокировки два «прочитал → добавил → записал» теряют записи."""
        n = 12
        errors = []

        def add(i):
            try:
                coach_storage.add_diary_entry(f"запись {i}")
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


if __name__ == "__main__":
    unittest.main()
