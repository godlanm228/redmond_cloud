"""Слой БД и миграция JSON → SQLite."""

import json
import os
import sqlite3
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

from utils import db


class DbTestBase(unittest.TestCase):
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


class SchemaTests(DbTestBase):
    EXPECTED = {"goals", "deadlines", "diary", "pantry", "shifts",
                "shift_events", "kv", "chat_history"}

    def test_all_tables_created(self):
        rows = db.query("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in rows}
        self.assertTrue(self.EXPECTED <= names, self.EXPECTED - names)

    def test_schema_version_is_set(self):
        v = db.query_one("PRAGMA user_version")[0]
        self.assertEqual(v, db.SCHEMA_VERSION)

    def test_init_is_idempotent(self):
        db.init_schema(db.connect())
        db.init_schema(db.connect())
        rows = db.query("SELECT name FROM sqlite_master WHERE type='table'")
        self.assertTrue(self.EXPECTED <= {r["name"] for r in rows})

    def test_wal_enabled(self):
        mode = db.query_one("PRAGMA journal_mode")[0]
        self.assertEqual(mode.lower(), "wal")


class KvTests(DbTestBase):
    def test_roundtrip(self):
        db.kv_set("mute", {"until": "2026-08-20T10:00", "scope": "pings"})
        self.assertEqual(db.kv_get("mute")["scope"], "pings")

    def test_missing_key_returns_default(self):
        self.assertEqual(db.kv_get("нет-такого", {"a": 1}), {"a": 1})

    def test_upsert_replaces_not_duplicates(self):
        db.kv_set("ping_style", {"idx": 1})
        db.kv_set("ping_style", {"idx": 2})
        self.assertEqual(db.kv_get("ping_style")["idx"], 2)
        n = db.query_one("SELECT COUNT(*) c FROM kv WHERE key='ping_style'")["c"]
        self.assertEqual(n, 1)

    def test_corrupt_value_gives_default(self):
        db.execute("INSERT INTO kv(key, value, updated) VALUES('bad','{битый','now')")
        self.assertEqual(db.kv_get("bad", "дефолт"), "дефолт")


class ConcurrencyTests(DbTestBase):
    def test_parallel_writes_from_threads_all_land(self):
        """Каждый поток получает своё соединение; busy_timeout не даёт упасть."""
        n = 16
        errors = []

        def write(i):
            try:
                db.execute("INSERT INTO diary(ts, text, tags) VALUES(?,?,'[]')",
                           (f"2026-08-13T0{i % 10}:00", f"запись {i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(db.query_one("SELECT COUNT(*) c FROM diary")["c"], n)


class BackupTests(DbTestBase):
    def _read_kv(self, dest, key):
        conn = sqlite3.connect(str(dest))
        try:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def test_backup_creates_readable_copy(self):
        db.kv_set("presence", {"wake_time": "13:37"})
        dest = db.backup_to(Path("data/backup/hub-test.sqlite"))
        self.assertTrue(dest.exists())
        self.assertIn("13:37", self._read_kv(dest, "presence"))

    def test_backup_overwrites_previous(self):
        db.kv_set("presence", {"wake_time": "10:00"})
        db.backup_to(Path("data/backup/snap.sqlite"))
        db.kv_set("presence", {"wake_time": "11:00"})
        dest = db.backup_to(Path("data/backup/snap.sqlite"))
        self.assertIn("11:00", self._read_kv(dest, "presence"))


class MigrationTests(DbTestBase):
    GOALS = [{"id": 1, "title": "Лечь спать до 23:59", "why": "восстановление",
              "status": "active", "created": "2026-05-15", "progress_log": []}]
    DEADLINES = [{"id": 6, "title": "Экзамен", "due": "2026-07-31",
                  "importance": "high", "status": "done", "created": "2026-07-28",
                  "closed": "2026-08-02"}]
    DIARY = [{"id": 1, "timestamp": "2026-06-09T23:37", "text": "не спал",
              "tags": ["усталость"]},
             {"id": 2, "timestamp": "2026-08-12T21:53", "text": "салат оливье",
              "tags": ["питание"]}]
    SHIFTS = {"2026-08-13": {"start": "17:00", "end": "23:00", "status": "planned",
                             "source": "photo", "confidence": "high",
                             "updated": "2026-08-12T21:55+02:00"}}

    def setUp(self):
        super().setUp()
        Path("data/coach").mkdir(parents=True, exist_ok=True)
        self._write("goals.json", self.GOALS)
        self._write("deadlines.json", self.DEADLINES)
        self._write("diary.json", self.DIARY)
        self._write("shifts.json", self.SHIFTS)
        self._write("pantry.json", {"items": ["Wok Mix", "Овощи"], "updated": "2026-06-18"})
        self._write("mute.json", {"until": "2026-08-09T15:36+02:00"})
        self._write("gemini_usage.json", {"date": "2026-08-13", "count": 7})

    def _write(self, name, data):
        (Path("data/coach") / name).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _run(self, **kw):
        from utils.migrate_json_to_db import run
        return run(**kw)

    def test_every_table_reports_ok(self):
        for r in self._run():
            self.assertTrue(r["ok"], r)

    def test_counts_match_source(self):
        report = {r["table"]: r for r in self._run()}
        self.assertEqual(report["goals"]["db"], len(self.GOALS))
        self.assertEqual(report["diary"]["db"], len(self.DIARY))
        self.assertEqual(report["shifts"]["db"], len(self.SHIFTS))

    def test_values_survive_intact(self):
        self._run()
        row = db.query_one("SELECT * FROM diary WHERE id=2")
        self.assertEqual(row["text"], "салат оливье")
        self.assertEqual(json.loads(row["tags"]), ["питание"])

    def test_kv_files_land_in_kv(self):
        self._run()
        self.assertEqual(db.kv_get("gemini_usage")["count"], 7)
        self.assertIn("2026-08-09", db.kv_get("mute")["until"])

    def test_shift_import_event_recorded(self):
        self._run()
        row = db.query_one(
            "SELECT * FROM shift_events WHERE date='2026-08-13' AND action='import'")
        self.assertIsNotNone(row)
        self.assertEqual(row["source"], "photo")

    def test_rerun_does_not_duplicate(self):
        self._run()
        self._run()
        self.assertEqual(db.query_one("SELECT COUNT(*) c FROM diary")["c"], len(self.DIARY))
        n = db.query_one("SELECT COUNT(*) c FROM shift_events")["c"]
        self.assertEqual(n, len(self.SHIFTS))

    def test_dry_run_changes_nothing(self):
        self._run(dry_run=True)
        self.assertEqual(db.query_one("SELECT COUNT(*) c FROM diary")["c"], 0)

    def test_json_files_are_not_deleted(self):
        self._run()
        self.assertTrue((Path("data/coach") / "diary.json").exists())

    def test_corrupt_source_aborts_without_partial_write(self):
        (Path("data/coach") / "diary.json").write_text("{битый", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self._run()
        # goals успели пройти до diary — но транзакция откатилась целиком
        self.assertEqual(db.query_one("SELECT COUNT(*) c FROM goals")["c"], 0)


if __name__ == "__main__":
    unittest.main()
