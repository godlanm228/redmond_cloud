import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
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

from logic import coach_storage, pings
from logic.tools import execute_tool
from logic.week_schedule import get_shift, get_shift_record, save_shifts
from utils.time import OWNER_TZ


class ScheduleToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_save_work_shift_updates_schedule_and_diary(self):
        result = execute_tool(
            "save_work_shift",
            {"date": "2026-07-06", "start": "17", "end": "23"},
        )

        self.assertIn("2026-07-06 17:00–23:00", result)
        shifts = json.loads(Path("data/coach/shifts.json").read_text(encoding="utf-8"))
        self.assertEqual(shifts["2026-07-06"]["start"], "17:00")
        self.assertEqual(shifts["2026-07-06"]["end"], "23:00")
        self.assertEqual(shifts["2026-07-06"]["status"], "confirmed")
        self.assertEqual(shifts["2026-07-06"]["source"], "text")
        diary = json.loads(Path("data/coach/diary.json").read_text(encoding="utf-8"))
        self.assertEqual(diary[-1]["tags"], ["работа"])

    def test_cancel_shift_hides_it_from_active_schedule(self):
        save_shifts([{"date": "2026-07-06", "start": "17:00", "end": "23:00"}])

        result = execute_tool(
            "set_work_shift_status",
            {"date": "2026-07-06", "status": "cancelled", "note": "не иду"},
        )

        self.assertIn("отмен", result)
        self.assertIsNone(get_shift(datetime(2026, 7, 6, tzinfo=OWNER_TZ).date()))
        self.assertEqual(get_shift_record(datetime(2026, 7, 6, tzinfo=OWNER_TZ).date())["status"], "cancelled")


class DayTickerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()

        self.fixed_now = datetime(2026, 7, 6, 17, 30, tzinfo=OWNER_TZ)
        self.old_pings_now = pings.now_local
        self.old_storage_now = coach_storage.now_local
        pings.now_local = lambda: self.fixed_now
        coach_storage.now_local = lambda: self.fixed_now

    def tearDown(self):
        pings.now_local = self.old_pings_now
        coach_storage.now_local = self.old_storage_now
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def _mark_owner_seen_and_fed(self):
        coach_storage.mark_owner_seen()
        coach_storage.add_diary_entry("Поел утром", tags=["питание"])

    def test_training_ping_when_free_day_and_fed(self):
        self._mark_owner_seen_and_fed()

        decision = pings.decide_ping()

        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], "training")

    def test_no_training_ping_after_work_logged(self):
        self._mark_owner_seen_and_fed()
        coach_storage.add_diary_entry("Еду на работу", tags=["работа"])

        self.assertIsNone(pings.decide_ping())

    def test_no_training_ping_late_evening(self):
        self.fixed_now = datetime(2026, 7, 6, 22, 0, tzinfo=OWNER_TZ)
        self._mark_owner_seen_and_fed()

        self.assertIsNone(pings.decide_ping())

    def test_shift_confirmation_for_legacy_shift_only_once(self):
        self.fixed_now = datetime(2026, 7, 8, 15, 0, tzinfo=OWNER_TZ)
        self._mark_owner_seen_and_fed()
        save_shifts([{
            "date": "2026-07-08",
            "start": "17:00",
            "end": "23:00",
            "source": "legacy",
            "confidence": "medium",
        }])

        decision = pings.decide_ping()

        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], "shift_confirm")

    def test_no_shift_confirmation_for_fresh_confirmed_text_shift(self):
        self.fixed_now = datetime(2026, 7, 8, 15, 0, tzinfo=OWNER_TZ)
        self._mark_owner_seen_and_fed()
        execute_tool(
            "save_work_shift",
            {"date": "2026-07-08", "start": "17", "end": "23"},
        )

        self.assertIsNone(pings.decide_ping())


if __name__ == "__main__":
    unittest.main()
