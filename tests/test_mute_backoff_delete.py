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
from logic.situation_engine import DaySituation, ShiftSituation
from logic.tools import execute_tool
from utils.time import now_local


class _TmpDataDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()


class DeleteDeadlineTests(_TmpDataDir):
    def test_delete_removes_entirely_unlike_done(self):
        d1 = coach_storage.add_deadline("Ошибочный", "2099-01-01")
        d2 = coach_storage.add_deadline("Сдать тест", "2099-01-02")

        coach_storage.mark_deadline_done(d2["id"])
        removed = coach_storage.delete_deadline(d1["id"])

        self.assertEqual(removed["title"], "Ошибочный")
        left = coach_storage.list_deadlines()
        self.assertEqual([d["id"] for d in left], [d2["id"]])
        self.assertEqual(left[0]["status"], "done")  # done остаётся в истории

    def test_delete_missing_id_returns_none(self):
        self.assertIsNone(coach_storage.delete_deadline(404))

    def test_tool_delete_deadline(self):
        d = coach_storage.add_deadline("Лишний", "2099-03-03")
        result = execute_tool("delete_deadline", {"deadline_id": d["id"]})
        self.assertIn("удалён насовсем", result)
        self.assertEqual(coach_storage.list_deadlines(), [])

        self.assertIn("не найден", execute_tool("delete_deadline", {"deadline_id": 99}))


class SoftMuteTests(_TmpDataDir):
    def test_default_scope_pings_keeps_digests(self):
        coach_storage.set_mute(mode="hours", hours=4)  # дефолт scope='pings'
        self.assertTrue(coach_storage.muted_now())        # тикер молчит
        self.assertFalse(coach_storage.hard_muted_now())  # дайджесты живут

    def test_scope_all_silences_everything(self):
        coach_storage.set_mute(mode="hours", hours=4, scope="all")
        self.assertTrue(coach_storage.muted_now())
        self.assertTrue(coach_storage.hard_muted_now())

    def test_legacy_record_without_scope_is_hard(self):
        # Старый mute.json (поставлен до введения scope) — полная тишина,
        # смысл уже действующей просьбы не меняем.
        Path("data/coach").mkdir(parents=True, exist_ok=True)
        Path("data/coach/mute.json").write_text(
            json.dumps({"until": "forever"}), encoding="utf-8"
        )
        self.assertTrue(coach_storage.muted_now())
        self.assertTrue(coach_storage.hard_muted_now())

    def test_unmute_clears_both(self):
        coach_storage.set_mute(mode="forever", scope="all")
        coach_storage.unmute()
        self.assertFalse(coach_storage.muted_now())
        self.assertFalse(coach_storage.hard_muted_now())

    def test_tool_messages_explain_scope(self):
        msg = execute_tool("mute_notifications", {"hours": 2})
        self.assertIn("дайджест", msg.lower())
        self.assertIn("остаются", msg)

        msg_all = execute_tool("mute_notifications", {"hours": 2, "scope": "all"})
        self.assertIn("Полная тишина", msg_all)


def _situation(now, pings_dict, last_msg, muted=False):
    shift = ShiftSituation(
        record=None, active_record=None, start_at=None, end_at=None,
        status="planned", source="legacy", confidence="medium",
        updated_at=None, confirmed_at=None,
    )
    return DaySituation(
        now=now, day_state={}, pings=pings_dict, owner_seen=bool(last_msg),
        muted=muted, tags=set(), entries_today=0, wake_time=None,
        shift=shift, in_study_block=False, last_msg=last_msg,
    )


class BackoffTests(unittest.TestCase):
    NOW = datetime(2026, 8, 3, 18, 0)

    def test_two_unanswered_pings_stop_ticker(self):
        s = _situation(self.NOW, {"greeting": "12:00", "meal": "14:00"}, last_msg="11:00")
        self.assertEqual(s.ignored_pings_streak(), 2)
        self.assertFalse(s.proactive_allowed(5, 90))

    def test_reply_resets_backoff(self):
        s = _situation(self.NOW, {"greeting": "12:00", "meal": "14:00"}, last_msg="15:00")
        self.assertEqual(s.ignored_pings_streak(), 0)
        self.assertTrue(s.proactive_allowed(5, 90))

    def test_one_unanswered_ping_still_allowed(self):
        s = _situation(self.NOW, {"greeting": "12:00"}, last_msg="11:00")
        self.assertEqual(s.ignored_pings_streak(), 1)
        self.assertTrue(s.proactive_allowed(5, 90))

    def test_min_gap_still_enforced(self):
        s = _situation(self.NOW, {"meal": "17:30"}, last_msg="17:45")
        self.assertFalse(s.proactive_allowed(5, 90))  # 30 мин < 90


class StyleVariantsTests(_TmpDataDir):
    def test_rotation_never_repeats_consecutively(self):
        n = len(pings.STYLE_VARIANTS)
        seen = [coach_storage.next_style_index(n) for _ in range(n * 2)]
        for a, b in zip(seen, seen[1:]):
            self.assertNotEqual(a, b)

    def test_gentle_tone_on_work_day(self):
        s = _situation(now_local(), {}, last_msg="10:00")
        object.__setattr__(s, "tags", {"работа"})
        hint = pings._style_hint(s)
        self.assertIn("бережный", hint)

    def test_variants_rotate_on_free_day(self):
        s = _situation(now_local(), {}, last_msg="10:00")
        h1 = pings._style_hint(s)
        h2 = pings._style_hint(s)
        self.assertNotEqual(h1, h2)
        for h in (h1, h2):
            self.assertIn("Не повторяй формулировки", h)


if __name__ == "__main__":
    unittest.main()
