"""Records code just made from the owner's current message survive a delete.

30.09.2026: "delete the gym entry ... I worked 5 hours on development".
Code recorded the new facts (#110, #111); the model deleted them together
with the old entries (#107, #108).
"""
import unittest

from logic.tools import ToolSession, _refuse_protected


def _session():
    s = ToolSession()
    s.protect_code_records([
        ("add_diary_entry", {}, "Запись #110 в дневник: «Завтра у меня собеседование» [план, работа]."),
        ("add_diary_entry", {}, "Запись #111 в дневник: «Я 5 часов занимался разработкой» [сделано, работа]."),
    ])
    s.note("diary", [107, 108, 110, 111])
    return s


class FreshRecordsProtectedTest(unittest.TestCase):
    def test_fresh_records_are_refused(self):
        refusal = _refuse_protected("delete_diary_entry", {"entry_ids": [107, 108, 110, 111]}, _session())
        self.assertIn("#110", refusal)
        self.assertIn("#111", refusal)
        self.assertNotIn("#107", refusal)

    def test_old_records_can_be_deleted(self):
        self.assertEqual(_refuse_protected("delete_diary_entry", {"entry_ids": [107, 108]}, _session()), "")

    def test_reading_is_not_affected(self):
        self.assertEqual(_refuse_protected("read_diary", {"last_n": 5}, _session()), "")

    def test_no_code_actions_means_no_protection(self):
        s = ToolSession()
        s.protect_code_records([])
        self.assertEqual(_refuse_protected("delete_diary_entry", {"entry_ids": [110]}, s), "")


if __name__ == "__main__":
    unittest.main()
