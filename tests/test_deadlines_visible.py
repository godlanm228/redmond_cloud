"""Просроченный дедлайн — самое важное, что можно показать. Он не исчезает.

Что было до 22.08.2026:
  • `list_deadlines(upcoming_days=N)` фильтровал `today <= due <= cutoff`,
    то есть окно отрезало всё, что раньше сегодня. Этой функцией отвечает
    инструмент — и на вопрос «какие у меня дедлайны» владелец не видел
    именно тот, который пропустил. Скедулер звал версию без окна и просрочку
    видел: система знала, а показать не могла.
  • `top_priorities` выбрасывал всё, просроченное больше чем на неделю —
    чем дольше владелец тянул, тем реже коуч напоминал, а через семь дней
    замолкал совсем.
  • Кривая дата означала «лежит в базе и не показывается никогда», молча.
"""

import unittest
from datetime import timedelta

from logic import coach_storage as cs
from logic import priorities, tools
from utils.time import now_local


class OverdueStaysVisible(unittest.TestCase):
    def setUp(self):
        self.today = now_local().date()

    def _add(self, title, days, importance="medium"):
        return cs.add_deadline(title, (self.today + timedelta(days=days)).isoformat(),
                               importance)

    def test_tool_output_marks_overdue(self):
        self._add("Матан", -12)
        out = tools.execute_tool("list_deadlines", {"upcoming_days": 7}, None)
        self.assertIn("Матан", out)
        self.assertIn("ПРОСРОЧЕН на 12 дн", out)

    def test_tool_output_marks_today(self):
        self._add("Сегодняшний", 0)
        out = tools.execute_tool("list_deadlines", {"upcoming_days": 7}, None)
        self.assertIn("СЕГОДНЯ", out)

    def test_long_overdue_survives_in_the_prompt_block(self):
        """Двенадцать дней просрочки — раньше исчезало из промпта совсем."""
        self._add("Матан", -12, "high")
        block = priorities.build_priorities_block()
        self.assertIn("Матан", block)
        self.assertIn("ПРОСРОЧЕН", block)

    def test_block_does_not_truncate_silently(self):
        for n in range(6):
            self._add(f"Дедлайн {n}", n + 1)
        block = priorities.build_priorities_block()
        self.assertIn("и ещё", block,
                      "блок показал часть списка и не сказал, что это не всё")

    def test_unreadable_date_is_shown_not_swallowed(self):
        cs.add_deadline("кривая дата", "завтра")
        out = tools.execute_tool("list_deadlines", {"upcoming_days": 7}, None)
        self.assertIn("кривая дата", out)
        self.assertIn("не читается", out)

    def test_closed_deadline_does_not_come_back(self):
        d = self._add("сдал", -3)
        cs.mark_deadline_done(d["id"])
        out = tools.execute_tool("list_deadlines", {"upcoming_days": 7}, None)
        self.assertNotIn("сдал", out)


if __name__ == "__main__":
    unittest.main()
