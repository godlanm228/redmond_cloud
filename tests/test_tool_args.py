"""Аргументы инструмента проверяются по его же схеме.

Схемы уже объявляют `required` и типы — у 21 инструмента из 34. Не проверял
их никто, и это давало два класса поломок:

  • `add_goal({})` создавал цель с пустым названием, `add_deadline({})` —
    дедлайн без названия и срока. Мусор ложился в базу и потом показывался
    владельцу как настоящая запись.
  • Модели регулярно шлют числа словами. `int(args.get("top_k", 3))` на
    `"три"` бросал ValueError, тот всплывал в _generate_with_providers и
    записывался как «Provider groq failed» — баг инструмента выглядел
    отказом провайдера. На правдоподобных аргументах падало 7 случаев из 10.

Инвариант: декларация уже есть — надо ею пользоваться, а не дописывать
проверки в каждый инструмент по отдельности.
"""

import unittest

from logic import coach_storage as cs
from logic import tools


class RequiredFieldsAreEnforced(unittest.TestCase):
    def test_goal_without_title_is_not_created(self):
        out = tools.execute_tool("add_goal", {}, None)
        self.assertEqual(cs.list_goals(), [], "в базу легла пустая цель")
        self.assertIn("title", out)

    def test_blank_title_counts_as_missing(self):
        out = tools.execute_tool("add_goal", {"title": "   "}, None)
        self.assertEqual(cs.list_goals(), [])
        self.assertIn("title", out)

    def test_deadline_without_due_is_not_created(self):
        out = tools.execute_tool("add_deadline", {"title": "Матан"}, None)
        self.assertEqual(cs.list_deadlines(), [], "в базу лёг дедлайн без срока")
        self.assertIn("due", out)

    def test_error_names_every_missing_field(self):
        out = tools.execute_tool("add_deadline", {}, None)
        self.assertIn("title", out)
        self.assertIn("due", out)

    def test_valid_call_still_works(self):
        out = tools.execute_tool("add_goal", {"title": "Лечь спать до 23:59"}, None)
        self.assertIn("создана", out)
        self.assertEqual(len(cs.list_goals()), 1)


class TypesAreCoercedOrRefusedHonestly(unittest.TestCase):
    def test_number_as_word_does_not_crash(self):
        """Раньше это было ValueError, всплывавший как отказ провайдера."""
        out = tools.execute_tool("read_diary", {"last_n": "десять"}, None)
        self.assertIn("last_n", out)
        self.assertIn("десять", out)

    def test_numeric_string_is_accepted(self):
        """Модели часто шлют «5» строкой — это не повод отказывать."""
        cs.add_diary_entry("запись про спорт", tags=["спорт"])
        out = tools.execute_tool("read_diary", {"last_n": "5"}, None)
        self.assertIn("спорт", out)

    def test_none_where_a_number_is_expected(self):
        cs.add_diary_entry("запись")
        out = tools.execute_tool("read_diary", {"last_n": None}, None)
        self.assertNotIn("не отработал", out, "None должен читаться как «не задано»")

    def test_no_tool_raises_on_plausible_garbage(self):
        """Тот самый набор, на котором раньше падало 7 из 10."""
        cases = [
            ("read_diary", {"last_n": "десять"}),
            ("list_deadlines", {"upcoming_days": "неделя"}),
            ("get_week_schedule", {"days": "8 дней"}),
            ("mute_notifications", {"hours": "два"}),
            ("postpone_deadline", {"deadline_id": "первый", "new_due": "завтра"}),
            ("delete_diary_entry", {"entry_ids": "последнюю"}),
            ("mark_goal_done", {"goal_id": "первую"}),
        ]
        for name, args in cases:
            try:
                out = tools.execute_tool(name, args, None)
            except Exception as e:  # pragma: no cover
                self.fail(f"{name}{args} бросил {type(e).__name__}: {e}")
            self.assertTrue(out, f"{name} вернул пустоту вместо объяснения")


class ValidationDoesNotBlockToolsWithoutSchema(unittest.TestCase):
    def test_tools_without_required_still_run(self):
        out = tools.execute_tool("get_current_time", {}, None)
        self.assertRegex(out, r"\d{4}-\d{2}-\d{2}")

    def test_optional_fields_may_be_omitted(self):
        cs.add_goal("цель")
        out = tools.execute_tool("list_goals", {}, None)
        self.assertIn("цель", out)


if __name__ == "__main__":
    unittest.main()
