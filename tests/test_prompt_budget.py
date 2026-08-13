"""Бюджет промпта и теневой отбор инструментов.

Теневой режим ничего не отключает, поэтому тесты проверяют не поведение бота,
а корректность самой метрики: правильно ли раскладывается промпт и не отрезает
ли отбор инструменты, которые модель реально просит.
"""

import logging
import sys
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

from logic import prompt_budget
from logic.prompt_budget import describe, estimate_tokens, select_tools


def tool(name, description):
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": {}}}}


TOOLS = [
    tool("get_weather", "Погода в городе, температура, осадки, прогноз"),
    tool("web_search", "Поиск в интернете по запросу"),
    tool("get_current_time", "Текущее время и дата"),
    tool("log_meal", "Записать приём пищи: блюдо, калории, белок, питание"),
    tool("save_work_shift", "Сохранить рабочую смену: график, время начала и конца"),
    tool("add_deadline", "Добавить дедлайн с датой"),
    tool("read_diary", "Прочитать записи дневника"),
    tool("get_crypto_market", "Курс криптовалюты, биткоин, рынок"),
    tool("read_dossier_section", "Секция досье владельца"),
    tool("list_goals", "Список целей владельца"),
]


class EstimateTests(unittest.TestCase):
    def test_string_estimate(self):
        self.assertEqual(estimate_tokens("a" * 400), 100)

    def test_structure_estimate_is_positive(self):
        self.assertGreater(estimate_tokens(TOOLS), 0)

    def test_unserializable_does_not_crash(self):
        self.assertGreater(estimate_tokens({"плохое": {1, 2, 3}}), 0)


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "s" * 400},
            {"role": "user", "content": "u" * 80},
            {"role": "tool", "content": "t" * 4000},
        ]

    def test_parts_are_split_by_role(self):
        d = describe(self.messages, TOOLS)
        self.assertEqual(d["system"], 100)
        self.assertEqual(d["user"], 20)
        self.assertEqual(d["tool_results"], 1000)

    def test_schemas_counted_separately(self):
        d = describe(self.messages, TOOLS)
        self.assertEqual(d["tool_schemas"], estimate_tokens(TOOLS))

    def test_total_is_sum_of_parts(self):
        d = describe(self.messages, TOOLS)
        self.assertEqual(d["total"],
                         sum(v for k, v in d.items() if k != "total"))

    def test_no_tools_means_no_schema_cost(self):
        self.assertEqual(describe(self.messages, None)["tool_schemas"], 0)


class SelectToolsTests(unittest.TestCase):
    def _names(self, text):
        return {t["function"]["name"] for t in select_tools(text, TOOLS)}

    def test_core_tools_always_present(self):
        names = self._names("абсолютно ничего общего")
        self.assertTrue(prompt_budget.CORE_TOOLS <= names)

    def test_relevant_tool_is_picked(self):
        self.assertIn("log_meal", self._names("запиши что я поел, калории"))

    def test_crypto_question_picks_crypto_tool(self):
        self.assertIn("get_crypto_market", self._names("какой курс биткоина"))

    def test_selection_is_smaller_than_full_catalog(self):
        picked = select_tools("какая погода", TOOLS, max_selected=4)
        self.assertLessEqual(len(picked), 4)
        self.assertLess(estimate_tokens(picked), estimate_tokens(TOOLS))

    def test_empty_catalog_is_safe(self):
        self.assertEqual(select_tools("что угодно", []), [])

    def test_yesterday_case_keeps_web_search(self):
        """«Почему гемини упал?» → модель пошла в web_search. Отбор обязан его дать."""
        self.assertIn("web_search", self._names("Почему гемини упал ?"))


class ShadowLoggingTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = self.records.append
        self.logger = logging.getLogger("logic.prompt_budget")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def test_oversized_prompt_warns(self):
        huge = [{"role": "user", "content": "x" * (prompt_budget.GROQ_TPM_LIMIT * 4)}]
        prompt_budget.log_shadow("Redmond", "вопрос", huge, TOOLS)
        self.assertTrue(any(r.levelno == logging.WARNING for r in self.records))

    def test_small_prompt_does_not_warn(self):
        small = [{"role": "user", "content": "коротко"}]
        prompt_budget.log_shadow("Redmond", "коротко", small, TOOLS)
        self.assertFalse(any(r.levelno == logging.WARNING for r in self.records))

    def test_miss_is_reported(self):
        prompt_budget.log_selection_miss("Redmond", "get_weather", {"web_search"})
        self.assertTrue(any("Shadow-промах" in r.getMessage() for r in self.records))

    def test_hit_is_not_reported(self):
        prompt_budget.log_selection_miss("Redmond", "web_search", {"web_search"})
        self.assertFalse(any("Shadow-промах" in r.getMessage() for r in self.records))

    def test_empty_selection_is_not_a_miss(self):
        """Отбор не считался (не тот хоп) — это не промах."""
        prompt_budget.log_selection_miss("Redmond", "get_weather", set())
        self.assertFalse(any("Shadow-промах" in r.getMessage() for r in self.records))


if __name__ == "__main__":
    unittest.main()
