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
from logic.prompt_budget import describe, estimate_tokens


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


class SizeLoggingTests(unittest.TestCase):
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
        prompt_budget.log_size("Redmond", huge, TOOLS)
        self.assertTrue(any(r.levelno == logging.WARNING for r in self.records))

    def test_small_prompt_does_not_warn(self):
        small = [{"role": "user", "content": "коротко"}]
        prompt_budget.log_size("Redmond", small, TOOLS)
        self.assertFalse(any(r.levelno == logging.WARNING for r in self.records))


if __name__ == "__main__":
    unittest.main()
