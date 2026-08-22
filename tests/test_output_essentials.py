"""Инструмент сам объявляет, что в его выдаче обязано пережить сжатие.

Предыстория. `_compress_tool_content` писалась под web_search и умела
сохранять строки `URL:`. Выдача `read_diary(last_n=10)` — до 2000 символов —
резалась до 400, номера записей пропадали, и через две реплики модель не
могла сослаться на прочитанное. 17.08.2026 она назвала порядковый номер:
ушло `delete_diary_entry([3])`, погибла непричастная запись месячной
давности, а ошибочная осталась на месте.

Первая попытка починки доучила компрессор формату `#id`. Это та же ошибка
в меньшем масштабе: универсальная функция копит знание о чужих предметных
областях, и следующий инструмент с иным форматом ссылки сломается так же
молча — `find_photo` тогда отдавал `[N]`, и регексп его не видел.

Инвариант: **знание о своей выдаче принадлежит инструменту**. Компрессор
спрашивает декларацию и не догадывается по виду текста.

Тест ниже — сторож против возврата класса: он сам находит инструменты, чья
выдача содержит ссылки, и требует, чтобы они были объявлены.
"""

import re
import unittest

from logic import coach_storage as cs
from logic import tools
from logic.response_generator import _compress_tool_content

# Ходят в сеть — в оффлайн-прогоне не проверяются.
NETWORK_TOOLS = {
    "get_weather", "web_search", "get_news_headlines", "web_fetch",
    "get_crypto_market", "lookup_food", "log_meal",
}
# Отдают маркер делегирования, а не текст выдачи.
DELEGATING_TOOLS = {"delegate_research", "ask_iris", "handoff_to_iris"}

_REFERENCE_RX = re.compile(r"#\d+")
_URL_RX = re.compile(r"^\s*URL:", re.MULTILINE)


def _seed():
    cs.add_diary_entry("Позанимался спортом, начал принимать креатин", tags=["спорт"])
    cs.add_diary_entry("Поел гречку с курицей", tags=["питание"])
    cs.add_goal("Лечь спать до 23:59", why="сон")
    cs.add_deadline("Матан", "2026-09-01")


class DeclarationCoversReality(unittest.TestCase):
    def test_every_tool_that_returns_references_is_declared(self):
        """Сторож класса: выдал ссылку — объяви её, иначе она умрёт при сжатии."""
        _seed()
        undeclared = []
        for schema in tools.TOOL_SCHEMAS:
            name = schema["function"]["name"]
            if name in NETWORK_TOOLS or name in DELEGATING_TOOLS:
                continue
            try:
                out = tools.execute_tool(name, {}, None) or ""
            except Exception:
                continue
            if _REFERENCE_RX.search(out) and tools.OUTPUT_ESSENTIALS.get(name) != "ids":
                undeclared.append((name, out.strip().splitlines()[:2]))
        self.assertFalse(
            undeclared,
            "инструменты отдают ссылки #N, но не объявили это в OUTPUT_ESSENTIALS "
            f"— при сжатии ссылки исчезнут: {undeclared}",
        )

    def test_declaration_has_no_typos(self):
        known = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
        unknown = sorted(set(tools.OUTPUT_ESSENTIALS) - known)
        self.assertFalse(unknown, f"объявлены несуществующие инструменты: {unknown}")

    def test_declared_kinds_are_supported(self):
        self.assertTrue(set(tools.OUTPUT_ESSENTIALS.values()) <= {"ids", "urls"})

    def test_reference_format_is_uniform(self):
        """Один формат ссылки на все инструменты — иначе компрессору снова
        придётся знать про каждый по отдельности."""
        _seed()
        for name in ("read_diary", "list_goals", "list_deadlines"):
            out = tools.execute_tool(name, {}, None)
            self.assertRegex(out, r"#\d+", f"{name} отдаёт ссылки не в формате #N")


class CompressorHasNoDomainKnowledge(unittest.TestCase):
    def _diary_output(self):
        lines = ["Последние записи (10):"]
        for i in range(80, 90):
            lines.append(f"  #{i} 2026-08-17T12:0{i % 10} [спорт]: Запись номер {i}, "
                         f"достаточно длинная чтобы выдача перевалила лимит сжатия.")
        return "\n".join(lines)

    def test_without_declaration_nothing_is_kept(self):
        """Компрессор сам по себе не знает ни про дневник, ни про поиск."""
        compressed = _compress_tool_content(self._diary_output())
        self.assertNotIn("#89", compressed)

    def test_with_ids_declaration_references_survive(self):
        compressed = _compress_tool_content(
            self._diary_output(), essentials=tools.OUTPUT_ESSENTIALS["read_diary"])
        missing = [i for i in range(80, 90) if f"#{i}" not in compressed]
        self.assertFalse(missing, f"ссылки потеряны: {missing}")

    def test_with_urls_declaration_links_survive(self):
        text = "Результаты поиска:\n" + "\n".join(
            f"  {i}. Заголовок номер {i}, достаточно длинный чтобы добить лимит сжатия\n"
            f"     URL: https://example.com/article-{i}" for i in range(1, 12))
        compressed = _compress_tool_content(
            text, essentials=tools.OUTPUT_ESSENTIALS["web_search"])
        self.assertIn("https://example.com/article-11", compressed)

    def test_compression_is_idempotent(self):
        once = _compress_tool_content(self._diary_output(), essentials="ids")
        twice = _compress_tool_content(once, essentials="ids")
        self.assertEqual(once, twice)

    def test_short_output_is_untouched(self):
        text = "Записей в дневнике нет."
        self.assertEqual(_compress_tool_content(text, essentials="ids"), text)




class DeclarationCoversSuccessPaths(unittest.TestCase):
    """Слепое пятно первого сторожа: часть инструментов отдаёт ссылку только
    при УСПЕХЕ, а вызов с пустыми аргументами до него не доходит."""

    def test_writing_tools_declare_their_references(self):
        _seed()
        entry = cs.read_diary(last_n=1)[0]
        goal = cs.list_goals()[0]
        deadline = cs.list_deadlines()[0]
        calls = [
            ("add_goal", {"title": "новая цель"}),
            ("add_deadline", {"title": "новый дедлайн", "due": "2026-12-01"}),
            ("add_diary_entry", {"text": "съел борщ", "tags": ["питание"]}),
            ("mark_goal_done", {"goal_id": goal["id"]}),
            ("postpone_deadline", {"deadline_id": deadline["id"], "new_due": "2026-12-31"}),
            ("mark_deadline_done", {"deadline_id": deadline["id"]}),
            ("delete_diary_entry", {"entry_ids": [entry["id"]]}),
        ]
        undeclared = []
        for name, args in calls:
            out = tools.execute_tool(name, args, None) or ""
            if _REFERENCE_RX.search(out) and tools.OUTPUT_ESSENTIALS.get(name) != "ids":
                undeclared.append((name, out[:70]))
        self.assertFalse(
            undeclared,
            "на успешном пути инструмент отдаёт ссылку #N, но не объявил её: "
            f"{undeclared}",
        )


if __name__ == "__main__":
    unittest.main()
