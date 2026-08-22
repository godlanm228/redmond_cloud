"""Менять можно только то, что модель видела в этой генерации.

Инцидент 17.08.2026 целиком:

    10:01:31  read_diary(last_n=10)          ← листинг прочитан
    10:02:16  add_diary_entry('Позанимался спортом…')
    10:02:32  Влад: «Ты говори мне, что записываешь»
    10:03:21  Влад: «Не позанимался еще, это план на тудей»
    10:03:31  delete_diary_entry(entry_ids=[3])   ← порядковый номер вместо id

К этому моменту листинг уже вытеснился из контекста, id модель не помнила
и назвала «третью». Запись #3 — месячной давности, к спорту отношения не
имела — исчезла. Ошибочная #88 осталась на месте и висела там четыре дня.

Сжатие теперь сохраняет ссылки (см. test_output_essentials), то есть
ПРИЧИНА закрыта. Здесь — структурная защита на случай, когда модель всё
равно называет номер, которого не видела: обращение к записи по id
требует, чтобы этот id пришёл из чтения в этой же генерации.
"""

import unittest

from logic import coach_storage as cs
from logic import tools


class WithoutReadingNothingChanges(unittest.TestCase):
    def setUp(self):
        self.entries = [cs.add_diary_entry(f"Старая запись номер {n}")
                        for n in range(1, 6)]
        self.session = tools.ToolSession()

    def test_delete_by_unseen_id_is_refused(self):
        """Воспроизведение 17.08: модель называет «3», ничего не прочитав."""
        victim = self.entries[2]["id"]
        out = tools.execute_tool("delete_diary_entry", {"entry_ids": [3]},
                                 None, session=self.session)

        survived = {e["id"] for e in cs.read_diary(last_n=50)}
        self.assertIn(victim, survived,
                      "удалена запись, которую модель в этой генерации не видела")
        self.assertNotIn("Удалила", out)

    def test_refusal_tells_the_model_what_to_do(self):
        out = tools.execute_tool("delete_diary_entry", {"entry_ids": [3]},
                                 None, session=self.session)
        self.assertIn("read_diary", out,
                      "отказ должен подсказывать, как получить настоящий id")

    def test_after_reading_the_same_id_deletion_works(self):
        tools.execute_tool("read_diary", {"last_n": 10}, None, session=self.session)
        target = self.entries[2]["id"]
        out = tools.execute_tool("delete_diary_entry", {"entry_ids": [target]},
                                 None, session=self.session)
        self.assertIn("Удалила", out)
        self.assertNotIn(target, {e["id"] for e in cs.read_diary(last_n=50)})

    def test_just_created_record_counts_as_seen(self):
        """Модель сама создала запись и знает её номер — перечитывать незачем."""
        created = tools.execute_tool("add_diary_entry", {"text": "съел борщ"},
                                     None, session=self.session)
        new_id = cs.read_diary(last_n=1)[0]["id"]
        self.assertIn(f"#{new_id}", created)
        out = tools.execute_tool("delete_diary_entry", {"entry_ids": [new_id]},
                                 None, session=self.session)
        self.assertIn("Удалила", out)


class GuardCoversEveryAddressingTool(unittest.TestCase):
    """Защищено не только удаление дневника: любое обращение к существующей
    записи по номеру может попасть не в ту."""

    def setUp(self):
        self.session = tools.ToolSession()
        self.goal = cs.add_goal("Лечь спать до 23:59")
        self.deadline = cs.add_deadline("Матан", "2026-09-01")
        self.entry = cs.add_diary_entry("Позанимался спортом")

    def test_all_addressing_tools_refuse_unseen_ids(self):
        calls = [
            ("delete_diary_entry", {"entry_ids": [self.entry["id"]]}),
            ("delete_deadline", {"deadline_id": self.deadline["id"]}),
            ("postpone_deadline", {"deadline_id": self.deadline["id"],
                                   "new_due": "2026-12-31"}),
            ("mark_deadline_done", {"deadline_id": self.deadline["id"]}),
            ("mark_goal_done", {"goal_id": self.goal["id"]}),
        ]
        for name, args in calls:
            out = tools.execute_tool(name, args, None, session=tools.ToolSession())
            self.assertIn("не видел", out, f"{name} принял id без чтения: {out!r}")

    def test_declared_guard_matches_the_tool_list(self):
        known = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
        unknown = sorted(set(tools.ADDRESSED_RECORDS) - known)
        self.assertFalse(unknown, f"защита объявлена для несуществующих: {unknown}")


class LegacyCallsAreNotBroken(unittest.TestCase):
    def test_without_session_behaviour_is_unchanged(self):
        """Прямые вызовы без сессии (тесты, разовые скрипты) не ломаются:
        защита включается там, где есть контекст генерации."""
        entry = cs.add_diary_entry("запись")
        out = tools.execute_tool("delete_diary_entry", {"entry_ids": [entry["id"]]}, None)
        self.assertIn("Удалила", out)




class GuardIsActuallyWiredIn(unittest.TestCase):
    """Защита работает только если боевой цикл передаёт сессию.

    Без этого её можно отключить целиком, убрав один именованный аргумент,
    и ни один тест выше этого не заметит: они зовут execute_tool напрямую.
    """

    def test_every_production_call_passes_a_session(self):
        import ast
        import pathlib

        src = pathlib.Path("logic/response_generator.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if called != "execute_tool":
                continue
            if not any(kw.arg == "session" for kw in node.keywords):
                bad.append(node.lineno)
        self.assertFalse(
            bad,
            f"execute_tool вызывается без session= в строках {bad} — "
            f"защита «менять только виденное» там не работает",
        )

    def test_both_provider_loops_create_a_session(self):
        import pathlib
        src = pathlib.Path("logic/response_generator.py").read_text(encoding="utf-8")
        self.assertEqual(
            src.count("ToolSession()"), 2,
            "сессия должна создаваться в обоих циклах — Groq и Gemini",
        )


if __name__ == "__main__":
    unittest.main()
