"""«Прочитал → поменял → записал» под транзакцией.

Докстринг coach_storage объявляет этот класс закрытым переходом на SQLite:
«прочитал → поменял → записал без транзакции теряет параллельные записи».
На деле пять функций так и остались без транзакции — аудит 21.08.2026:
mark_goal_done, mark_deadline_done, delete_deadline, update_deadline и
add_diary_entry (две последних закрыты раньше).

Гонка воспроизводится принудительно: чтение притормаживается, чтобы оба
потока успели прочитать одно состояние до того, как кто-то начнёт писать.
Без транзакции второй писатель затирает правку первого. С BEGIN IMMEDIATE
второй ждёт коммита и читает уже обновлённую строку.
"""

import threading
import time
import unittest

from logic import coach_storage as cs
from utils import db


class _SlowRead:
    """Заставляет читателей перекрыться по времени — иначе гонка ловится
    случайно и тест мигает."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self._orig = None

    def __enter__(self):
        self._orig = db.query_one

        def slow(sql, params=()):
            row = self._orig(sql, params)
            time.sleep(self.delay)
            return row

        db.query_one = slow
        return self

    def __exit__(self, *exc):
        db.query_one = self._orig


def _run(*fns):
    threads = [threading.Thread(target=f) for f in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class ParallelEditsDoNotOverwriteEachOther(unittest.TestCase):
    def test_two_fields_updated_at_once_both_survive(self):
        d = cs.add_deadline("Матан", "2026-09-01")
        with _SlowRead():
            _run(lambda: cs.update_deadline(d["id"], title="Матан переименован"),
                 lambda: cs.update_deadline(d["id"], due="2026-12-31"))
        after = cs.list_deadlines()[0]
        self.assertEqual(after["title"], "Матан переименован",
                         "правка заголовка затёрта параллельной правкой срока")
        self.assertEqual(after["due"], "2026-12-31",
                         "правка срока затёрта параллельной правкой заголовка")

    def test_progress_notes_are_not_lost(self):
        g = cs.add_goal("Лечь спать до 23:59")
        with _SlowRead():
            _run(lambda: cs.mark_goal_done(g["id"], note="первая заметка"),
                 lambda: cs.mark_goal_done(g["id"], note="вторая заметка"))
        log = cs.list_goals()[0]["progress_log"]
        notes = [e.get("note") for e in log]
        self.assertIn("первая заметка", notes)
        self.assertIn("вторая заметка", notes)

    def test_nobody_reports_success_over_a_row_that_is_gone(self):
        """Два одновременных удаления одной записи: удалить её можно только
        один раз, значит и отчитаться об удалении — тоже один раз.

        (Пара «удалить + закрыть» таким тестом не годится: под транзакцией
        операции сериализуются, и закрыть, а потом удалить — законная
        последовательность, где обе работают с реальной строкой.)
        """
        d = cs.add_deadline("Матан", "2026-09-01")
        results = []
        with _SlowRead():
            _run(lambda: results.append(cs.delete_deadline(d["id"])),
                 lambda: results.append(cs.delete_deadline(d["id"])))
        succeeded = [r for r in results if r is not None]
        self.assertEqual(len(succeeded), 1,
                         "об удалении одной записи отчитались дважды")
        self.assertEqual(cs.list_deadlines(), [])




class DedupSurvivesConcurrency(unittest.TestCase):
    def test_same_entry_written_twice_at_once_lands_once(self):
        """Антидубль читал «последнюю запись» вне транзакции: два
        одновременных вызова видели одно состояние, и в дневник ложился
        двойник."""
        with _SlowRead():
            _run(lambda: cs.add_diary_entry("Поел гречку с курицей"),
                 lambda: cs.add_diary_entry("Поел гречку с курицей"))
        entries = cs.read_diary(last_n=50)
        self.assertEqual(len(entries), 1, f"в дневник лёг дубль: {entries}")


if __name__ == "__main__":
    unittest.main()
