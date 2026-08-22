"""Учебное расписание живёт в данных и имеет срок годности.

До 21.08.2026 расписание было константой `STUDY_TIMETABLE` в исходнике
(«SoSe 2026, HRW Campus Bottrop»). Владелец 17.08 сообщил боту, что у него
семестрфериен; запись легла в дневник — и не изменила ничего, потому что
источник утверждения лежал вне данных. Бот продолжал звать на пары
17, 18 и 19 августа, а в промпт попадали оба факта разом.

Инвариант, который проверяется здесь: **утверждение о жизни владельца живёт
в данных, а не в коде, и перестаёт действовать по истечении срока.**
"""

import unittest
from datetime import date, timedelta

from logic import week_schedule as ws
from utils import db


def _seed(valid_from, valid_to=None):
    """Расписание «как было в SoSe 2026» с заданным сроком действия."""
    return ws.set_timetable(
        [
            (0, "14:05", "15:45", "Лекция Grundlagen der Ingenieurmathematik (Ботроп)", "lecture"),
            (1, "12:20", "14:00", "Лекция Ingenieurmathematik (Ботроп)", "lecture"),
            (2, "13:15", "14:50", "Домашняя учёба", "home_study"),
            (3, "", "", "Домашняя учёба", "home_study"),
            (4, "08:00", "09:35", "Лекция Projektmanagement (Ботроп)", "lecture"),
        ],
        valid_from=valid_from,
        valid_to=valid_to,
    )


class TimetableIsData(unittest.TestCase):
    def test_empty_db_means_no_classes(self):
        """Никакого расписания «по умолчанию» из кода быть не должно."""
        for offset in range(7):
            d = date(2026, 8, 17) + timedelta(days=offset)
            self.assertEqual(
                ws.study_slots(d), [],
                f"на пустой базе {d} не должно быть пар — иначе источник в коде",
            )

    def test_active_timetable_is_visible(self):
        _seed(valid_from="2026-04-01", valid_to="2026-07-15")
        monday = date(2026, 5, 4)
        slots = ws.study_slots(monday)
        self.assertEqual(len(slots), 1)
        self.assertIn("Ingenieurmathematik", slots[0][2])

    def test_expired_timetable_is_invisible(self):
        """Главный инвариант: истёкшее расписание не попадает никуда."""
        _seed(valid_from="2026-04-01", valid_to="2026-07-15")
        for d, what in [
            (date(2026, 8, 17), "понедельник — бот звал на лекцию в 14:05"),
            (date(2026, 8, 18), "вторник — «учёба в Ботропе уже позади»"),
            (date(2026, 8, 19), "среда — «домашняя учёба с часу»"),
        ]:
            self.assertEqual(ws.study_slots(d), [], what)

    def test_future_timetable_is_invisible(self):
        today = date(2026, 8, 17)
        _seed(valid_from="2026-10-01")
        self.assertEqual(ws.study_slots(today), [])

    def test_expire_closes_the_running_timetable(self):
        """«У меня каникулы» должно уметь закрыть действующее расписание."""
        _seed(valid_from="2026-04-01")
        self.assertTrue(ws.study_slots(date(2026, 8, 17)))
        ws.expire_timetable("2026-08-16")
        self.assertEqual(ws.study_slots(date(2026, 8, 17)), [])

    def test_home_study_day_comes_from_data(self):
        self.assertFalse(ws.is_home_study_day(date(2026, 8, 19)))
        _seed(valid_from="2026-04-01")
        self.assertTrue(ws.is_home_study_day(date(2026, 8, 19)))   # среда
        self.assertTrue(ws.is_home_study_day(date(2026, 8, 20)))   # четверг
        self.assertFalse(ws.is_home_study_day(date(2026, 8, 21)))  # пятница


class ExpiredTimetableLeavesTheContext(unittest.TestCase):
    """Интеграция: истёкшее расписание не должно доезжать до промпта и пингов."""

    def setUp(self):
        import datetime as _dt
        import utils.time as ut
        self._real = ut.now_local
        moment = _dt.datetime(2026, 8, 17, 13, 30, tzinfo=ut.OWNER_TZ)
        for mod in ("utils.time", "logic.coach_storage", "logic.priorities",
                    "logic.week_schedule", "logic.situation_engine"):
            __import__(mod)
        import logic.coach_storage as cs
        import logic.priorities as pr
        import logic.situation_engine as se
        self._patched = [(ut, "now_local"), (cs, "now_local"),
                         (pr, "now_local"), (ws, "now_local"), (se, "now_local")]
        self._saved = [(m, n, getattr(m, n, None)) for m, n in self._patched]
        for m, n in self._patched:
            if hasattr(m, n):
                setattr(m, n, lambda: moment)

    def tearDown(self):
        for m, n, old in self._saved:
            if old is not None:
                setattr(m, n, old)

    def test_priorities_block_has_no_lecture_after_semester_ended(self):
        _seed(valid_from="2026-04-01", valid_to="2026-07-15")
        from logic.priorities import build_priorities_block
        block = build_priorities_block()
        self.assertNotIn("Лекция", block)
        self.assertNotIn("Ботроп", block)

    def test_pings_are_not_suppressed_by_a_lecture_that_ended(self):
        """13:30 понедельника попадало в окно «за 30 мин до пары 14:05»."""
        _seed(valid_from="2026-04-01", valid_to="2026-07-15")
        from logic.situation_engine import build_day_situation
        self.assertFalse(build_day_situation().in_study_block)




class SeedMigration(unittest.TestCase):
    """Перенос расписания из кода в данные — разовый и закрытый по сроку."""

    def test_seed_lands_closed_so_holidays_are_not_overridden(self):
        n = ws.seed_timetable_if_empty()
        self.assertGreater(n, 0)
        rows = ws.timetable_rows()
        self.assertTrue(all(r["valid_to"] == ws.SEED_VALID_TO for r in rows))
        # После переноса бот не зовёт на пары в дни, на которые жаловался владелец
        for d in (date(2026, 8, 17), date(2026, 8, 18), date(2026, 8, 19),
                  date(2026, 8, 21)):
            self.assertEqual(ws.study_slots(d), [], f"{d}: пары не должны воскреснуть")

    def test_seed_is_idempotent(self):
        first = ws.seed_timetable_if_empty()
        second = ws.seed_timetable_if_empty()
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(len(ws.timetable_rows()), first)

    def test_history_stays_answerable(self):
        ws.seed_timetable_if_empty()
        may_monday = date(2026, 5, 4)
        self.assertTrue(ws.study_slots(may_monday),
                        "прошлое расписание должно оставаться видимым в своём интервале")


if __name__ == "__main__":
    unittest.main()
