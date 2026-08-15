"""Полное покрытие coach_storage и смен после переезда на SQLite (15.08.2026).

Задача этих тестов — не «код запускается», а «поведение не изменилось».
Публичный API остался прежним, значит и семантика должна совпадать до мелочей:
какие поля есть в ответе, что возвращается при отсутствии записи, как работает
дедуп, как ведут себя сквозные id.

Отдельный блок — приоритет источников (B1) и журнал изменений (B2): ровно то,
чего не хватило 12.08.2026, когда фото графика с двумя сотрудниками записало
чужие смены, а восстановить причину можно было только вручную по логам.
"""

import sys
import threading
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic import coach_storage
from logic.week_schedule import (
    apply_shifts,
    describe_conflicts,
    get_conflict_policy,
    get_shift,
    get_shift_record,
    pending_conflicts,
    resolve_pending_conflicts,
    save_shifts,
    set_conflict_policy,
    shift_history,
)
from utils import db


class GoalTests(unittest.TestCase):
    def test_add_returns_full_record(self):
        g = coach_storage.add_goal("Спать до 23:59", why="восстановление")
        self.assertEqual(g["title"], "Спать до 23:59")
        self.assertEqual(g["status"], "active")
        self.assertEqual(g["progress_log"], [])
        self.assertIsInstance(g["id"], int)

    def test_ids_are_sequential(self):
        ids = [coach_storage.add_goal(f"цель {i}")["id"] for i in range(3)]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), 3)

    def test_list_filters_by_status(self):
        a = coach_storage.add_goal("активная")
        b = coach_storage.add_goal("закрытая")
        coach_storage.mark_goal_done(b["id"])
        active = coach_storage.list_goals(status="active")
        self.assertEqual([g["id"] for g in active], [a["id"]])

    def test_done_records_note_in_progress_log(self):
        g = coach_storage.add_goal("цель")
        done = coach_storage.mark_goal_done(g["id"], note="закрыл досрочно")
        self.assertEqual(done["status"], "done")
        self.assertEqual(len(done["progress_log"]), 1)
        self.assertEqual(done["progress_log"][0]["note"], "закрыл досрочно")
        self.assertEqual(coach_storage.list_goals()[0]["progress_log"],
                         done["progress_log"])

    def test_done_without_note_leaves_log_empty(self):
        g = coach_storage.add_goal("цель")
        self.assertEqual(coach_storage.mark_goal_done(g["id"])["progress_log"], [])

    def test_missing_goal_returns_none(self):
        self.assertIsNone(coach_storage.mark_goal_done(404))

    def test_title_is_stripped(self):
        self.assertEqual(coach_storage.add_goal("  с пробелами  ")["title"],
                         "с пробелами")


class DeadlineTests(unittest.TestCase):
    def test_add_defaults(self):
        d = coach_storage.add_deadline("Экзамен", "2026-09-01")
        self.assertEqual(d["status"], "pending")
        self.assertEqual(d["importance"], "medium")

    def test_optional_fields_absent_until_set(self):
        """closed/note не должны появляться как null — промпт Iris их читает."""
        d = coach_storage.add_deadline("Экзамен", "2026-09-01")
        self.assertNotIn("closed", d)
        self.assertNotIn("note", d)
        self.assertNotIn("closed", coach_storage.list_deadlines()[0])

    def test_done_sets_closed(self):
        d = coach_storage.add_deadline("Тест", "2026-09-01")
        done = coach_storage.mark_deadline_done(d["id"])
        self.assertEqual(done["status"], "done")
        self.assertIn("closed", done)

    def test_delete_removes_completely(self):
        d = coach_storage.add_deadline("Ошибочный", "2026-09-01")
        removed = coach_storage.delete_deadline(d["id"])
        self.assertEqual(removed["id"], d["id"])
        self.assertEqual(coach_storage.list_deadlines(), [])

    def test_delete_missing_returns_none(self):
        self.assertIsNone(coach_storage.delete_deadline(404))

    def test_update_changes_only_given_fields(self):
        d = coach_storage.add_deadline("Матан", "2026-09-01", importance="high")
        upd = coach_storage.update_deadline(d["id"], due="2026-09-08")
        self.assertEqual(upd["due"], "2026-09-08")
        self.assertEqual(upd["title"], "Матан")
        self.assertEqual(upd["importance"], "high")

    def test_update_missing_returns_none(self):
        self.assertIsNone(coach_storage.update_deadline(404, due="2026-09-08"))

    def test_upcoming_window_filters_by_date(self):
        today = datetime.now().date()
        coach_storage.add_deadline("скоро", (today + timedelta(days=2)).isoformat())
        coach_storage.add_deadline("нескоро", (today + timedelta(days=90)).isoformat())
        coach_storage.add_deadline("прошлый", (today - timedelta(days=5)).isoformat())
        near = coach_storage.list_deadlines(upcoming_days=7)
        self.assertEqual([d["title"] for d in near], ["скоро"])

    def test_broken_date_does_not_break_window(self):
        coach_storage.add_deadline("кривая дата", "не-дата")
        self.assertEqual(coach_storage.list_deadlines(upcoming_days=7), [])
        self.assertEqual(len(coach_storage.list_deadlines()), 1)

    def test_deleting_middle_id_does_not_cause_reuse(self):
        """Удалённый в середине номер не должен достаться новому дедлайну —
        иначе Iris, помнящая «#2 Матан», обратится к чужой записи."""
        a = coach_storage.add_deadline("первый", "2026-09-01")
        b = coach_storage.add_deadline("второй", "2026-09-02")
        coach_storage.delete_deadline(a["id"])
        c = coach_storage.add_deadline("третий", "2026-09-03")
        self.assertNotEqual(c["id"], a["id"])
        self.assertGreater(c["id"], b["id"])

    def test_deleting_the_newest_reuses_its_id_as_before(self):
        """Честно фиксируем сохранённое поведение: max+1 и на JSON, и на SQLite
        переиспользует номер последней удалённой записи. Не регрессия переезда —
        так было всегда, а тест не даёт этому измениться незамеченным."""
        coach_storage.add_deadline("первый", "2026-09-01")
        b = coach_storage.add_deadline("второй", "2026-09-02")
        coach_storage.delete_deadline(b["id"])
        c = coach_storage.add_deadline("третий", "2026-09-03")
        self.assertEqual(c["id"], b["id"])


class DiaryTests(unittest.TestCase):
    def test_add_and_read(self):
        coach_storage.add_diary_entry("Поел салат", tags=["питание"])
        rows = coach_storage.read_diary()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tags"], ["питание"])
        self.assertIn("timestamp", rows[0])

    def test_too_short_text_is_rejected(self):
        self.assertIsNone(coach_storage.add_diary_entry("ок"))
        self.assertIsNone(coach_storage.add_diary_entry("   "))
        self.assertEqual(coach_storage.read_diary(), [])

    def test_exact_duplicate_of_last_is_not_added(self):
        first = coach_storage.add_diary_entry("Поел салат оливье")
        again = coach_storage.add_diary_entry("поел САЛАТ оливье")
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(len(coach_storage.read_diary()), 1)

    def test_duplicate_not_last_is_allowed(self):
        coach_storage.add_diary_entry("Поел салат")
        coach_storage.add_diary_entry("Сходил в зал")
        coach_storage.add_diary_entry("Поел салат")
        self.assertEqual(len(coach_storage.read_diary()), 3)

    def test_structured_data_survives(self):
        payload = {"dish": "Оливье", "kcal": [250, 350], "place": "дом"}
        coach_storage.add_diary_entry("Поел: Оливье", tags=["питание"], data=payload)
        self.assertEqual(coach_storage.read_diary()[0]["data"], payload)

    def test_entries_without_data_have_no_key(self):
        coach_storage.add_diary_entry("Просто запись")
        self.assertNotIn("data", coach_storage.read_diary()[0])

    def test_read_last_n_takes_the_tail(self):
        for i in range(5):
            coach_storage.add_diary_entry(f"запись номер {i}")
        rows = coach_storage.read_diary(last_n=2)
        self.assertEqual([r["text"] for r in rows], ["запись номер 3", "запись номер 4"])

    def test_tag_filter_applies_before_limit(self):
        coach_storage.add_diary_entry("еда раз", tags=["питание"])
        coach_storage.add_diary_entry("спорт", tags=["спорт"])
        coach_storage.add_diary_entry("еда два", tags=["питание"])
        rows = coach_storage.read_diary(last_n=2, tag="питание")
        self.assertEqual([r["text"] for r in rows], ["еда раз", "еда два"])

    def test_delete_returns_only_really_removed(self):
        a = coach_storage.add_diary_entry("первая запись")
        coach_storage.add_diary_entry("вторая запись")
        removed = coach_storage.delete_diary_entries([a["id"], 999])
        self.assertEqual(removed, [a["id"]])
        self.assertEqual(len(coach_storage.read_diary()), 1)

    def test_delete_empty_list_is_noop(self):
        coach_storage.add_diary_entry("запись остаётся")
        self.assertEqual(coach_storage.delete_diary_entries([]), [])
        self.assertEqual(len(coach_storage.read_diary()), 1)

    def test_last_entry_per_tag_takes_freshest(self):
        coach_storage.add_diary_entry("старая еда", tags=["питание"])
        coach_storage.add_diary_entry("что-то", tags=["работа"])
        coach_storage.add_diary_entry("свежая еда", tags=["питание"])
        found = coach_storage.last_entry_per_tag(["питание", "спорт"])
        self.assertEqual(found["питание"]["text"], "свежая еда")
        self.assertNotIn("спорт", found)

    def test_today_tags_and_count(self):
        coach_storage.add_diary_entry("поел", tags=["питание"])
        coach_storage.add_diary_entry("зал", tags=["спорт"])
        self.assertEqual(coach_storage.today_tags(), {"питание", "спорт"})
        self.assertEqual(coach_storage.entries_today(), 2)

    def test_yesterday_entry_not_counted_as_today(self):
        coach_storage.add_diary_entry("вчерашняя запись")
        yesterday = (datetime.now() - timedelta(days=1)).isoformat(timespec="minutes")
        db.execute("UPDATE diary SET ts=?", (yesterday,))
        self.assertEqual(coach_storage.entries_today(), 0)
        self.assertEqual(coach_storage.today_tags(), set())


class PantryTests(unittest.TestCase):
    def test_empty_pantry_shape(self):
        self.assertEqual(coach_storage.get_pantry(), {"items": [], "updated": None})

    def test_add_keeps_order(self):
        coach_storage.pantry_update(add=["Овощи", "Wok Mix", "Рис"])
        self.assertEqual(coach_storage.get_pantry()["items"],
                         ["Овощи", "Wok Mix", "Рис"])

    def test_duplicates_are_normalized_away(self):
        coach_storage.pantry_update(add=["Wok Mix"])
        coach_storage.pantry_update(add=["  wok   mix  ", "Рис"])
        self.assertEqual(coach_storage.get_pantry()["items"], ["Wok Mix", "Рис"])

    def test_remove_is_case_insensitive(self):
        coach_storage.pantry_update(add=["Овощи", "Рис"])
        coach_storage.pantry_update(remove=["ОВОЩИ"])
        self.assertEqual(coach_storage.get_pantry()["items"], ["Рис"])

    def test_remove_missing_is_noop(self):
        coach_storage.pantry_update(add=["Рис"])
        coach_storage.pantry_update(remove=["Гречка"])
        self.assertEqual(coach_storage.get_pantry()["items"], ["Рис"])

    def test_age_days_zero_right_after_update(self):
        coach_storage.pantry_update(add=["Рис"])
        self.assertEqual(coach_storage.pantry_age_days(), 0)

    def test_age_days_none_when_empty(self):
        self.assertIsNone(coach_storage.pantry_age_days())

    def test_age_days_counts_from_stored_date(self):
        coach_storage.pantry_update(add=["Рис"])
        old = (datetime.now() - timedelta(days=9)).strftime("%Y-%m-%d")
        db.execute("UPDATE pantry SET added=?", (old,))
        self.assertEqual(coach_storage.pantry_age_days(), 9)


class DayStateAndPresenceTests(unittest.TestCase):
    def test_fresh_day_state_shape(self):
        state = coach_storage.get_day_state()
        self.assertIn("date", state)
        self.assertEqual(state["pings"], {})

    def test_owner_seen_sets_first_and_last(self):
        coach_storage.mark_owner_seen()
        first = coach_storage.get_day_state()["last_seen"]
        coach_storage.mark_owner_seen()
        state = coach_storage.get_day_state()
        self.assertEqual(state["last_seen"], first)
        self.assertTrue(coach_storage.owner_seen_today())

    def test_ping_is_recorded(self):
        coach_storage.mark_ping("checkin")
        self.assertIn("checkin", coach_storage.get_day_state()["pings"])

    def test_state_resets_on_new_date(self):
        coach_storage.mark_ping("checkin")
        stale = coach_storage.get_day_state()
        stale["date"] = "2020-01-01"
        coach_storage.save_day_state(stale)
        self.assertEqual(coach_storage.get_day_state()["pings"], {})
        self.assertFalse(coach_storage.owner_seen_today())

    def test_wake_is_idempotent_per_day(self):
        """Время фиксируем принудительно: иначе тест проходил бы вхолостую,
        когда прогон случился вне окна пробуждения."""
        real = coach_storage.now_local
        coach_storage.now_local = lambda: real().replace(hour=11, minute=5)
        try:
            first = coach_storage.log_wake_if_first()
            second = coach_storage.log_wake_if_first()
        finally:
            coach_storage.now_local = real
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(first["tags"], ["сон"])
        self.assertTrue(coach_storage.woke_today())
        self.assertEqual(coach_storage.wake_time_today(), "11:05")
        self.assertEqual(len(coach_storage.read_diary()), 1)

    def test_wake_outside_window_is_skipped(self):
        real = coach_storage.now_local
        coach_storage.now_local = lambda: real().replace(hour=3)
        try:
            self.assertIsNone(coach_storage.log_wake_if_first())
            self.assertFalse(coach_storage.woke_today())
        finally:
            coach_storage.now_local = real


class MuteTests(unittest.TestCase):
    def test_default_scope_is_soft(self):
        coach_storage.set_mute(mode="hours", hours=4)
        self.assertTrue(coach_storage.muted_now())
        self.assertFalse(coach_storage.hard_muted_now())

    def test_scope_all_is_hard(self):
        coach_storage.set_mute(mode="hours", hours=4, scope="all")
        self.assertTrue(coach_storage.hard_muted_now())

    def test_unmute_clears_everything(self):
        coach_storage.set_mute(mode="forever", scope="all")
        coach_storage.unmute()
        self.assertFalse(coach_storage.muted_now())
        self.assertFalse(coach_storage.hard_muted_now())

    def test_expired_mute_is_inactive(self):
        past = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="minutes")
        db.kv_set("mute", {"until": past, "scope": "pings"})
        self.assertFalse(coach_storage.muted_now())

    def test_broken_until_is_inactive(self):
        db.kv_set("mute", {"until": "когда-нибудь"})
        self.assertFalse(coach_storage.muted_now())

    def test_hours_are_clamped(self):
        coach_storage.set_mute(mode="hours", hours=9999)
        self.assertTrue(coach_storage.muted_now())


class CountersTests(unittest.TestCase):
    def test_style_index_rotates(self):
        seen = [coach_storage.next_style_index(3) for _ in range(6)]
        self.assertEqual(seen, [0, 1, 2, 0, 1, 2])

    def test_style_index_survives_zero_n(self):
        self.assertEqual(coach_storage.next_style_index(0), 0)

    def test_radar_marks_once(self):
        self.assertFalse(coach_storage.radar_pinged(7))
        coach_storage.mark_radar(7)
        self.assertTrue(coach_storage.radar_pinged(7))
        self.assertFalse(coach_storage.radar_pinged(8))

    def test_gemini_counter_increments(self):
        for expected in (1, 2, 3):
            self.assertEqual(coach_storage.gemini_bump(), expected)
        self.assertEqual(coach_storage.gemini_count_today(), 3)

    def test_gemini_counter_resets_on_new_date(self):
        coach_storage.gemini_bump()
        db.kv_set("gemini_usage", {"date": "2020-01-01", "count": 500})
        self.assertEqual(coach_storage.gemini_count_today(), 0)
        self.assertEqual(coach_storage.gemini_bump(), 1)


class WeekPlanTests(unittest.TestCase):
    def test_empty_by_default(self):
        self.assertEqual(coach_storage.get_week_plan(), {})

    def test_save_and_read(self):
        saved = coach_storage.save_week_plan("  **Четверг:** смена  ")
        self.assertEqual(saved["text"], "**Четверг:** смена")
        self.assertEqual(coach_storage.get_week_plan()["text"], saved["text"])

    def test_save_replaces_previous(self):
        coach_storage.save_week_plan("старый план")
        coach_storage.save_week_plan("новый план")
        self.assertEqual(coach_storage.get_week_plan()["text"], "новый план")


class ShiftPriorityTests(unittest.TestCase):
    """B1: машинная догадка не перетирает то, что сказал человек."""

    D = "2026-08-20"

    def _save(self, source, start="17:00", end="23:00", **kw):
        return save_shifts([{"date": self.D, "start": start, "end": end,
                             "source": source, **kw}])

    def test_photo_creates_when_nothing_known(self):
        self.assertEqual(self._save("photo"), 1)
        self.assertEqual(get_shift(date(2026, 8, 20))["source"], "photo")

    def test_text_overrides_photo(self):
        self._save("photo", start="18:00", end="23:30")
        self.assertEqual(self._save("text", start="16:00"), 1)
        shift = get_shift(date(2026, 8, 20))
        self.assertEqual(shift["start"], "16:00")
        self.assertEqual(shift["source"], "text")

    def test_photo_does_not_override_text(self):
        """Ровно кейс 12.08: правку текстом стирало следующее фото."""
        self._save("text", start="16:00")
        self.assertEqual(self._save("photo", start="18:00"), 0)
        shift = get_shift(date(2026, 8, 20))
        self.assertEqual(shift["start"], "16:00")
        self.assertEqual(shift["source"], "text")

    def test_conflict_is_logged_with_reason(self):
        self._save("text", start="16:00")
        self._save("photo", start="18:00")
        events = [e for e in shift_history(self.D) if e["action"] == "conflict"]
        self.assertEqual(len(events), 1)
        self.assertIn("расходится", events[0]["reason"])

    def test_identical_photo_confirmation_is_not_a_conflict(self):
        """Фото, подтверждающее известную смену, ничего не портит."""
        self._save("text", start="16:00", end="23:00")
        self.assertEqual(self._save("photo", start="16:00", end="23:00"), 1)
        self.assertEqual(get_shift(date(2026, 8, 20))["source"], "photo")

    def test_manual_beats_everything(self):
        self._save("text", start="16:00")
        self.assertEqual(self._save("manual", start="14:00"), 1)
        self.assertEqual(get_shift(date(2026, 8, 20))["start"], "14:00")

    def test_unknown_source_does_not_override_photo(self):
        self._save("photo", start="18:00")
        self.assertEqual(self._save("unknown", start="09:00"), 0)


class ShiftConflictQuestionTests(unittest.TestCase):
    """Молча отклонить фото мало: Влад об этом не узнает и будет думать, что
    график обновился. По умолчанию расхождение превращается в вопрос."""

    D = "2026-08-25"
    DD = date(2026, 8, 25)

    def _text(self, start="16:00"):
        return apply_shifts([{"date": self.D, "start": start, "end": "23:00",
                              "source": "text", "status": "confirmed"}])

    def _photo(self, start="18:00"):
        return apply_shifts([{"date": self.D, "start": start, "end": "23:30",
                              "source": "photo", "status": "planned"}])

    def test_default_policy_is_ask(self):
        self.assertEqual(get_conflict_policy(), "ask")

    def test_conflict_is_reported_not_swallowed(self):
        self._text()
        result = self._photo()
        self.assertEqual(result.saved, 0)
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0].date, self.D)

    def test_question_names_both_versions(self):
        self._text()
        text = describe_conflicts(self._photo().conflicts)
        self.assertIn("18:00", text)   # что в графике
        self.assertIn("16:00", text)   # что сказал Влад
        self.assertIn("вт 25.08", text)

    def test_no_question_when_photo_agrees(self):
        self._text(start="16:00")
        result = apply_shifts([{"date": self.D, "start": "16:00", "end": "23:00",
                                "source": "photo", "status": "confirmed"}])
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.saved, 1)

    def test_no_question_when_nothing_was_said_before(self):
        result = self._photo()
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.saved, 1)

    def test_pending_survives_until_answered(self):
        self._text()
        self._photo()
        self.assertEqual(len(pending_conflicts()), 1)

    def test_answer_photo_applies_it(self):
        self._text()
        self._photo()
        applied = resolve_pending_conflicts("photo")
        self.assertEqual(applied, 1)
        self.assertEqual(get_shift(self.DD)["start"], "18:00")
        self.assertEqual(pending_conflicts(), [])

    def test_answer_photo_is_stored_as_manual(self):
        """Решение владельца — не догадка машины: следующее фото его не тронет."""
        self._text()
        self._photo()
        resolve_pending_conflicts("photo")
        self.assertEqual(get_shift(self.DD)["source"], "manual")
        result = apply_shifts([{"date": self.D, "start": "20:00", "end": "23:00",
                                "source": "photo"}])
        self.assertEqual(result.saved, 0)

    def test_answer_mine_keeps_correction(self):
        self._text()
        self._photo()
        self.assertEqual(resolve_pending_conflicts("mine"), 0)
        self.assertEqual(get_shift(self.DD)["start"], "16:00")
        self.assertEqual(pending_conflicts(), [])

    def test_answer_mine_is_logged(self):
        self._text()
        self._photo()
        resolve_pending_conflicts("mine")
        reasons = [e["reason"] for e in shift_history(self.D)]
        self.assertTrue(any("оставил свою версию" in (r or "") for r in reasons))

    def test_remember_photo_stops_asking(self):
        self._text()
        self._photo()
        resolve_pending_conflicts("photo", remember=True)
        self.assertEqual(get_conflict_policy(), "photo_wins")
        result = apply_shifts([{"date": "2026-08-26", "start": "09:00",
                                "end": "17:00", "source": "text"}])
        self.assertEqual(result.saved, 1)
        second = apply_shifts([{"date": "2026-08-26", "start": "10:00",
                                "end": "18:00", "source": "photo"}])
        self.assertEqual(second.conflicts, [])
        self.assertEqual(second.saved, 1)

    def test_remember_mine_stops_asking_and_keeps_correction(self):
        self._text()
        self._photo()
        resolve_pending_conflicts("mine", remember=True)
        self.assertEqual(get_conflict_policy(), "keep_mine")
        result = self._photo(start="19:00")
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.saved, 0)
        self.assertEqual(get_shift(self.DD)["start"], "16:00")

    def test_unknown_policy_falls_back_to_ask(self):
        set_conflict_policy("что-то странное")
        self.assertEqual(get_conflict_policy(), "ask")

    def test_resolve_without_pending_is_safe(self):
        self.assertEqual(resolve_pending_conflicts("photo"), 0)

    def test_batch_reports_every_conflicting_day(self):
        for d in ("2026-08-27", "2026-08-28"):
            apply_shifts([{"date": d, "start": "16:00", "end": "23:00",
                           "source": "text"}])
        result = apply_shifts([
            {"date": "2026-08-27", "start": "18:00", "end": "23:30", "source": "photo"},
            {"date": "2026-08-28", "start": "19:00", "end": "23:30", "source": "photo"},
            {"date": "2026-08-29", "start": "17:00", "end": "23:00", "source": "photo"},
        ])
        self.assertEqual(result.saved, 1)          # свободный день записался
        self.assertEqual(len(result.conflicts), 2)  # два дня спорные
        self.assertEqual(len(pending_conflicts()), 2)


class ConflictToolTests(unittest.TestCase):
    """Ответ приходит словами в чат, значит должен работать через tool."""

    D = "2026-08-25"

    def setUp(self):
        from logic.tools import execute_tool
        self.run = execute_tool
        apply_shifts([{"date": self.D, "start": "16:00", "end": "23:00",
                       "source": "text", "status": "confirmed"}])
        apply_shifts([{"date": self.D, "start": "18:00", "end": "23:30",
                       "source": "photo", "status": "planned"}])

    def test_take_photo_applies(self):
        reply = self.run("resolve_shift_conflict", {"take": "photo"})
        self.assertIn("Принял график", reply)
        self.assertEqual(get_shift(date(2026, 8, 25))["start"], "18:00")

    def test_take_mine_keeps(self):
        reply = self.run("resolve_shift_conflict", {"take": "mine"})
        self.assertIn("Оставил", reply)
        self.assertEqual(get_shift(date(2026, 8, 25))["start"], "16:00")

    def test_always_sets_policy(self):
        self.run("resolve_shift_conflict", {"take": "photo", "always": True})
        self.assertEqual(get_conflict_policy(), "photo_wins")

    def test_without_always_policy_stays_ask(self):
        self.run("resolve_shift_conflict", {"take": "photo"})
        self.assertEqual(get_conflict_policy(), "ask")

    def test_bad_argument_is_reported(self):
        self.assertIn("Неясно", self.run("resolve_shift_conflict", {"take": "хз"}))

    def test_no_pending_says_so(self):
        self.run("resolve_shift_conflict", {"take": "mine"})
        self.assertIn("нет", self.run("resolve_shift_conflict", {"take": "mine"}))


class ShiftBehaviourTests(unittest.TestCase):
    D = date(2026, 8, 21)
    DS = "2026-08-21"

    def test_invalid_date_is_skipped(self):
        self.assertEqual(save_shifts([{"date": "21.08.2026", "start": "17:00",
                                       "end": "23:00"}]), 0)

    def test_missing_date_is_skipped(self):
        self.assertEqual(save_shifts([{"start": "17:00", "end": "23:00"}]), 0)

    def test_incomplete_time_is_skipped(self):
        self.assertEqual(save_shifts([{"date": self.DS, "start": "17:00"}]), 0)

    def test_confirmed_sets_last_confirmed_at(self):
        save_shifts([{"date": self.DS, "start": "17:00", "end": "23:00",
                      "status": "confirmed", "source": "text"}])
        self.assertIn("last_confirmed_at", get_shift_record(self.D))

    def test_cancelled_hides_from_active_but_stays_visible(self):
        save_shifts([{"date": self.DS, "start": "17:00", "end": "23:00"}])
        save_shifts([{"date": self.DS, "status": "cancelled", "source": "text",
                      "note": "не иду"}])
        self.assertIsNone(get_shift(self.D))
        record = get_shift_record(self.D)
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["note"], "не иду")

    def test_note_absent_when_not_given(self):
        save_shifts([{"date": self.DS, "start": "17:00", "end": "23:00"}])
        self.assertNotIn("note", get_shift_record(self.D))

    def test_missing_shift_returns_none(self):
        self.assertIsNone(get_shift_record(date(2030, 1, 1)))
        self.assertIsNone(get_shift(date(2030, 1, 1)))

    def test_every_write_lands_in_the_journal(self):
        save_shifts([{"date": self.DS, "start": "17:00", "end": "23:00",
                      "source": "photo"}])
        save_shifts([{"date": self.DS, "start": "16:00", "end": "23:00",
                      "source": "text"}])
        actions = [e["action"] for e in shift_history(self.DS)]
        self.assertEqual(actions, ["set", "set"])

    def test_journal_is_per_date(self):
        save_shifts([{"date": self.DS, "start": "17:00", "end": "23:00"}])
        self.assertEqual(shift_history("2026-12-31"), [])

    def test_batch_counts_only_saved(self):
        n = save_shifts([
            {"date": "2026-08-22", "start": "17:00", "end": "23:00"},
            {"date": "кривая", "start": "17:00", "end": "23:00"},
            {"date": "2026-08-23", "start": "18:00", "end": "00:00"},
        ])
        self.assertEqual(n, 2)


class LegacyShiftCompatibilityTests(unittest.TestCase):
    """Смены июня лежали в JSON без метаданных: только start/end.

    Миграция подставляет им status='planned', source='unknown', confidence=
    'medium', updated=''. Старый код на отсутствующих полях подставлял свои
    дефолты — 'planned', 'legacy', 'medium'. Значения разные, решения должны
    совпадать: 'unknown' и 'legacy' обрабатываются одной веткой, а пустой
    updated парсится в None ровно как отсутствующий.
    """

    DS = "2026-06-26"
    D = date(2026, 6, 26)

    def _insert_legacy(self):
        db.execute(
            "INSERT INTO shifts(date, start, end, status, source, confidence, updated)"
            " VALUES(?,?,?,?,?,?,?)",
            (self.DS, "18:00", "00:00", "planned", "unknown", "medium", ""),
        )

    def test_legacy_shift_is_active(self):
        self._insert_legacy()
        self.assertIsNotNone(get_shift(self.D))

    def test_empty_updated_parses_as_absent(self):
        from logic.situation_engine import _parse_iso
        self.assertIsNone(_parse_iso(""))
        self.assertIsNone(_parse_iso(None))

    @staticmethod
    def _situation(now, **over):
        """Смена через два часа — внутри окна, в котором вопрос уместен."""
        from logic.situation_engine import ShiftSituation
        record = {"start": "18:00", "end": "00:00"}
        params = dict(record=record, active_record=record,
                      start_at=now + timedelta(hours=2), end_at=now + timedelta(hours=8),
                      status="planned", source="unknown", confidence="medium",
                      updated_at=None, confirmed_at=None)
        params.update(over)
        return ShiftSituation(**params)

    def test_unknown_source_needs_confirmation_like_legacy(self):
        now = datetime.now().astimezone()
        for source in ("", "unknown", "legacy"):
            self.assertTrue(
                self._situation(now, source=source).needs_confirmation(now), source)

    def test_confirmed_text_shift_does_not_need_confirmation(self):
        now = datetime.now().astimezone()
        sit = self._situation(now, status="confirmed", source="text",
                              confidence="high", updated_at=now, confirmed_at=now)
        self.assertFalse(sit.needs_confirmation(now))

    def test_fresh_text_shift_outside_window_is_not_asked(self):
        now = datetime.now().astimezone()
        sit = self._situation(now, source="text", confidence="high",
                              updated_at=now, start_at=now + timedelta(hours=10))
        self.assertFalse(sit.needs_confirmation(now))


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_diary_writes_keep_unique_ids(self):
        n = 16
        errors = []

        def add(i):
            try:
                coach_storage.add_diary_entry(f"параллельная запись {i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        rows = coach_storage.read_diary(last_n=100)
        self.assertEqual(len(rows), n)
        self.assertEqual(len({r["id"] for r in rows}), n)

    def test_parallel_shift_writes_do_not_deadlock(self):
        errors = []

        def save(i):
            try:
                save_shifts([{"date": f"2026-09-{i + 1:02d}", "start": "17:00",
                              "end": "23:00", "source": "photo"}])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(db.query_one("SELECT COUNT(*) c FROM shifts")["c"], 10)


if __name__ == "__main__":
    unittest.main()
