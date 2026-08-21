"""Cipher: сессии, срезание триггеров, разбор вывода CLI.

Разбор 12.08.2026. Каждое сообщение запускало `claude -p` заново: он не помнил
ни своего вопроса, ни ответа Влада. На «Даю разрешение» уже не знал, о чём
речь. Плюс в задачу попадал мусор «_redberry_bot» — обрезался короткий триггер
вместо полного @username.
"""

import sys
import unittest
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import cipher_wrapper as cw
from logic.agents import CIPHER, IRIS, NEWSER, REDMOND, find_by_trigger
from utils import db
from utils.time import now_local

CHAT = -1001234567890


class TriggerStrippingTests(unittest.TestCase):
    """Полный @username должен срезаться целиком, а не по короткому префиксу."""

    def test_full_username_leaves_no_tail(self):
        self.assertEqual(
            CIPHER.strip_trigger("@cipher_redberry_bot чек логи"), "чек логи")

    def test_short_trigger_still_works(self):
        self.assertEqual(CIPHER.strip_trigger("@cipher чек логи"), "чек логи")
        self.assertEqual(CIPHER.strip_trigger("@c чек логи"), "чек логи")

    def test_every_agent_username_strips_clean(self):
        for agent in (REDMOND, IRIS, NEWSER, CIPHER):
            text = f"@{agent.bot_username} задача"
            self.assertEqual(agent.strip_trigger(text), "задача", agent.name)

    def test_name_alias_still_strips(self):
        self.assertEqual(IRIS.strip_trigger("айрис, что по еде"), "что по еде")

    def test_routing_by_full_username_finds_agent(self):
        found = find_by_trigger("@cipher_redberry_bot чек логи")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Cipher")

    def test_multiline_task_keeps_body(self):
        text = "@cipher_redberry_bot не применяй фиксы\nсделай диагностику"
        self.assertEqual(CIPHER.strip_trigger(text),
                         "не применяй фиксы\nсделай диагностику")


class SessionDecisionTests(unittest.TestCase):
    def _remember(self, session_id, minutes_ago=0, message_id=None):
        cw.remember_session(CHAT, session_id, topic="тест", message_id=message_id)
        if minutes_ago:
            ts = (now_local() - timedelta(minutes=minutes_ago)).isoformat(timespec="minutes")
            db.execute("UPDATE cipher_sessions SET updated=? WHERE session_id=?",
                       (ts, session_id))

    def test_first_task_starts_fresh(self):
        session, reason = cw.decide_session(CHAT)
        self.assertIsNone(session)
        self.assertIn("перв", reason)

    def test_reply_wins_over_everything(self):
        self._remember("old-session", minutes_ago=1)
        self._remember("replied-session", minutes_ago=600, message_id=777)
        session, reason = cw.decide_session(CHAT, reply_to_message_id=777)
        self.assertEqual(session, "replied-session")
        self.assertEqual(reason, "reply")

    def test_fresh_conversation_continues(self):
        self._remember("s1", minutes_ago=5)
        session, reason = cw.decide_session(CHAT)
        self.assertEqual(session, "s1")
        self.assertIn("продолжение", reason)

    def test_stale_conversation_still_resumes_but_flagged(self):
        self._remember("s1", minutes_ago=200)
        session, reason = cw.decide_session(CHAT)
        self.assertEqual(session, "s1")
        self.assertIn("решит сам", reason)

    def test_older_than_a_day_starts_fresh(self):
        self._remember("s1", minutes_ago=60 * 25)
        session, reason = cw.decide_session(CHAT)
        self.assertIsNone(session)
        self.assertIn("24", reason)

    def test_latest_session_wins_among_several(self):
        self._remember("old", minutes_ago=20)
        self._remember("new", minutes_ago=2)
        session, _ = cw.decide_session(CHAT)
        self.assertEqual(session, "new")

    def test_reply_to_unknown_message_falls_back_to_time(self):
        self._remember("s1", minutes_ago=3)
        session, reason = cw.decide_session(CHAT, reply_to_message_id=424242)
        self.assertEqual(session, "s1")
        self.assertIn("продолжение", reason)

    def test_other_chat_does_not_leak(self):
        self._remember("s1", minutes_ago=2)
        session, _ = cw.decide_session(999)
        self.assertIsNone(session)

    def test_bind_message_enables_reply_lookup(self):
        cw.remember_session(CHAT, "s1", topic="тема")
        cw.bind_message(CHAT, "s1", 555)
        self.assertEqual(cw.session_for_reply(CHAT, 555), "s1")

    def test_bind_is_idempotent(self):
        cw.remember_session(CHAT, "s1")
        cw.bind_message(CHAT, "s1", 555)
        cw.bind_message(CHAT, "s1", 556)
        rows = db.query("SELECT * FROM cipher_sessions WHERE session_id='s1'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(cw.session_for_reply(CHAT, 556), "s1")

    def test_remember_twice_updates_not_duplicates(self):
        cw.remember_session(CHAT, "s1", topic="раз")
        cw.remember_session(CHAT, "s1", topic="два")
        rows = db.query("SELECT * FROM cipher_sessions WHERE session_id='s1'")
        self.assertEqual(len(rows), 1)

    def test_empty_session_id_is_ignored(self):
        cw.remember_session(CHAT, "")
        self.assertEqual(db.query("SELECT * FROM cipher_sessions"), [])


class OutputParsingTests(unittest.TestCase):
    def test_json_result_and_session(self):
        raw = '{"type":"result","result":"готово","session_id":"abc","is_error":false}'
        text, session, err = cw._parse_output(raw)
        self.assertEqual((text, session, err), ("готово", "abc", False))

    def test_error_flag_is_read(self):
        raw = '{"result":"Not logged in","session_id":"x","is_error":true}'
        _, _, err = cw._parse_output(raw)
        self.assertTrue(err)

    def test_plain_text_output_still_works(self):
        """Формат вывода CLI может смениться — Cipher не должен из-за этого умереть."""
        text, session, err = cw._parse_output("просто текст")
        self.assertEqual(text, "просто текст")
        self.assertEqual(session, "")
        self.assertFalse(err)

    def test_empty_output_is_an_error(self):
        self.assertEqual(cw._parse_output(""), ("", "", True))

    def test_json_array_is_treated_as_text(self):
        text, session, _ = cw._parse_output("[1, 2, 3]")
        self.assertEqual(text, "[1, 2, 3]")
        self.assertEqual(session, "")


class SystemPromptTests(unittest.TestCase):
    """Приписка чинит два конкретных провала 12.08."""

    def test_says_no_ssh_needed(self):
        self.assertIn("ssh", cw.SYSTEM_APPENDIX.lower())
        self.assertIn("уже на этой машине", cw.SYSTEM_APPENDIX)

    def test_gives_correct_log_path(self):
        self.assertIn("logs/v2.log", cw.SYSTEM_APPENDIX)

    def test_points_at_the_project_reference(self):
        self.assertIn("docs/ARCHITECTURE.md", cw.SYSTEM_APPENDIX)

    def test_forbids_markdown_tables(self):
        self.assertIn("markdown-таблиц", cw.SYSTEM_APPENDIX)

    def test_points_at_sqlite_not_json(self):
        # Переносы строк в тексте промпта не должны ломать проверку смысла.
        flat = " ".join(cw.SYSTEM_APPENDIX.split())
        self.assertIn("memory.sqlite", flat)
        self.assertIn("замороженный архив", flat)


class LockTests(unittest.TestCase):
    def test_no_lock_by_default(self):
        self.assertIsNone(cw._get_lock())

    def test_lock_is_reported(self):
        cw._set_lock(2)
        self.assertIsNotNone(cw._get_lock())

    def test_expired_lock_is_ignored(self):
        past = (now_local() - timedelta(hours=1)).isoformat(timespec="minutes")
        db.kv_set("cipher_lock", {"locked_until": past})
        self.assertIsNone(cw._get_lock())

    def test_reset_hours_parsed_from_message(self):
        hours = cw._parse_reset_hours("Your limit resets 9:30pm (UTC)")
        self.assertGreater(hours, 0)
        self.assertLessEqual(hours, 12)

    def test_unparsable_reset_falls_back_to_one_hour(self):
        self.assertEqual(cw._parse_reset_hours("что-то непонятное"), 1.0)


if __name__ == "__main__":
    unittest.main()


class AuthStatusTests(unittest.TestCase):
    """13–15.08.2026 Cipher был мёртв трое суток, и это никак не всплывало."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / ".credentials.json"
        self._orig = cw.CREDENTIALS_PATH
        cw.CREDENTIALS_PATH = self.path

    def tearDown(self):
        cw.CREDENTIALS_PATH = self._orig
        self.tmp.cleanup()

    def _write(self, payload):
        import json as _json
        self.path.write_text(_json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _in_days(days):
        from datetime import datetime as _dt
        return int((_dt.now() + timedelta(days=days)).timestamp() * 1000)

    def test_missing_file_is_the_reported_failure(self):
        status = cw.auth_status()
        self.assertFalse(status["ok"])
        self.assertIn("/login", status["reason"])

    def test_healthy_token_is_ok_and_quiet(self):
        self._write({"claudeAiOauth": {"refreshTokenExpiresAt": self._in_days(28)}})
        status = cw.auth_status()
        self.assertTrue(status["ok"])
        self.assertEqual(status["reason"], "")

    def test_expiring_soon_warns_without_breaking(self):
        self._write({"claudeAiOauth": {"refreshTokenExpiresAt": self._in_days(2)}})
        status = cw.auth_status()
        self.assertTrue(status["ok"])
        self.assertIn("/login", status["reason"])

    def test_expired_refresh_is_a_failure(self):
        self._write({"claudeAiOauth": {"refreshTokenExpiresAt": self._in_days(-1)}})
        self.assertFalse(cw.auth_status()["ok"])

    def test_unreadable_file_is_a_failure(self):
        self.path.write_text("{битый", encoding="utf-8")
        self.assertFalse(cw.auth_status()["ok"])

    def test_unknown_format_does_not_cry_wolf(self):
        """Формат файла сменится — это не повод объявлять Cipher мёртвым."""
        self._write({"что-то": "новое"})
        status = cw.auth_status()
        self.assertTrue(status["ok"])

    def test_seconds_timestamps_also_understood(self):
        from datetime import datetime as _dt
        secs = int((_dt.now() + timedelta(days=20)).timestamp())
        self._write({"claudeAiOauth": {"refreshTokenExpiresAt": secs}})
        self.assertTrue(cw.auth_status()["ok"])
        self.assertGreater(cw.auth_status()["expires_in_days"], 19)

    def test_healthcheck_maps_missing_to_gone(self):
        from utils.model_healthcheck import GONE, check_cipher_auth
        status, detail = check_cipher_auth()
        self.assertEqual(status, GONE)
        self.assertIn("/login", detail)

    def test_healthcheck_maps_healthy_to_ok(self):
        from utils.model_healthcheck import OK, check_cipher_auth
        self._write({"claudeAiOauth": {"refreshTokenExpiresAt": self._in_days(28)}})
        status, _ = check_cipher_auth()
        self.assertEqual(status, OK)

    def test_token_values_are_never_read(self):
        """Читаем только сроки. Токены в память не тянем — незачем."""
        self._write({"claudeAiOauth": {"accessToken": "СЕКРЕТ",
                                       "refreshToken": "ТОЖЕ-СЕКРЕТ",
                                       "refreshTokenExpiresAt": self._in_days(10)}})
        status = cw.auth_status()
        self.assertNotIn("СЕКРЕТ", repr(status))
        self.assertNotIn("ТОЖЕ-СЕКРЕТ", repr(status))
