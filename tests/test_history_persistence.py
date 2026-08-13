"""История диалога и sticky роутера переживают рестарт.

До 13.08.2026 и то и другое жило только в RAM. Рестартов в тот день было пять
подряд, и каждый стирал контекст: бот начинал разговор с чистого листа, а
«лол»/«продолжи» не находили, к кому относятся.
"""

import os
import sys
import tempfile
import threading
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

from logic import agent_router
from utils import db

CHAT = -1001234567890


class HistoryDbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        db.set_db_path(Path("data/hub-test.sqlite"))

    def tearDown(self):
        db.close_all()
        db.set_db_path(db.DEFAULT_DB_PATH)
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_roundtrip_preserves_order(self):
        db.history_add(CHAT, "первое", "ответ 1", agent="Iris")
        db.history_add(CHAT, "второе", "ответ 2", agent="Newser")
        rows = db.history_load(CHAT, 10)
        self.assertEqual([r["user"] for r in rows], ["первое", "второе"])

    def test_limit_takes_the_latest(self):
        for i in range(10):
            db.history_add(CHAT, f"реплика {i}", f"ответ {i}", agent="Iris")
        rows = db.history_load(CHAT, 3)
        self.assertEqual([r["user"] for r in rows],
                         ["реплика 7", "реплика 8", "реплика 9"])

    def test_chats_are_isolated(self):
        db.history_add(CHAT, "в хабе", "ага", agent="Iris")
        db.history_add(777, "в другом чате", "ага", agent="Newser")
        self.assertEqual(len(db.history_load(CHAT, 10)), 1)
        self.assertEqual(db.history_load(777, 10)[0]["user"], "в другом чате")

    def test_trim_keeps_the_tail(self):
        for i in range(20):
            db.history_add(CHAT, f"реплика {i}", "ок", agent="Iris")
        removed = db.history_trim(CHAT, 5)
        self.assertEqual(removed, 15)
        rows = db.history_load(CHAT, 100)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["user"], "реплика 19")

    def test_trim_does_not_touch_other_chats(self):
        for i in range(10):
            db.history_add(CHAT, f"a{i}", "ок", agent="Iris")
        db.history_add(777, "чужое", "ок", agent="Iris")
        db.history_trim(CHAT, 2)
        self.assertEqual(len(db.history_load(777, 10)), 1)

    def test_parallel_writes_all_land(self):
        n = 12
        errors = []

        def add(i):
            try:
                db.history_add(CHAT, f"реплика {i}", "ок", agent="Iris")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=add, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(db.history_load(CHAT, 100)), n)


class StickyRehydrationTests(HistoryDbTests):
    """get_state поднимает состояние роутера из базы после рестарта."""

    def test_empty_db_gives_fresh_state(self):
        state = agent_router.get_state({}, CHAT)
        self.assertEqual(state.recent_messages, [])
        self.assertEqual(state.last_agent_name, "Redmond")

    def test_last_agent_restored(self):
        db.history_add(CHAT, "чек логи", "разобрал", agent="Cipher")
        state = agent_router.get_state({}, CHAT)
        self.assertEqual(state.last_agent_name, "Cipher")

    def test_history_restored_into_recent_messages(self):
        db.history_add(CHAT, "что по еде", "записала обед", agent="Iris")
        state = agent_router.get_state({}, CHAT)
        texts = [m["text"] for m in state.recent_messages]
        self.assertIn("что по еде", texts)
        self.assertIn("записала обед", texts)

    def test_reaction_works_right_after_restart(self):
        """Главный смысл: «лол» сразу после рестарта находит адресата."""
        db.history_add(CHAT, "что по еде", "записала обед", agent="Iris")
        state = agent_router.get_state({}, CHAT)
        orig = agent_router._ask_gemini, agent_router._ask_groq
        agent_router._ask_gemini = lambda s, u: agent_router.NOBODY
        agent_router._ask_groq = lambda s, u, k: ""
        try:
            agent, _ = agent_router.route("лол", state, "key")
        finally:
            agent_router._ask_gemini, agent_router._ask_groq = orig
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Iris")

    def test_existing_state_is_not_reloaded(self):
        db.history_add(CHAT, "старое", "ответ", agent="Cipher")
        states = {CHAT: agent_router.RouterState()}
        states[CHAT].last_agent_name = "Newser"
        state = agent_router.get_state(states, CHAT)
        self.assertEqual(state.last_agent_name, "Newser")

    def test_broken_db_does_not_raise(self):
        db.set_db_path(Path("/нет/такого/пути/hub.sqlite"))
        state = agent_router.get_state({}, CHAT)
        self.assertIsNotNone(state)


if __name__ == "__main__":
    unittest.main()
