"""Роутинг: кто отвечает и когда не отвечает никто.

Разбор 12.08.2026: два промаха подряд («почему гемини упал?» → Redmond полез
искать курс валют; «Даю разрешение» → Newser ответил «это к Cipher»). Причина
общая — правильного ответа не было в меню: Cipher был исключён из роутинга,
реплаи не читались, а варианта «промолчать» не существовало вовсе.
"""

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

from logic import agent_router
from logic.agent_router import NOBODY, RouterState, _is_short_followup, route


class RouterTestBase(unittest.TestCase):
    def setUp(self):
        self.llm_calls = []
        self._orig_gemini = agent_router._ask_gemini
        self._orig_groq = agent_router._ask_groq

    def tearDown(self):
        agent_router._ask_gemini = self._orig_gemini
        agent_router._ask_groq = self._orig_groq

    def _llm(self, reply):
        """Подменяем оба провайдера роутера и считаем обращения."""
        def gem(system, user_msg):
            self.llm_calls.append(user_msg)
            return reply
        agent_router._ask_gemini = gem
        agent_router._ask_groq = lambda s, u, k: ""


class ReplyOverrideTests(RouterTestBase):
    def test_reply_to_cipher_goes_to_cipher_without_llm(self):
        self._llm("Newser")  # роутер сказал бы Newser — не должно спрашиваться
        state = RouterState()
        agent, _ = route("Так ты должен был прочекать логи. Даю разрешение",
                         state, "key", reply_to_agent="Cipher")
        self.assertEqual(agent.name, "Cipher")
        self.assertEqual(self.llm_calls, [])

    def test_reply_to_cipher_updates_sticky(self):
        self._llm("Newser")
        state = RouterState()
        route("продолжай разбор", state, "key", reply_to_agent="Cipher")
        self.assertEqual(state.last_agent_name, "Cipher")

    def test_explicit_name_beats_reply(self):
        """Реплай на Newser + «айрис, глянь» → Айрис. Ровно кейс Влада."""
        self._llm("Newser")
        state = RouterState()
        agent, _ = route("айрис, глянь что там по калориям",
                         state, "key", reply_to_agent="Newser")
        self.assertEqual(agent.name, "Iris")

    def test_reply_to_non_cipher_is_only_a_hint(self):
        """Не Cipher — реплай уходит в промпт подсказкой, решает всё равно LLM."""
        self._llm("Iris")
        state = RouterState()
        agent, _ = route("а что по еде", state, "key", reply_to_agent="Newser")
        self.assertEqual(agent.name, "Iris")
        self.assertIn("Newser", self.llm_calls[0])


class ShortFollowupTests(RouterTestBase):
    def test_markers_are_followups(self):
        for t in ("да", "ага", "продолжи", "Даю разрешение", "ок", "понял"):
            self.assertTrue(_is_short_followup(t), t)

    def test_topic_change_is_not_a_followup(self):
        """Короткое, но меняет тему — обязано идти в классификатор."""
        for t in ("а погода?", "курс биткоина", "что по учёбе", "найди отель"):
            self.assertFalse(_is_short_followup(t), t)

    def test_long_text_is_never_a_followup(self):
        self.assertFalse(_is_short_followup("да, но только если это не сломает график"))

    def test_followup_sticks_to_last_agent_without_llm(self):
        self._llm("Newser")
        state = RouterState()
        state.last_agent_name = "Cipher"
        agent, _ = route("даю разрешение", state, "key")
        self.assertEqual(agent.name, "Cipher")
        self.assertEqual(self.llm_calls, [])

    def test_followup_without_history_falls_through_to_llm(self):
        self._llm("Redmond")
        state = RouterState()
        state.last_agent_name = ""
        route("да", state, "key")
        self.assertEqual(len(self.llm_calls), 1)


class ReactionTests(RouterTestBase):
    """«лол» после реплики агента — реакция на неё, а не мысли вслух."""

    def _state_with_agent_reply(self, agent="Iris"):
        state = RouterState()
        state.add("user", "что там по еде", agent)
        state.add("assistant", "записала обед", agent)
        state.last_agent_name = agent
        return state

    def test_reaction_after_agent_reply_goes_to_that_agent(self):
        self._llm(NOBODY)  # LLM бы промолчал — до неё дойти не должно
        agent, _ = route("лол", self._state_with_agent_reply("Iris"), "key")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Iris")
        self.assertEqual(self.llm_calls, [])

    def test_reaction_without_agent_reply_goes_to_llm(self):
        self._llm(NOBODY)
        state = RouterState()
        state.add("user", "просто пишу себе", "Iris")
        agent, _ = route("лол", state, "key")
        self.assertIsNone(agent)
        self.assertEqual(len(self.llm_calls), 1)

    def test_laughter_variants_are_reactions(self):
        for t in ("ахахах", "хахаха", "ахахахаха", "hahaha", "лоооол", "ЛОЛ!!"):
            state = self._state_with_agent_reply("Newser")
            self._llm(NOBODY)
            agent, _ = route(t, state, "key")
            self.assertIsNotNone(agent, t)
            self.assertEqual(agent.name, "Newser", t)

    def test_smileys_are_reactions(self):
        for t in (")", "))", ":)", ";-)", "=)"):
            state = self._state_with_agent_reply()
            self._llm(NOBODY)
            agent, _ = route(t, state, "key")
            self.assertIsNotNone(agent, t)

    def test_normal_short_word_is_not_a_smiley(self):
        """Регресс на слишком широкую регулярку: «ок)» — не смайл."""
        from logic.agent_router import _is_reaction
        self.assertFalse(_is_reaction("хм)"))
        self.assertFalse(_is_reaction("курс?"))

    def test_double_letter_words_survive_normalization(self):
        from logic.agent_router import _normalize_short
        self.assertEqual(_normalize_short("класс"), "класс")
        self.assertEqual(_normalize_short("лоооол"), "лол")


class NobodyTests(RouterTestBase):
    def test_nobody_returns_no_agent(self):
        self._llm(NOBODY)
        agent, research = route("бля как я задолбался", RouterState(), "key")
        self.assertIsNone(agent)
        self.assertFalse(research)

    def test_nobody_does_not_change_sticky(self):
        self._llm(NOBODY)
        state = RouterState()
        state.last_agent_name = "Iris"
        route("просто мысли вслух", state, "key")
        self.assertEqual(state.last_agent_name, "Iris")

    def test_nobody_is_case_insensitive(self):
        self._llm("никто")
        agent, _ = route("что-то бормочу", RouterState(), "key")
        self.assertIsNone(agent)


class RouterMenuTests(unittest.TestCase):
    """Регрессия: правильный ответ обязан быть в меню."""

    def setUp(self):
        self.prompt = agent_router._build_router_prompt()

    def test_cipher_is_offered(self):
        self.assertIn("Cipher", self.prompt)

    def test_cipher_is_marked_expensive(self):
        self.assertIn("дорог", self.prompt.lower())

    def test_nobody_option_is_documented(self):
        self.assertIn(NOBODY, self.prompt)

    def test_all_four_agents_present(self):
        for name in ("Redmond", "Iris", "Newser", "Cipher"):
            self.assertIn(name, self.prompt)

    def test_infra_names_are_disambiguated(self):
        """«Почему гемини упал» — жалоба на сбой, а не новость про компанию.
        Без этой подсказки роутер уводил такое в Newser+research."""
        low = self.prompt.lower()
        for word in ("gemini", "groq", "telegram"):
            self.assertIn(word, low)


class ReplyTargetExtractionTests(unittest.TestCase):
    """Извлечение агента из Telegram-реплая."""

    def setUp(self):
        self.extract = agent_router.reply_target_agent

    def _update(self, username=None, is_bot=True, has_reply=True):
        author = (types.SimpleNamespace(username=username, is_bot=is_bot)
                  if username is not None else None)
        replied = types.SimpleNamespace(from_user=author) if has_reply else None
        return types.SimpleNamespace(message=types.SimpleNamespace(reply_to_message=replied))

    def test_reply_to_cipher_bot(self):
        self.assertEqual(self.extract(self._update("cipher_redberry_bot")), "Cipher")

    def test_reply_to_newser_bot(self):
        self.assertEqual(self.extract(self._update("newser_redmond_bot")), "Newser")

    def test_no_reply_gives_empty(self):
        self.assertEqual(self.extract(self._update(has_reply=False)), "")

    def test_reply_to_human_gives_empty(self):
        self.assertEqual(self.extract(self._update("vlad", is_bot=False)), "")

    def test_reply_to_foreign_bot_gives_empty(self):
        self.assertEqual(self.extract(self._update("some_other_bot")), "")


if __name__ == "__main__":
    unittest.main()
