"""Модели провайдеров: конфиг размышлений Gemini + цепочка роутера.

Регрессии, которые ловим (обе стоили нам живых сбоев 12–13.08.2026):
  • `qwen/qwen3-32b` Groq снёс — 404 model_not_found, вся Groq-цепочка легла;
  • `thinkingBudget: 0` на gemini-3.x → 400, а без конфига модель думает
    и съедает весь maxOutputTokens (ответ приходит пустым).
"""

import json
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
from utils import gemini

DEAD_MODELS = ("qwen/qwen3-32b",)


class ThinkingConfigTests(unittest.TestCase):
    def test_gemini_3_family_uses_thinking_level(self):
        for model in ("gemini-3.6-flash", "gemini-3.5-flash",
                      "gemini-3.1-flash-lite", "gemini-3-flash-preview"):
            self.assertEqual(gemini._thinking_config(model),
                             {"thinkingLevel": "minimal"}, model)

    def test_gemini_25_uses_thinking_budget(self):
        # 2.5 отвечает 400 «Thinking level is not supported for this model»
        self.assertEqual(gemini._thinking_config("gemini-2.5-flash"),
                         {"thinkingBudget": 0})

    def test_unknown_family_falls_back_to_budget(self):
        self.assertEqual(gemini._thinking_config("some-future-model"),
                         {"thinkingBudget": 0})

    def test_empty_model_follows_default_model(self):
        expected = ({"thinkingLevel": "minimal"}
                    if gemini.DEFAULT_MODEL.startswith("gemini-3")
                    else {"thinkingBudget": 0})
        self.assertEqual(gemini._thinking_config(""), expected)


class GenerateBodyTests(unittest.TestCase):
    """thinkingConfig должен доезжать до тела запроса, а не только до хелпера."""

    def setUp(self):
        self.sent = []
        self._orig = gemini._post_generate
        gemini._post_generate = lambda key, body, model, timeout: (
            self.sent.append((model, body)) or {}
        )

    def tearDown(self):
        gemini._post_generate = self._orig

    def test_generate_puts_level_for_3x(self):
        gemini.generate([{"text": "hi"}], model="gemini-3.6-flash", api_key="k")
        _, body = self.sent[0]
        self.assertEqual(body["generationConfig"]["thinkingConfig"],
                         {"thinkingLevel": "minimal"})

    def test_generate_puts_budget_for_25(self):
        gemini.generate([{"text": "hi"}], model="gemini-2.5-flash", api_key="k")
        _, body = self.sent[0]
        self.assertEqual(body["generationConfig"]["thinkingConfig"],
                         {"thinkingBudget": 0})

    def test_generate_contents_matches_generate(self):
        gemini.generate_contents([{"role": "user", "parts": [{"text": "hi"}]}],
                                 model="gemini-3.6-flash", api_key="k")
        _, body = self.sent[0]
        self.assertEqual(body["generationConfig"]["thinkingConfig"],
                         {"thinkingLevel": "minimal"})


class RouterProviderChainTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._gem, self._groq = agent_router._ask_gemini, agent_router._ask_groq

    def tearDown(self):
        agent_router._ask_gemini, agent_router._ask_groq = self._gem, self._groq

    def _patch(self, gemini_reply, groq_reply):
        agent_router._ask_gemini = lambda s, u: (
            self.calls.append("gemini") or gemini_reply)
        agent_router._ask_groq = lambda s, u, k: (
            self.calls.append("groq") or groq_reply)

    def test_gemini_answers_groq_not_touched(self):
        self._patch("Iris", "Newser")
        name, research = agent_router._llm_route("что с целями", [], "key")
        self.assertEqual((name, research), ("Iris", False))
        self.assertEqual(self.calls, ["gemini"])

    def test_groq_picks_up_when_gemini_silent(self):
        self._patch("", "Newser+research")
        name, research = agent_router._llm_route("что нового по крипте", [], "key")
        self.assertEqual((name, research), ("Newser", True))
        self.assertEqual(self.calls, ["gemini", "groq"])

    def test_both_silent_gives_none(self):
        self._patch("", "")
        self.assertEqual(agent_router._llm_route("привет", [], "key"), (None, False))
        self.assertEqual(self.calls, ["gemini", "groq"])

    def test_keyword_fallback_still_routes_when_llm_dead(self):
        self._patch("", "")
        state = agent_router.RouterState()
        agent, _ = agent_router.route("что нового по крипте", state, "key")
        self.assertEqual(agent.name, "Newser")


class GroundingModelChainTests(unittest.TestCase):
    """У google_search своя узкая квота — одна модель ненадёжна, нужен перебор."""

    def setUp(self):
        self.tried = []
        self._orig = gemini.generate

    def tearDown(self):
        gemini.generate = self._orig

    def _patch(self, answers):
        """answers: {model: текст ответа или '' если модель молчит}."""
        def fake(parts, **kw):
            model = kw.get("model", "")
            self.tried.append(model)
            text = answers.get(model, "")
            return {"candidates": [{"content": {"parts": [{"text": text}]}}]} if text else None
        gemini.generate = fake

    def test_first_model_answers_second_not_touched(self):
        self._patch({"": "ответ"})
        result = gemini.grounded_search("погода")
        self.assertIsNotNone(result)
        self.assertEqual(self.tried, [""])

    def test_falls_through_to_backup_model(self):
        self._patch({"gemini-2.5-flash": "ответ с запасной"})
        result = gemini.grounded_search("погода")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "ответ с запасной")
        self.assertEqual(self.tried, list(gemini.GROUNDING_MODELS))

    def test_all_silent_gives_none(self):
        self._patch({})
        self.assertIsNone(gemini.grounded_search("погода"))
        self.assertEqual(self.tried, list(gemini.GROUNDING_MODELS))

    def test_default_model_is_first_in_chain(self):
        # Основная модель должна пробоваться первой: у неё квота здоровее.
        self.assertEqual(gemini.GROUNDING_MODELS[0], "")


class DeadModelGuardTests(unittest.TestCase):
    """Простой стоп-кран: снятая провайдером модель не должна вернуться в конфиг."""

    def test_config_json_has_no_dead_models(self):
        raw = (ROOT / "config" / "config.json").read_text(encoding="utf-8")
        cfg = json.loads(raw)
        for dead in DEAD_MODELS:
            self.assertNotIn(dead, raw, f"{dead} снят Groq — 404 model_not_found")
        self.assertTrue(cfg["gemini_model"].startswith("gemini-"))

    def test_config_defaults_have_no_dead_models(self):
        raw = (ROOT / "config" / "config.py").read_text(encoding="utf-8")
        for dead in DEAD_MODELS:
            self.assertNotIn(dead, raw)


if __name__ == "__main__":
    unittest.main()
