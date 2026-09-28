"""Цикл инструментов против реального контракта провайдеров.

Почему этот файл появился. С 15.08 по 28.09.2026 Iris отвечала заглушкой
«Готово: записала в дневник.» на 14 из 34 сообщений владельца. Причина: модели
Gemini 3.x кладут рядом с functionCall непрозрачную `thoughtSignature` и требуют
вернуть её на следующем шаге, а код пересобирал ход модели из имени и
аргументов. Второй шаг получал 400, а «страховка» выдавала провал за успех.
Весь набор из 459 тестов был зелёным: цикл Gemini не проверял ни один тест,
а выдуманные ответы провайдера подписи не содержали.

Здесь провайдер подделан так, как ведёт себя настоящий API: форма ответа снята
с живого gemini-3.6-flash на VM 28.09.2026, и ход модели без подписи получает
отказ ровно как в бою.
"""

import copy
import threading
from types import SimpleNamespace

from logic import response_generator as rgm
from logic.intent_recognizer import Intent
from utils import gemini

DIARY_RESULT = "Запись #107 в дневник: «Поел» [питание]."

# Снято с живого API: одна часть, в ней functionCall и thoughtSignature рядом.
HOP1 = {"candidates": [{"content": {"role": "model", "parts": [{
    "functionCall": {"name": "add_diary_entry", "args": {"text": "Поел", "tags": ["питание"]}},
    "thoughtSignature": "CiQBjz1rX2NvbnRleHQtc2lnbmF0dXJlLWZyb20tdGhlLW1vZGVs",
}]}}]}
EMPTY = {"candidates": [{"content": {"role": "model", "parts": []}}]}


def _text(t):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": t}]}}]}


class FakeGemini:
    """generate_contents, который ведёт себя как живой API.

    Ход модели с functionCall без thoughtSignature → None: так клиент
    возвращает 400 «Function call is missing a thought_signature».
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.tools_seen = []

    def __call__(self, contents, **kw):
        self.sent.append(copy.deepcopy(contents))
        self.tools_seen.append(kw.get("tools"))
        for c in contents:
            if c.get("role") != "model":
                continue
            for p in c.get("parts", []):
                if "functionCall" in p and "thoughtSignature" not in p:
                    return None
        return self.replies.pop(0) if self.replies else None


IRIS = SimpleNamespace(name="Iris", provider_order=["gemini"], allowed_tools=None,
                       temperature=0.5, max_tokens=800, emoji="🎯")


def _rg(monkeypatch, fake, compose=""):
    rg = object.__new__(rgm.ResponseGenerator)
    rg.config = SimpleNamespace(
        gemini_api_key="k", gemini_model="gemini-3.6-flash",
        groq_api_key="", groq_model="", groq_fallback_model="",
        llm_provider_order=["gemini"],
    )
    rg.mem = None
    rg.top_k = 3
    rg.max_history = 6
    rg.history_by_chat = {}
    rg._history_guard = threading.RLock()
    rg._history_loaded = set()
    rg._build_system_prompt = lambda ctx: "system"
    rg._build_user_message = lambda ctx: ctx.user_text
    saved = []
    rg._save_interaction = lambda user, resp, chat_id=0, *a, **kw: saved.append((user, resp))
    calls = []

    def fake_tool(name, args, rg=None, session=None):
        calls.append((name, args))
        return DIARY_RESULT

    monkeypatch.setattr(gemini, "generate_contents", fake)
    monkeypatch.setattr(gemini, "generate_text", lambda prompt, **kw: compose)
    monkeypatch.setattr("logic.tools.execute_tool", fake_tool)
    return rg, saved, calls


def _ask(rg, text="Поел", agent=IRIS):
    return rg.generate(Intent(name="chat", slots={}), text, "owner", agent, 1)


def test_model_turn_keeps_the_signature():
    turn = gemini.model_turn(HOP1)
    assert turn["role"] == "model"
    assert turn["parts"][0]["thoughtSignature"], "подпись размышления потеряна"


def test_second_step_carries_the_signature(monkeypatch):
    fake = FakeGemini([HOP1, _text("Приятного!")])
    rg, _saved, _calls = _rg(monkeypatch, fake)
    reply = _ask(rg)

    model_turns = [c for c in fake.sent[1] if c.get("role") == "model"]
    assert model_turns and "thoughtSignature" in model_turns[0]["parts"][0], \
        "второй шаг ушёл без подписи — живой API ответит 400"
    assert reply.startswith("Приятного!")
    assert reply.failed is False


def test_receipt_comes_from_the_tool_result_not_from_the_model(monkeypatch):
    rg, _saved, _calls = _rg(monkeypatch, FakeGemini([HOP1, _text("Приятного!")]))
    reply = _ask(rg)
    assert "📝 Запись #107 в дневник: «Поел» [питание]." in reply


def test_silence_after_an_action_is_composed_not_stubbed(monkeypatch):
    """Модель промолчала после записи → отдельный вызов составляет ответ."""
    rg, _saved, calls = _rg(monkeypatch, FakeGemini([HOP1, EMPTY]), compose="Записала, приятного.")
    reply = _ask(rg)
    assert "Готово" not in reply
    assert reply.startswith("Записала, приятного.")
    assert len(calls) == 1, "запись выполнена дважды"


def test_all_models_silent_is_an_honest_failure(monkeypatch):
    """Никто не ответил: честный отказ + что реально сделано, и никакого «Готово»."""
    rg, saved, _calls = _rg(monkeypatch, FakeGemini([HOP1, EMPTY]), compose="")
    reply = _ask(rg)
    assert "Готово" not in reply
    assert "Модели не ответили" in reply
    assert "📝 Запись #107 в дневник: «Поел»" in reply
    assert reply.failed is True
    assert saved == [], "отказ записан в историю как реплика разговора"


def test_provider_outage_mid_loop_does_not_repeat_the_action(monkeypatch):
    """Gemini лёг на втором шаге: запись не повторяется, ответ составляется."""
    rg, _saved, calls = _rg(monkeypatch, FakeGemini([HOP1]), compose="Записала.")
    reply = _ask(rg)
    assert reply.startswith("Записала.")
    assert len(calls) == 1


def test_groq_silence_after_an_action_is_composed_not_stubbed(monkeypatch):
    """Тот же класс на пути Groq: пустой ответ после инструмента."""
    tool_call = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "add_diary_entry", "arguments": '{"text": "Поел"}'}}]}}]}
    silent = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    composed = {"choices": [{"finish_reason": "stop", "message": {"content": "Записала."}}]}
    seq = [tool_call, silent, composed]

    rg, _saved, calls = _rg(monkeypatch, FakeGemini([]), compose="")
    rg.config.groq_api_key = "k"
    rg.config.groq_model = "openai/gpt-oss-120b"
    rg._groq_chat = lambda *a, **kw: (seq.pop(0), "")
    redmond = SimpleNamespace(name="Redmond", provider_order=["groq"], allowed_tools=None,
                              temperature=0.5, max_tokens=800, emoji="🦞")
    reply = _ask(rg, agent=redmond)
    assert "Готово" not in reply
    assert reply.startswith("Записала.")
    assert "📝 Запись #107" in reply
    assert len(calls) == 1


def test_distress_gets_a_question_even_when_models_are_down(monkeypatch):
    """04.09.2026: сообщение в острой ситуации получило заглушку. Теперь вопрос."""
    from logic import distress
    fake = FakeGemini([])
    rg, saved, calls = _rg(monkeypatch, fake, compose="")
    reply = _ask(rg, text="Суисайд")
    assert reply == distress.FALLBACK_REPLY
    assert calls == [], "в острой ситуации инструменты не вызываются"
    assert all(t is None for t in fake.tools_seen), "модели предложили инструменты"
