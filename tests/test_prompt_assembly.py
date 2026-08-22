"""Сборка сообщения для модели: что обязано в нём быть.

Зачем отдельный файл. 22.08.2026 в рабочем дереве оказалась подмена
`for turn in ctx.history[-4:]` → `for turn in []`, то есть история диалога
не уходила в промпт вообще. Полный набор из 371 теста прошёл зелёным:
единственный тест, который трогал `_build_user_message`, помечен xfail
(он про вытеснение реплик пингами, К4) — и его падение выглядело ожидаемым.

Здесь — ЗЕЛЁНЫЕ сторожа на то, что сейчас работает. Они не про будущие
улучшения контекста, а про то, что уже собранные части не должны молча
исчезнуть из промпта.
"""

import threading

from logic.intent_recognizer import Intent
from logic.response_generator import GenerationContext, ResponseGenerator


def _rg():
    rg = object.__new__(ResponseGenerator)
    rg.mem = None
    rg.top_k = 3
    rg.max_history = 6
    rg.history_by_chat = {}
    rg._history_guard = threading.RLock()
    rg._history_loaded = set()
    return rg


def _ctx(**kw):
    base = dict(intent=Intent(name="chat", slots={}), user_text="что у меня сегодня?")
    base.update(kw)
    return GenerationContext(**base)


HISTORY = [
    {"user": "у меня семестрфериен, учебы нету", "bot": "Записала: каникулы"},
    {"user": "поел гречку с курицей", "bot": "Записала [питание]"},
    {"user": "вечером в бильярд", "bot": "Хорошей игры"},
    {"user": "креатин пью с молоком", "bot": "Отметила"},
]


def test_history_actually_reaches_the_prompt():
    """Сторож против «истории больше нет». Падает, если блок исчез целиком."""
    rendered = _rg()._build_user_message(_ctx(history=HISTORY))
    assert "[Предыдущий диалог]" in rendered, "блок истории пропал из промпта"
    missing = [h["user"] for h in HISTORY if h["user"] not in rendered]
    assert not missing, f"реплики владельца не доехали до промпта: {missing}"


def test_bot_replies_reach_the_prompt_too():
    """Без ответов бота «продолжи» и «что записала?» теряют смысл."""
    rendered = _rg()._build_user_message(_ctx(history=HISTORY))
    missing = [h["bot"] for h in HISTORY if h["bot"] not in rendered]
    assert not missing, f"ответы бота не доехали до промпта: {missing}"


def test_no_history_means_no_empty_header():
    """Пустая история не должна оставлять висящий заголовок."""
    rendered = _rg()._build_user_message(_ctx(history=[]))
    assert "[Предыдущий диалог]" not in rendered


def test_memory_hits_reach_the_prompt():
    docs = ["что я говорил про креатин? => пьёшь с молоком с 17.08"]
    rendered = _rg()._build_user_message(_ctx(retrieved_docs=docs))
    assert "[Релевантное из памяти]" in rendered
    assert "креатин" in rendered


def test_search_results_reach_the_prompt():
    rendered = _rg()._build_user_message(_ctx(
        search_results=[{"title": "Заголовок новости", "snippet": "выжимка"}],
        search_source="duckduckgo",
    ))
    assert "duckduckgo" in rendered
    assert "Заголовок новости" in rendered


def test_user_question_is_always_last():
    """Вопрос владельца — последняя строка: модель отвечает на него,
    а не на хвост контекста."""
    rendered = _rg()._build_user_message(_ctx(
        history=HISTORY,
        retrieved_docs=["старый диалог => старый ответ"],
        user_text="так что с планом на вечер?",
    ))
    assert rendered.strip().endswith("так что с планом на вечер?")
