"""Инвариант И1: ошибка кладёт своё тело, уровень — по последствию.

Аудит 21.08.2026 нашёл 42 места, где исключение гасится без единой записи
в лог, и 15 сбоев, записанных в DEBUG при рабочем уровне INFO. Среди
последних — «Memory search failed» и «Failed to persist memory»: память
могла умереть насовсем, и в логе не было бы ни строки.

Отдельно: `raise_for_status()` в utils/gemini.py оставляет от ошибки только
строку статуса. Тело ответа Google, где написано ЧТО не так с запросом,
выбрасывалось — поэтому восемь 400-х, ронявших основного провайдера Iris,
остались неразобранными.

Правило уровня — по последствию для владельца, а не по месту в коде:
  data_loss — что-то не записалось или потеряно     → ERROR
  degraded  — ответ будет хуже, но он будет         → WARNING
  transient — само пройдёт, шум провайдера          → INFO
"""

import logging

import pytest

from utils import failures


class _Resp:
    """Минимальный двойник requests.Response."""

    def __init__(self, status, text):
        self.status_code = status
        self.text = text
        self.reason = "Bad Request"
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent"


def test_level_follows_consequence(caplog):
    with caplog.at_level(logging.INFO):
        failures.report("память", RuntimeError("boom"), consequence=failures.DATA_LOSS)
        failures.report("поиск", RuntimeError("boom"), consequence=failures.DEGRADED)
        failures.report("сеть", RuntimeError("boom"), consequence=failures.TRANSIENT)
    levels = [r.levelno for r in caplog.records]
    assert levels == [logging.ERROR, logging.WARNING, logging.INFO]


def test_unknown_consequence_is_not_silently_downgraded(caplog):
    """Опечатка в consequence не должна тихо понизить уровень сбоя."""
    with caplog.at_level(logging.INFO):
        failures.report("что-то", RuntimeError("boom"), consequence="опечатка")
    assert caplog.records[0].levelno == logging.ERROR


def test_body_of_http_error_is_kept():
    """Тело ответа провайдера обязано доезжать до лога."""
    body = '{"error":{"code":400,"message":"thinkingLevel is not supported"}}'
    err = failures.HttpFailure(_Resp(400, body))
    assert "thinkingLevel is not supported" in failures.detail(err)
    assert "400" in failures.detail(err)


def test_report_writes_the_body_not_just_the_status(caplog):
    body = '{"error":{"message":"API key expired"}}'
    with caplog.at_level(logging.WARNING):
        failures.report("Gemini", failures.HttpFailure(_Resp(400, body)),
                        consequence=failures.DEGRADED)
    text = caplog.text
    assert "API key expired" in text, "тело ответа потеряно — сбой недиагностируем"


def test_long_body_is_trimmed_but_keeps_the_start(caplog):
    body = "x" * 5000 + "ХВОСТ"
    with caplog.at_level(logging.WARNING):
        failures.report("Gemini", failures.HttpFailure(_Resp(500, body)),
                        consequence=failures.DEGRADED)
    assert len(caplog.text) < 3000, "лог не должен раздуваться на гигантском теле"
    assert "xxx" in caplog.text


def test_context_is_included(caplog):
    with caplog.at_level(logging.WARNING):
        failures.report("Groq", RuntimeError("rate limit"),
                        consequence=failures.DEGRADED, model="qwen", hop=1)
    assert "model=qwen" in caplog.text
    assert "hop=1" in caplog.text


def test_report_never_raises():
    """Сообщение о сбое не имеет права стать вторым сбоем."""
    class Nasty:
        def __str__(self):
            raise ValueError("сломанный __str__")

    failures.report("что-то", Nasty(), consequence=failures.DEGRADED)


def test_check_returns_body_bearing_failure_for_bad_status():
    ok = _Resp(200, "{}")
    bad = _Resp(429, '{"error":{"message":"Too Many Requests"}}')
    assert failures.check(ok) is None
    fail = failures.check(bad)
    assert isinstance(fail, failures.HttpFailure)
    assert "Too Many Requests" in failures.detail(fail)


# --------------------------------------------------------------------------
# Инструменты: в лог обязан попадать ИСХОД, а исключение — оставаться
# на месте, а не превращаться в «Provider failed» уровнем выше.
# --------------------------------------------------------------------------

def test_tool_result_is_logged_not_just_the_call(caplog):
    from logic import tools
    with caplog.at_level(logging.INFO):
        tools.execute_tool("get_current_time", {}, None)
    assert "Tool: get_current_time" in caplog.text, "вызов не записан"
    assert "Tool result: get_current_time" in caplog.text, (
        "исход инструмента не записан — по логу не понять, что произошло"
    )


def test_tool_exception_stays_at_the_tool(caplog, monkeypatch):
    from logic import tools

    def boom(name, args, rg=None):
        raise ValueError("invalid literal for int() with base 10: 'три'")

    monkeypatch.setattr(tools, "_dispatch_tool", boom)
    with caplog.at_level(logging.WARNING):
        out = tools.execute_tool("web_search", {"query": "новости", "top_k": "три"}, None)

    assert "web_search" in caplog.text
    assert "три" in caplog.text, "причина сбоя не доехала до лога"
    assert "Provider" not in caplog.text, (
        "сбой инструмента не должен выглядеть отказом провайдера"
    )
    assert "не отработал" in out, "модель должна получить честную ошибку, а не пустоту"
