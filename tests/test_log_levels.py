"""Уровень записи в логе соответствует последствию (И1) — включая чужие логгеры.

Аудит 21.08.2026: из 244 строк уровня ERROR за три месяца **242** — один и тот
же сетевой сбой Telegram, и 180 из них приходятся ровно на 01:00 UTC (окно
обслуживания провайдера). Библиотека переподключается сама, владелец ничего
не теряет — но пока это ERROR, настоящие ошибки в логе не видно: их там
попросту нет, все они живут уровнем ниже.
"""

import logging

from utils.logging_config import TransientNetworkFilter


def _record(name, level, msg, exc=None):
    rec = logging.LogRecord(name, level, __file__, 1, msg, (), None)
    if exc is not None:
        rec.exc_info = (type(exc), exc, None)
    return rec


class _NetworkError(Exception):
    pass


def test_nightly_bad_gateway_is_not_an_error():
    f = TransientNetworkFilter()
    rec = _record("telegram.ext.Updater", logging.ERROR,
                  "Exception happened while polling for updates.",
                  _NetworkError("Bad Gateway"))
    assert f.filter(rec) is True
    assert rec.levelno == logging.INFO
    assert rec.levelname == "INFO"


def test_read_error_and_timeouts_too():
    f = TransientNetworkFilter()
    for text in ("httpx.ReadError: ", "Timed out", "httpx.RemoteProtocolError: "
                 "Server disconnected without sending a response"):
        rec = _record("telegram.ext.Updater", logging.ERROR,
                      "Exception happened while polling for updates.",
                      _NetworkError(text))
        f.filter(rec)
        assert rec.levelno == logging.INFO, text


def test_real_telegram_error_stays_an_error():
    """Понижаем только известную сетевую рябь, а не всё подряд от этого логгера."""
    f = TransientNetworkFilter()
    rec = _record("telegram.ext.Updater", logging.ERROR,
                  "Exception happened while polling for updates.",
                  _NetworkError("Unauthorized: bot token is invalid"))
    f.filter(rec)
    assert rec.levelno == logging.ERROR


def test_other_loggers_are_untouched():
    """«Bad Gateway» от нашего кода — не рябь polling'а, его глушить нельзя."""
    f = TransientNetworkFilter()
    rec = _record("logic.response_generator", logging.ERROR,
                  "запись в долгую память", _NetworkError("Bad Gateway"))
    f.filter(rec)
    assert rec.levelno == logging.ERROR


def test_filter_never_drops_records():
    """Фильтр меняет уровень, но ничего не выбрасывает — иначе шум станет
    невидимым совсем, а он нужен для диагностики."""
    f = TransientNetworkFilter()
    for rec in (_record("telegram.ext.Updater", logging.ERROR, "x",
                        _NetworkError("Bad Gateway")),
                _record("что.угодно", logging.WARNING, "y")):
        assert f.filter(rec) is True


def test_filter_survives_a_broken_exception():
    class Nasty(Exception):
        def __str__(self):
            raise ValueError("сломанный __str__")

    f = TransientNetworkFilter()
    rec = _record("telegram.ext.Updater", logging.ERROR, "x", Nasty())
    assert f.filter(rec) is True
    assert rec.levelno == logging.ERROR
