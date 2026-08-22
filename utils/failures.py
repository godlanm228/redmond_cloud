"""Единая точка сообщения о сбоях.

**Инвариант И1: ошибка кладёт в лог своё тело, а уровень определяется
последствием для владельца — не тем, в каком файле она случилась.**

Зачем понадобилась отдельная точка. Аудит 21.08.2026 насчитал в хабе
42 места, где исключение гасится вообще без записи в лог, и 15 сбоев,
записанных в `logger.debug` при рабочем уровне INFO. Среди последних —
«Memory search failed» и «Failed to persist memory»: память могла умереть
насовсем, а в логе не осталось бы ни строки. Одновременно `raise_for_status()`
в клиенте Gemini оставлял от ошибки только строку статуса, и тело ответа
Google — где написано, ЧТО именно не так с запросом — выбрасывалось. Из-за
этого восемь 400-х, ронявших основного провайдера Iris, разобрать задним
числом было нечем.

Чинить это построчно бессмысленно: класс воспроизведётся в следующем модуле.
Поэтому один вход, через который сбой обязан проходить целиком.

Как выбирать `consequence` — по тому, что теряет владелец:

    DATA_LOSS  что-то не записалось или потеряно      → ERROR
    DEGRADED   ответ будет хуже, но он будет          → WARNING
    TRANSIENT  само пройдёт, шум провайдера           → INFO

Незнакомое значение поднимается до ERROR: опечатка в вызове не имеет права
тихо понизить уровень сбоя.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("failures")

DATA_LOSS = "data_loss"
DEGRADED = "degraded"
TRANSIENT = "transient"

_LEVELS = {
    DATA_LOSS: logging.ERROR,
    DEGRADED: logging.WARNING,
    TRANSIENT: logging.INFO,
}

# Сколько тела ответа кладём в лог. Хватает, чтобы прочитать message
# провайдера; при этом одна ошибка не раздувает лог на мегабайт.
_BODY_LIMIT = 600


class HttpFailure(Exception):
    """HTTP-ответ с плохим статусом, сохранивший своё тело.

    `requests.raise_for_status()` бросает HTTPError, у которого в тексте
    только строка статуса и URL. Тело — единственное место, где написана
    причина, — теряется. Этот класс носит ответ с собой.
    """

    def __init__(self, response: Any):
        self.response = response
        self.status = getattr(response, "status_code", 0)
        super().__init__(f"HTTP {self.status}")

    def __str__(self) -> str:
        return f"HTTP {self.status}: {_body(self.response)}"


def _body(response: Any) -> str:
    text = ""
    try:
        text = (getattr(response, "text", "") or "").strip()
    except Exception:  # noqa: BLE001 — чтение тела не должно ронять отчёт
        text = ""
    if not text:
        text = str(getattr(response, "reason", "") or "нет тела ответа")
    return text[:_BODY_LIMIT] + ("…" if len(text) > _BODY_LIMIT else "")


def check(response: Any) -> Optional[HttpFailure]:
    """Замена `raise_for_status()`: не бросает, но и не теряет тело.

    None — статус в порядке. Иначе HttpFailure, готовый к `report()`.
    """
    status = getattr(response, "status_code", 0)
    return None if 200 <= int(status) < 300 else HttpFailure(response)


def detail(err: Any) -> str:
    """Человекочитаемая причина: для HttpFailure — со статусом и телом."""
    try:
        if isinstance(err, HttpFailure):
            return str(err)
        text = str(err)
        return text or err.__class__.__name__
    except Exception:  # noqa: BLE001
        return "причина не читается"


def report(where: str, err: Any, *, consequence: str, **context: Any) -> None:
    """Записать сбой. Никогда не бросает — отчёт о сбое не имеет права
    стать вторым сбоем."""
    try:
        level = _LEVELS.get(consequence, logging.ERROR)
        tail = " ".join(f"{k}={v}" for k, v in context.items() if v not in (None, ""))
        logger.log(level, "%s: %s%s", where, detail(err), f" [{tail}]" if tail else "")
    except Exception:  # noqa: BLE001
        try:
            logger.error("%s: сбой, причину записать не удалось", where)
        except Exception:  # noqa: BLE001
            pass
