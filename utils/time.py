"""
Время владельца — единая точка. VM живёт в UTC, Влад — в Europe/Berlin:
без этого дневник/цели датируются «вчера» (ловили на записи 2026-06-09
при локальных 01:37 десятого). Все timestamp'ы пользовательских данных —
через now_local().
"""

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    OWNER_TZ = ZoneInfo("Europe/Berlin")
except Exception:  # pragma: no cover — zoneinfo есть с 3.9, страховка
    OWNER_TZ = None


_clock = None  # подмена часов для сценарных прогонов (evals/); в бою None


def now_local() -> datetime:
    if _clock is not None:
        return _clock()
    return datetime.now(OWNER_TZ) if OWNER_TZ else datetime.now()


def set_clock(clock) -> None:
    """Заменить «сейчас» функцией без аргументов (None — вернуть настоящие часы).

    Нужно, чтобы прогнать реальный диалог в том времени, когда он шёл: пинг
    «обед» в 20:30, «проснулся в 12» и прочее зависят от часов. Модули берут
    now_local по имени, поэтому подмена сделана внутри функции, а не заменой
    самой функции."""
    global _clock
    _clock = clock


