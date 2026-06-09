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


def now_local() -> datetime:
    return datetime.now(OWNER_TZ) if OWNER_TZ else datetime.now()


def current_time() -> str:
    return now_local().strftime("%H:%M")


def current_date() -> str:
    return now_local().strftime("%Y-%m-%d")


def is_night() -> bool:
    hour = now_local().hour
    return hour >= 21 or hour < 6
