"""Аварийные сообщения владельцу: доходят, но не долбят.

Инвариант И7: у каждого сигнала о поломке есть получатель.

До 28.09.2026 сигнал был, а получателя не было. Cipher потерял авторизацию,
и каждое утро в лог ложился ERROR «нужен /login на VM», а сразу за ним «полная
тишина, сообщение не отправляю»: полный mute глушил и аварии. Groq снял
резервную модель, healthcheck писал ERROR при каждом старте. Лог никто не
читает, так что две недели об этом не знал никто.

Правило:
  • без тишины авария приходит, но одна и та же не чаще раза в сутки;
  • при полной тишине тоже приходит (тишина про «как дела», а не про
    сломанный инструмент), но одна и та же не чаще раза в `MUTED_REPEAT_DAYS`.
«Одна и та же» — тот же `kind` и тот же текст: изменившаяся авария (например,
Cipher был «скоро истечёт», стал «истёк») приходит сразу.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)

REPEAT_DAYS = 1
MUTED_REPEAT_DAYS = 3


def _key(kind: str) -> str:
    return f"alert:{kind}"


def should_send(kind: str, text: str, muted: bool) -> bool:
    """Отправлять ли аварию сейчас. Никогда не бросает."""
    try:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        last: Dict[str, Any] = db.kv_get(_key(kind), {}) or {}
        if last.get("digest") != digest:
            return True
        sent = datetime.fromisoformat(last["sent"])
        gap = timedelta(days=MUTED_REPEAT_DAYS if muted else REPEAT_DAYS)
        # Минус час: ежедневная джоба с небольшим сдвигом не должна пропускать сутки.
        return now_local() - sent >= gap - timedelta(hours=1)
    except Exception:  # noqa: BLE001 — сомнение трактуем в пользу отправки
        logger.warning("alerts: не удалось прочитать историю «%s» — шлём", kind, exc_info=True)
        return True


def mark_sent(kind: str, text: str) -> None:
    try:
        db.kv_set(_key(kind), {
            "digest": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
            "sent": now_local().isoformat(timespec="minutes"),
        })
    except Exception:  # noqa: BLE001
        logger.warning("alerts: не удалось запомнить отправку «%s»", kind, exc_info=True)
