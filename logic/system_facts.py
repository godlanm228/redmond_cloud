"""Факты о состоянии системы для промпта агентов.

Зачем. 10.09.2026 владелец спросил Iris, почему вместо неё ответил Cipher.
Iris ответила, что «фоновый фоллбэк-блок выплюнул системный лог» и что «у
основной нейросети отвалился ключ». Ни то, ни другое не было правдой: Cipher
ответил, потому что роутер отправил ему сообщение, а сам Cipher лежал без
подписки. 26.08 на вопрос про пинг в 20:30 было «у бота сбился таймер».

Iris, Redmond и Newser — это вызовы модели. К серверу, логам и коду у них
доступа нет, и когда их спрашивают «почему», им нечем ответить, кроме
правдоподобной выдумки. Этот блок даёт им то, что код знает наверняка, и
правило: о технических причинах — только отсюда, иначе «не знаю».
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Проверка авторизации Cipher читает файл на диске — дёшево, но не на каждый
# промпт: кэшируем на несколько минут.
_CIPHER_TTL_SEC = 300
_cipher_cache: Optional[Tuple[float, str]] = None

RULES = (
    "SYSTEM RULES:\n"
    "- You (Redmond, Iris, Newser) are model calls. You have NO access to the "
    "server, logs or code. Cipher runs on the server and is the only one who can "
    "check logs.\n"
    "- Explain technical causes (why someone answered, why something failed, how "
    "the hub works) ONLY from SYSTEM FACTS below. If they do not contain the "
    "answer, say you don't know and that Cipher can check the logs. Never invent "
    "technical explanations.\n"
    "- A receipt of the actions you performed (diary entries, deadlines, mute…) "
    "is appended under your reply by code. Do not list recorded items again, and "
    "never claim an action that is not in your tool results."
)


def _cipher_line() -> str:
    global _cipher_cache
    now = time.time()
    if _cipher_cache and now - _cipher_cache[0] < _CIPHER_TTL_SEC:
        return _cipher_cache[1]
    try:
        from utils.model_healthcheck import GONE, check_cipher_auth
        status, detail = check_cipher_auth()
        if status == GONE:
            line = f"Cipher: не работает — {detail}"
        elif detail:
            line = f"Cipher: работает, но {detail}"
        else:
            line = "Cipher: работает"
    except Exception as e:  # noqa: BLE001 — блок фактов не имеет права ронять промпт
        line = f"Cipher: статус неизвестен ({e.__class__.__name__})"
    _cipher_cache = (now, line)
    return line


def _mute_line() -> str:
    try:
        from logic.coach_storage import mute_info
        info = mute_info()
    except Exception as e:  # noqa: BLE001
        return f"Тишина: статус неизвестен ({e.__class__.__name__})"
    if not info:
        return "Тишина: не включена"
    scope = "полная (и дайджесты)" if info.get("scope") == "all" else "только пинги"
    until = info.get("until")
    when = "без срока, пока владелец не скажет «пиши»" if until == "forever" else f"до {until}"
    return f"Тишина: {scope}, {when}"


def _failures_lines() -> List[str]:
    try:
        from utils import failures
        items = failures.recent(hours=24, limit=3)
    except Exception:  # noqa: BLE001
        return []
    return [f"  {datetime.fromtimestamp(ts).strftime('%d.%m %H:%M')} {where}: {text}"
            for ts, where, text in items]


def block() -> str:
    """Готовый блок для конца system prompt: правила + факты."""
    lines = [RULES, "", "SYSTEM FACTS (from code, now):", f"- {_cipher_line()}",
             f"- {_mute_line()}"]
    fails = _failures_lines()
    if fails:
        lines.append("- Сбои за сутки (последние):")
        lines.extend(fails)
    else:
        lines.append("- Сбоев моделей за сутки в этом процессе не было")
    return "\n".join(lines)
