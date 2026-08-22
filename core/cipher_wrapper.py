"""
Cipher — Claude Code CLI на VM как subprocess (`claude -p`), Pro-подписка Влада.

Особенности:
  • Лимит Pro общий с десктопом Влада — при rate-limit ставим «замок» в kv,
    агенты отвечают «Cipher на КД до HH:MM».
  • Права CLI ограничены ~/.claude/settings.json на VM (read-only анализ).
  • Запуск из ~/redmond-hub — Cipher видит код бота и свои же логи.

Сессии (с 15.08.2026). Раньше КАЖДОЕ сообщение запускало CLI с нуля: он не
помнил ни своего прошлого ответа, ни вопроса Влада. 12.08 это выглядело так —
Cipher попросил разрешение на SSH, Влад ответил «Даю разрешение», и Cipher в
новой сессии уже не знал, о чём речь. Теперь решение принимается кодом:

  реплай на его сообщение  → эта сессия, без вариантов
  меньше 30 минут          → продолжаем последнюю
  30 минут … 24 часа       → продолжаем, но сам Cipher предупреждён: если тема
                             сменилась, он говорит об этом и начинает заново
  больше 24 часов          → новая сессия

Системная приписка (--append-system-prompt) решает вторую проблему 12.08:
проектный гид описывал деплой через ssh с виндовым путём к ключу, и Cipher,
находясь ВНУТРИ VM, честно пытался заssh-иться сам в себя. Плюс формат: он
отвечал markdown-таблицами, которые в Telegram превращаются в стену палок.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)

CIPHER_TIMEOUT_SEC = 420  # 7 минут на задачу — дольше в TG-чате бессмысленно
_LOCK_KEY = "cipher_lock"

# Окна принятия решения о сессии.
CONTINUE_WINDOW_MIN = 30      # свежий разговор — продолжаем молча
MAX_RESUME_HOURS = 24         # дальше только новая сессия

SYSTEM_APPENDIX = """\
Ты работаешь ВНУТРИ VM 203.0.113.10 (Oracle Cloud), пользователь ubuntu, рабочий
каталог ~/redmond-hub. Это боевой сервер телеграм-хаба, а не рабочая станция.

Про доступы:
- Никакого ssh. Ты уже на этой машине — все файлы, логи и systemd доступны локально.
- Логи бота: ~/redmond-hub/logs/v2.log (сервис redmond-hub.service под systemd).
  Справочник по проекту — docs/ARCHITECTURE.md (там же деплой и джобы).
- База: ~/redmond-hub/data/memory.sqlite (SQLite) — ЕДИНСТВЕННЫЙ источник
  живых данных. JSON-архив до миграции 15.08.2026 убран из рабочего дерева
  22.08 в ~/backups/coach-json-frozen-*.tar.gz: он вводил в заблуждение —
  правки в нём ни на что не влияли.
- Секретов в .env не читай, доступ туда закрыт намеренно.

Про формат ответа — это уходит прямо в Telegram:
- Никаких markdown-таблиц: телеграм их не рендерит, получается стена из палок.
  Нужна табличная выкладка — делай короткими строками «поле: значение».
- Коротко, по делу, без заголовков уровня документа и без длинных вступлений.
- Разделители «---» не нужны.

Если данных не хватает — скажи, чего именно, и что для этого нужно. Не выдумывай.
"""


# ---------------------------------------------------------------------------
# Авторизация: живой ли Cipher вообще
# ---------------------------------------------------------------------------

CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"
# За сколько до истечения refresh-токена начинать предупреждать. Access-токен
# обновляется сам, а вот refresh — нет: он просто кончается, и Cipher молча
# умирает до следующего ручного /login.
REFRESH_WARN_DAYS = 5


def auth_status() -> Dict[str, Any]:
    """Состояние авторизации CLI. Читаем ТОЛЬКО метаданные, не токены.

    13–15.08.2026 Cipher был мёртв, и узнали об этом случайно: файла
    credentials не было вообще (не «протух» — исчез). Пользователь замечает
    такое только когда сам обратится к Cipher и получит отказ.
    """
    out: Dict[str, Any] = {"ok": False, "reason": "", "expires_in_days": None}
    if not CREDENTIALS_PATH.exists():
        out["reason"] = "нет файла авторизации — нужен /login на VM"
        return out
    try:
        raw = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        out["reason"] = f"файл авторизации не читается: {e}"
        return out

    refresh_at = _find_key(raw, "refreshtokenexpiresat")
    if refresh_at is None:
        # Формат мог смениться — файл есть, значит логин был. Не паникуем.
        out["ok"] = True
        out["reason"] = "срок неизвестен (формат файла другой)"
        return out
    try:
        ts = float(refresh_at)
        ts = ts / 1000 if ts > 1e12 else ts
        left = (datetime.fromtimestamp(ts) - datetime.now()).total_seconds() / 86400
    except (TypeError, ValueError):
        out["ok"] = True
        out["reason"] = "срок не распарсился"
        return out

    out["expires_in_days"] = round(left, 1)
    if left <= 0:
        out["reason"] = "refresh-токен истёк — нужен /login на VM"
        return out
    out["ok"] = True
    if left <= REFRESH_WARN_DAYS:
        out["reason"] = f"refresh-токен кончится через {left:.1f} дн — нужен /login"
    return out


def _find_key(obj: Any, needle: str) -> Any:
    """Поиск ключа по имени на любой глубине: структура файла может меняться."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == needle:
                return v
            found = _find_key(v, needle)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key(item, needle)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Замок по лимиту Pro
# ---------------------------------------------------------------------------

def _get_lock() -> Optional[str]:
    """«HH:MM» до которого Cipher на КД, или None."""
    until = str((db.kv_get(_LOCK_KEY, {}) or {}).get("locked_until") or "")
    if until and now_local().isoformat() < until:
        return until[11:16]
    return None


def _set_lock(hours: float = 1.0) -> str:
    until = now_local() + timedelta(hours=hours)
    db.kv_set(_LOCK_KEY, {"locked_until": until.isoformat(timespec="minutes")})
    return until.strftime("%H:%M")


def _parse_reset_hours(text: str) -> float:
    """Часы до сброса лимита из сообщения CLI («resets 9:30pm (UTC)»).
    Не распарсилось — дефолт 1ч (перепроверим раньше, чем недождёмся)."""
    m = re.search(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, re.I)
    if not m:
        return 1.0
    from datetime import timezone
    h = int(m.group(1))
    minute = int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm == "pm" and h != 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(hour=h, minute=minute, second=0, microsecond=0)
    if target <= now_utc:
        target += timedelta(days=1)
    return min(max((target - now_utc).total_seconds() / 3600, 0.25), 12.0)


# ---------------------------------------------------------------------------
# Сессии
# ---------------------------------------------------------------------------

def session_for_reply(chat_id: int, message_id: Optional[int]) -> Optional[str]:
    """Сессия, к сообщению которой относится реплай."""
    if not message_id:
        return None
    row = db.query_one(
        "SELECT session_id FROM cipher_sessions WHERE chat_id=? AND message_id=?"
        " ORDER BY id DESC LIMIT 1", (chat_id, message_id))
    return row["session_id"] if row else None


def last_session(chat_id: int) -> Optional[Dict[str, Any]]:
    row = db.query_one(
        "SELECT * FROM cipher_sessions WHERE chat_id=? ORDER BY updated DESC, id DESC"
        " LIMIT 1", (chat_id,))
    return dict(row) if row else None


def _age_minutes(updated: str) -> Optional[float]:
    try:
        return (now_local() - datetime.fromisoformat(updated)).total_seconds() / 60
    except (ValueError, TypeError):
        return None


def decide_session(chat_id: int, reply_to_message_id: Optional[int] = None
                   ) -> Tuple[Optional[str], str]:
    """Какую сессию продолжать. Возвращает (session_id | None, причина).

    None = новая сессия. Причина уходит в лог: по ней видно, почему Cipher
    помнит или не помнит предыдущий разговор.
    """
    replied = session_for_reply(chat_id, reply_to_message_id)
    if replied:
        return replied, "reply"

    last = last_session(chat_id)
    if not last:
        return None, "первая задача"

    age = _age_minutes(last["updated"])
    if age is None:
        return None, "непонятное время прошлой сессии"
    if age <= CONTINUE_WINDOW_MIN:
        return last["session_id"], f"продолжение ({int(age)} мин назад)"
    if age <= MAX_RESUME_HOURS * 60:
        return last["session_id"], f"давняя ({int(age // 60)} ч назад) — решит сам"
    return None, f"прошло больше {MAX_RESUME_HOURS} ч — новая"


def remember_session(chat_id: int, session_id: str, topic: str = "",
                     message_id: Optional[int] = None) -> None:
    if not session_id:
        return
    now = now_local().isoformat(timespec="minutes")
    existing = db.query_one(
        "SELECT id FROM cipher_sessions WHERE chat_id=? AND session_id=?",
        (chat_id, session_id))
    if existing:
        db.execute("UPDATE cipher_sessions SET updated=?, message_id=COALESCE(?, message_id)"
                   " WHERE id=?", (now, message_id, existing["id"]))
    else:
        db.execute(
            "INSERT INTO cipher_sessions(chat_id, session_id, message_id, topic,"
            " created, updated) VALUES(?,?,?,?,?,?)",
            (chat_id, session_id, message_id, topic[:120], now, now))


def bind_message(chat_id: int, session_id: str, message_id: int) -> None:
    """Привязать id отправленного сообщения к сессии — чтобы реплай на него
    нашёл нужный разговор."""
    if not (session_id and message_id):
        return
    db.execute("UPDATE cipher_sessions SET message_id=?, updated=? "
               "WHERE chat_id=? AND session_id=?",
               (message_id, now_local().isoformat(timespec="minutes"),
                chat_id, session_id))


_STALE_NOTE = (
    "[Системная заметка: с прошлой задачи в этом разговоре прошло заметное время. "
    "Если сообщение ниже — продолжение той же темы, продолжай как обычно. "
    "Если тема новая, скажи об этом одной строкой и работай с чистого листа, "
    "не притягивая старый контекст.]\n\n"
)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

async def run_cipher(task: str, chat_id: int = 0,
                     reply_to_message_id: Optional[int] = None) -> Tuple[str, str]:
    """Выполнить задачу через Claude Code CLI.

    Возвращает (текст для чата, session_id). session_id пустой, если сессии нет
    (ошибка, лимит) — тогда привязывать сообщение не к чему.
    """
    task = (task or "").strip()
    if not task:
        return "Пустая задача.", ""

    lock = _get_lock()
    if lock:
        return f"Я на КД до {lock} — лимит Pro общий с десктопом Влада. Позже.", ""

    session_id, reason = decide_session(chat_id, reply_to_message_id)
    prompt = task
    if session_id and reason.startswith("давняя"):
        prompt = _STALE_NOTE + task

    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--append-system-prompt", SYSTEM_APPENDIX]
    if session_id:
        cmd += ["--resume", session_id]

    logger.info("Cipher task (%s): %s", reason, task[:100])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(Path.home() / "redmond-hub"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return "Claude CLI не найден на VM (npm install -g @anthropic-ai/claude-code).", ""

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=CIPHER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("Cipher timeout (%ds)", CIPHER_TIMEOUT_SEC)
        return "Задача не уложилась в 7 минут — обрубил. Сузь запрос.", ""

    raw = (out or b"").decode(errors="replace").strip()
    err_text = (err or b"").decode(errors="replace").strip()
    text, new_session, is_error = _parse_output(raw)

    if text and not is_error and proc.returncode == 0:
        if new_session:
            remember_session(chat_id, new_session, topic=task)
        return text, new_session

    combined = text or raw or err_text
    logger.warning("Cipher failed (rc=%s): %s", proc.returncode, combined[:300])

    if re.search(r"rate.?limit|usage limit|session limit|hit your .*limit|limit.{0,3}resets?",
                 combined, re.I):
        until = _set_lock(_parse_reset_hours(combined))
        return f"Упёрся в лимит Pro (общий с десктопом Влада) — на КД до {until}.", ""
    if re.search(r"log ?in|login|authenticat|credentials|setup-token", combined, re.I):
        return ("Я не авторизован на VM. Влад: `ssh` на сервер → команда `claude` → "
                "`/login` под Pro-аккаунтом (один раз).", "")
    # Сессия могла протухнуть на стороне CLI — пробуем один раз с чистого листа,
    # иначе Cipher залипнет на мёртвом id до истечения суток.
    if session_id and re.search(r"session|resume|not found|no conversation", combined, re.I):
        logger.info("Cipher: сессия %s не подхватилась — начинаю новую", session_id[:8])
        _forget_session(chat_id, session_id)
        return await run_cipher(task, chat_id=chat_id)
    return f"Упал: {combined[:400] or 'пустой ответ от CLI'}", ""


def _forget_session(chat_id: int, session_id: str) -> None:
    db.execute("DELETE FROM cipher_sessions WHERE chat_id=? AND session_id=?",
               (chat_id, session_id))


def _parse_output(raw: str) -> Tuple[str, str, bool]:
    """JSON-вывод CLI → (текст, session_id, признак ошибки).

    Формат может смениться, а Cipher должен продолжать работать: если это не
    наш JSON — отдаём сырой текст, просто без session_id.
    """
    if not raw:
        return "", "", True
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return raw, "", False
    if not isinstance(data, dict):
        return raw, "", False
    text = str(data.get("result") or "").strip()
    return text, str(data.get("session_id") or ""), bool(data.get("is_error"))
