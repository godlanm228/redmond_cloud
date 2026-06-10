"""
Cipher — Claude Code CLI на VM как subprocess (`claude -p`), Pro-подписка Влада.

Особенности:
  • Лимит Pro общий с десктопом Влада (~45 msg / 5h) — при rate-limit ставим
    «замок» в data/cipher_state.json, агенты отвечают «Cipher на КД до HH:MM».
  • Права CLI ограничены ~/.claude/settings.json на VM (read-only анализ:
    Read/Grep/Glob + git log/diff/status + journalctl; .env в deny).
  • Запуск из ~/redmond-hub — Cipher видит код бота и может анализировать
    свои же логи/архитектуру.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Optional

from utils.time import now_local

logger = logging.getLogger(__name__)

CIPHER_TIMEOUT_SEC = 420  # 7 минут на задачу — дольше в TG-чате бессмысленно
_STATE_FILE = Path("data/cipher_state.json")


def _get_lock() -> Optional[str]:
    """«HH:MM» до которого Cipher на КД, или None."""
    try:
        state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        until = state.get("locked_until", "")
        if until and now_local().isoformat() < until:
            return until[11:16]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return None


def _set_lock(hours: float = 1.0) -> str:
    until = now_local() + timedelta(hours=hours)
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps({"locked_until": until.isoformat(timespec="minutes")}),
        encoding="utf-8",
    )
    return until.strftime("%H:%M")


async def run_cipher(task: str) -> str:
    """Выполнить задачу через Claude Code CLI. Возвращает текст для чата."""
    task = (task or "").strip()
    if not task:
        return "Пустая задача."

    lock = _get_lock()
    if lock:
        return f"Я на КД до {lock} — лимит Pro общий с десктопом Влада. Позже."

    logger.info("Cipher task: %s", task[:120])
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", task, "--output-format", "text",
            cwd=str(Path.home() / "redmond-hub"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return "Claude CLI не найден на VM (npm install -g @anthropic-ai/claude-code)."

    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=CIPHER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("Cipher timeout (%ds)", CIPHER_TIMEOUT_SEC)
        return "Задача не уложилась в 7 минут — обрубил. Сузь запрос."

    text = (out or b"").decode(errors="replace").strip()
    err_text = (err or b"").decode(errors="replace").strip()

    if proc.returncode == 0 and text:
        return text

    combined = text or err_text
    logger.warning("Cipher failed (rc=%s): %s", proc.returncode, combined[:300])

    if re.search(r"rate.?limit|usage limit|limit (will )?reset", combined, re.I):
        until = _set_lock(1.0)
        return f"Упёрся в лимит Pro — на КД до {until}."
    if re.search(r"log ?in|login|authenticat|credentials|setup-token", combined, re.I):
        return (
            "Я не авторизован на VM. Влад: `ssh` на сервер → команда `claude` → "
            "`/login` под Pro-аккаунтом (один раз)."
        )
    return f"Упал: {combined[:400] or 'пустой ответ от CLI'}"
