"""Разовая миграция data/coach/*.json → таблицы SQLite.

Правила:
  • идемпотентна — повторный запуск не плодит дублей и не портит уже
    перенесённое (для строк с id используется INSERT OR REPLACE);
  • ничего не удаляет — JSON остаётся замороженной копией на случай отката;
  • сверяет количество перенесённых строк с исходником и возвращает отчёт,
    чтобы «перенеслось» не приходилось принимать на веру.

Запуск:  python -m utils.migrate_json_to_db [--dry-run]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import db

logger = logging.getLogger(__name__)

COACH_DIR = Path("data/coach")

# Мелкое состояние переезжает в kv как есть — структура у каждого своя,
# а транзакционность нужна одинаково.
KV_FILES = {
    "day_state.json": "day_state",
    "presence.json": "presence",
    "mute.json": "mute",
    "ping_style.json": "ping_style",
    "radar.json": "radar",
    "gemini_usage.json": "gemini_usage",
    "week_plan.json": "week_plan",
}


def _read(name: str, default: Any) -> Any:
    p = COACH_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"{name} не читается: {e}") from e


def _migrate_goals(conn) -> Tuple[int, int]:
    rows = _read("goals.json", [])
    for g in rows:
        conn.execute(
            "INSERT OR REPLACE INTO goals"
            "(id, title, why, target_date, status, created, closed, progress_log)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (g.get("id"), g.get("title", ""), g.get("why", ""), g.get("target_date"),
             g.get("status", "active"), g.get("created", ""), g.get("closed"),
             json.dumps(g.get("progress_log") or [], ensure_ascii=False)),
        )
    return len(rows), conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]


def _migrate_deadlines(conn) -> Tuple[int, int]:
    rows = _read("deadlines.json", [])
    for d in rows:
        conn.execute(
            "INSERT OR REPLACE INTO deadlines"
            "(id, title, due, importance, status, created, closed, note)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (d.get("id"), d.get("title", ""), d.get("due", ""),
             d.get("importance", "medium"), d.get("status", "pending"),
             d.get("created", ""), d.get("closed"), d.get("note")),
        )
    return len(rows), conn.execute("SELECT COUNT(*) FROM deadlines").fetchone()[0]


def _migrate_diary(conn) -> Tuple[int, int]:
    rows = _read("diary.json", [])
    for e in rows:
        payload = e.get("data")
        conn.execute(
            "INSERT OR REPLACE INTO diary(id, ts, text, tags, data) VALUES(?,?,?,?,?)",
            (e.get("id"), e.get("timestamp", ""), e.get("text", ""),
             json.dumps(e.get("tags") or [], ensure_ascii=False),
             json.dumps(payload, ensure_ascii=False) if payload else None),
        )
    return len(rows), conn.execute("SELECT COUNT(*) FROM diary").fetchone()[0]


def _migrate_pantry(conn) -> Tuple[int, int]:
    data = _read("pantry.json", {})
    items = data.get("items") or []
    updated = data.get("updated", "")
    for it in items:
        conn.execute("INSERT OR REPLACE INTO pantry(item, added) VALUES(?,?)",
                     (str(it), updated))
    return len(items), conn.execute("SELECT COUNT(*) FROM pantry").fetchone()[0]


def _migrate_shifts(conn) -> Tuple[int, int]:
    data = _read("shifts.json", {})
    for date, s in data.items():
        if not isinstance(s, dict):
            continue
        conn.execute(
            "INSERT OR REPLACE INTO shifts"
            "(date, start, end, status, source, confidence, updated,"
            " last_confirmed_at, note) VALUES(?,?,?,?,?,?,?,?,?)",
            (date, s.get("start"), s.get("end"), s.get("status", "planned"),
             s.get("source", "unknown"), s.get("confidence", "medium"),
             s.get("updated", ""), s.get("last_confirmed_at"), s.get("note")),
        )
        # Стартовое событие: до миграции истории не было, фиксируем то, что есть,
        # чтобы журнал не начинался с пустоты.
        conn.execute(
            "INSERT INTO shift_events(ts, date, action, source, payload, reason)"
            " SELECT ?,?,?,?,?,? WHERE NOT EXISTS"
            " (SELECT 1 FROM shift_events WHERE date=? AND action='import')",
            (s.get("updated", ""), date, "import", s.get("source", "unknown"),
             json.dumps(s, ensure_ascii=False), "перенос из shifts.json", date),
        )
    return len(data), conn.execute("SELECT COUNT(*) FROM shifts").fetchone()[0]


def _migrate_kv(conn) -> Tuple[int, int]:
    from utils.time import now_local
    now = now_local().isoformat(timespec="minutes")
    moved = 0
    for fname, key in KV_FILES.items():
        if not (COACH_DIR / fname).exists():
            continue
        value = _read(fname, None)
        if value is None:
            continue
        conn.execute(
            "INSERT INTO kv(key, value, updated) VALUES(?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value,"
            " updated=excluded.updated",
            (key, json.dumps(value, ensure_ascii=False), now),
        )
        moved += 1
    return moved, conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]


MIGRATIONS = [
    ("goals", _migrate_goals),
    ("deadlines", _migrate_deadlines),
    ("diary", _migrate_diary),
    ("pantry", _migrate_pantry),
    ("shifts", _migrate_shifts),
    ("kv", _migrate_kv),
]


def run(dry_run: bool = False) -> List[Dict[str, Any]]:
    """Переносит всё и возвращает отчёт [{таблица, из json, в базе, ок}]."""
    conn = db.connect()
    report: List[Dict[str, Any]] = []
    try:
        conn.execute("BEGIN")
        for name, fn in MIGRATIONS:
            src, dst = fn(conn)
            report.append({"table": name, "json": src, "db": dst, "ok": dst >= src})
        if dry_run:
            conn.execute("ROLLBACK")
        else:
            conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return report


def needed() -> bool:
    """Есть ли что переносить: JSON на диске лежит, а таблицы пустые.

    Проверяем именно пустоту, а не факт наличия файлов: миграция идемпотентна,
    но запускать её поверх живых данных при каждом старте незачем — да и любая
    будущая ошибка в ней не должна получать доступ к рабочим таблицам.
    """
    if not COACH_DIR.exists():
        return False
    if not any(COACH_DIR.glob("*.json")):
        return False
    conn = db.connect()
    for table in ("goals", "deadlines", "diary", "shifts", "pantry", "kv"):
        if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
            return False
    return True


def run_if_needed() -> Optional[List[Dict[str, Any]]]:
    """Разовый автоперенос на старте. None — переносить нечего.
    Ошибку не проглатываем: подняться на пустой базе поверх существующих
    JSON — это молча начать с чистого листа, чего нам как раз и не надо."""
    if not needed():
        return None
    logger.warning("Обнаружены JSON-данные и пустые таблицы — переношу")
    report = run()
    for r in report:
        logger.info("Перенос %s: из json %d → в базе %d%s",
                    r["table"], r["json"], r["db"], "" if r["ok"] else "  РАСХОЖДЕНИЕ")
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dry = "--dry-run" in sys.argv
    report = run(dry_run=dry)
    print(f"{'таблица':<12} {'в json':>8} {'в базе':>8}   статус")
    print("-" * 44)
    bad = 0
    for r in report:
        mark = "ok" if r["ok"] else "РАСХОЖДЕНИЕ"
        bad += 0 if r["ok"] else 1
        print(f"{r['table']:<12} {r['json']:>8} {r['db']:>8}   {mark}")
    print()
    print("dry-run: изменения откачены" if dry else "записано")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
