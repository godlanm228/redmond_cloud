"""Единая SQLite-база хаба: соединения, схема, миграции.

Зачем уходим с JSON-файлов. Разбор 12–13.08.2026 нашёл целый класс потерь,
и все они — свойства формата, а не отдельные баги:
  • «прочитал → поменял → записал» без транзакции теряет параллельные записи;
  • запись не атомарна: падение посреди write оставляет обрубок;
  • битый файл раньше молча превращался в пустой (дневник → одна запись);
  • история изменений отсутствует: «откуда взялась эта смена» не ответить.
Транзакции SQLite закрывают всё это разом, а не по одному.

Файл — тот же `data/memory.sqlite`, где уже живёт таблица memory: одна база
проще бэкапить (`VACUUM INTO` даёт консистентный снимок без остановки сервиса).

Соединения — по одному на поток (генерация идёт в asyncio.to_thread), WAL,
busy_timeout: писатель не блокирует читателей, а конкурентная запись ждёт,
а не падает с 'database is locked'.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/memory.sqlite")
_BUSY_TIMEOUT_MS = 5000

_local = threading.local()
_db_path: Optional[Path] = None
_schema_lock = threading.Lock()
# Реестр всех выданных соединений: нужен, чтобы закрыть их разом при смене
# пути или на выходе. Потоки генерации живут в пуле и умирают, унося ссылку —
# без реестра файл остаётся открытым (на Windows это ещё и блокирует удаление).
_all_conns: list = []
_conns_guard = threading.Lock()


# Путь, для которого схема уже создана в этом процессе. WAL и DDL — операции
# уровня файла: их надо делать ОДИН раз, а не на каждом соединении. Иначе
# полтора десятка потоков одновременно берут эксклюзивную блокировку на
# CREATE TABLE и часть падает с 'database is locked'.
_schema_done_for: Optional[Path] = None


def close_all() -> None:
    """Закрыть все соединения. Для тестов и корректного завершения."""
    global _schema_done_for
    with _conns_guard:
        for c in _all_conns:
            try:
                c.close()
            except Exception:  # noqa: BLE001 — закрытие не должно ничего ронять
                pass
        _all_conns.clear()
    _schema_done_for = None
    _local.__dict__.pop("conn", None)
    _local.__dict__.pop("path", None)


def set_db_path(path: Any) -> None:
    """Переопределить путь к базе (конфиг, тесты). Закрывает старые соединения."""
    global _db_path
    close_all()
    _db_path = Path(path)


def db_path() -> Path:
    return _db_path or DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    """Соединение текущего потока. Схема создаётся один раз на процесс.

    Путь приводим к абсолютному: конфиг задаёт его относительно (data/…), а
    сравнение относительных путей после chdir считало бы «тот же путь» и
    возвращало соединение к файлу в СТАРОМ каталоге.
    """
    path = db_path().absolute()
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "path", None) == path:
        return conn

    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False только ради close_all() с другого потока —
    # само соединение по-прежнему используется лишь своим потоком (_local).
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000,
                           check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Автокоммит: транзакциями управляем сами через transaction() с BEGIN
    # IMMEDIATE. С отложенным BEGIN (поведение по умолчанию) два потока
    # спокойно читают одно значение, каждый пишет своё — и одно изменение
    # теряется. Ровно это тест и поймал на счётчике Gemini: 20 инкрементов
    # дали 2.
    conn.isolation_level = None
    # Пер-соединенческие настройки — дёшевы и блокировок не берут.
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    _local.conn = conn
    _local.path = path
    with _conns_guard:
        _all_conns.append(conn)
    _ensure_schema_once(conn, path)
    return conn


def _ensure_schema_once(conn: sqlite3.Connection, path: Path) -> None:
    """WAL и DDL — один раз на процесс и путь, под общей блокировкой."""
    global _schema_done_for
    if _schema_done_for == path:
        return
    with _schema_lock:
        if _schema_done_for == path:
            return
        conn.execute("PRAGMA journal_mode=WAL")
        init_schema(conn)
        _schema_done_for = path


# ---------------------------------------------------------------------------
# Схема
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 3

_SCHEMA = """
-- Цели и дедлайны: id остаётся сквозным, как в JSON (на него ссылаются tools).
CREATE TABLE IF NOT EXISTS goals (
    id           INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    why          TEXT DEFAULT '',
    target_date  TEXT,
    status       TEXT NOT NULL DEFAULT 'active',
    created      TEXT NOT NULL,
    closed       TEXT,
    progress_log TEXT NOT NULL DEFAULT '[]'   -- JSON-массив
);

CREATE TABLE IF NOT EXISTS deadlines (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    due        TEXT NOT NULL,
    importance TEXT NOT NULL DEFAULT 'medium',
    status     TEXT NOT NULL DEFAULT 'pending',
    created    TEXT NOT NULL,
    closed     TEXT,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_deadlines_due ON deadlines(due, status);

-- data — структурная нагрузка записи (еда: dish/kcal/protein/place; смена:
-- date/start/end). Есть у 14 записей из 84 на 15.08.2026; аналитический слой
-- будет считать тренды по ней, поэтому колонка нужна с самого начала.
CREATE TABLE IF NOT EXISTS diary (
    id   INTEGER PRIMARY KEY,
    ts   TEXT NOT NULL,
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',          -- JSON-массив
    data TEXT                                  -- JSON-объект или NULL
);
CREATE INDEX IF NOT EXISTS idx_diary_ts ON diary(ts);

CREATE TABLE IF NOT EXISTS pantry (
    item  TEXT PRIMARY KEY,
    added TEXT NOT NULL
);

-- Смены: текущее состояние. История — в shift_events (append-only).
CREATE TABLE IF NOT EXISTS shifts (
    date              TEXT PRIMARY KEY,
    start             TEXT,
    end               TEXT,
    status            TEXT NOT NULL DEFAULT 'planned',
    source            TEXT NOT NULL DEFAULT 'unknown',
    confidence        TEXT NOT NULL DEFAULT 'medium',
    updated           TEXT NOT NULL,
    last_confirmed_at TEXT,
    note              TEXT
);

-- Журнал изменений смен: отвечает на «откуда тут взялась эта смена».
-- 12.08.2026 в график попали чужие смены с фото, и восстановить это можно
-- было только вручную из логов.
CREATE TABLE IF NOT EXISTS shift_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    date    TEXT NOT NULL,
    action  TEXT NOT NULL,               -- set | cancel | delete | reject
    source  TEXT NOT NULL,               -- photo | text | manual | scheduler
    payload TEXT NOT NULL DEFAULT '{}',  -- JSON: что именно записали
    reason  TEXT                          -- почему отклонили/перезаписали
);
CREATE INDEX IF NOT EXISTS idx_shift_events_date ON shift_events(date, ts);

-- Мелкое состояние одним ключ-значением: day_state, presence, mute,
-- ping_style, radar, gemini_usage, week_plan. Заводить таблицу на каждое —
-- лишняя церемония, а транзакционность нужна одинаково.
CREATE TABLE IF NOT EXISTS kv (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,               -- JSON
    updated TEXT NOT NULL
);

-- История диалога. Раньше жила только в RAM и стиралась каждым рестартом
-- (13.08 их было пять подряд), плюс правилась из нескольких потоков без
-- синхронизации. Храним парами «реплика — ответ»: ровно в таком виде история
-- уходит в промпт, и разбирать её обратно из отдельных строк было бы незачем.
-- agent нужен, чтобы после рестарта восстановить sticky роутера: без него
-- «лол» сразу после перезапуска не находил, к кому относится.
CREATE TABLE IF NOT EXISTS chat_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    agent   TEXT NOT NULL DEFAULT '',
    user    TEXT NOT NULL,
    bot     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history ON chat_history(chat_id, id);
"""


# Колонки, добавленные в уже существующие таблицы. CREATE TABLE IF NOT EXISTS
# на живой базе не срабатывает вообще, поэтому новые поля надо доливать явно —
# иначе на проде, где таблица создана прошлой версией схемы, вставка падает на
# отсутствующей колонке. Формат: (таблица, колонка, определение).
_ADDED_COLUMNS = [
    ("diary", "data", "TEXT"),          # v3: структурная нагрузка записи
]


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, column, decl in _ADDED_COLUMNS:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            continue  # таблицу создаст _SCHEMA уже с этой колонкой
        columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            logger.info("Схема: добавляю %s.%s", table, column)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_schema(conn: sqlite3.Connection) -> None:
    """Идемпотентно: создаёт недостающие таблицы и колонки, поднимает user_version.
    Вызывается под _schema_lock из _ensure_schema_once (или напрямую в тестах)."""
    _add_missing_columns(conn)
    conn.executescript(_SCHEMA)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def query(sql: str, params: Iterable[Any] = ()) -> list:
    return connect().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
    return connect().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    """Одиночный statement. Соединение в автокоммите — commit не нужен."""
    return connect().execute(sql, tuple(params))


@contextmanager
def transaction():
    """Транзакция на «прочитал → поменял → записал».

    BEGIN IMMEDIATE берёт write-блокировку СРАЗУ, а не при первой записи:
    иначе два потока успевают прочитать одно и то же состояние до того, как
    кто-то из них начнёт писать, и одно из изменений исчезает. Плюс это
    убирает 'database is locked' — конкурент ждёт по busy_timeout, а не
    падает на попытке апгрейда уже начатой read-транзакции.
    """
    conn = connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def history_load(chat_id: int, limit: int) -> list:
    """Последние `limit` обменов чата, в хронологическом порядке."""
    rows = query(
        "SELECT ts, agent, user, bot FROM chat_history WHERE chat_id=?"
        " ORDER BY id DESC LIMIT ?",
        (chat_id, limit),
    )
    return [{"user": r["user"], "bot": r["bot"], "timestamp": r["ts"],
             "agent": r["agent"]} for r in reversed(rows)]


def history_add(chat_id: int, user_text: str, bot_text: str,
                agent: str = "", ts: str = "") -> None:
    execute(
        "INSERT INTO chat_history(chat_id, ts, agent, user, bot) VALUES(?,?,?,?,?)",
        (chat_id, ts or _now(), agent, user_text, bot_text),
    )


def history_trim(chat_id: int, keep: int) -> int:
    """Оставить последние `keep` обменов. Возвращает сколько удалено.

    История нужна для промпта (последние несколько реплик) — держать всё
    незачем, для долгой памяти есть таблица memory с полнотекстовым поиском.
    """
    cur = execute(
        "DELETE FROM chat_history WHERE chat_id=? AND id NOT IN"
        " (SELECT id FROM chat_history WHERE chat_id=? ORDER BY id DESC LIMIT ?)",
        (chat_id, chat_id, keep),
    )
    return cur.rowcount or 0


def kv_get(key: str, default: Any = None) -> Any:
    row = query_one("SELECT value FROM kv WHERE key=?", (key,))
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        logger.warning("kv[%s] не разбирается как JSON — отдаём дефолт", key)
        return default


def kv_set(key: str, value: Any, now: str = "") -> None:
    execute(
        "INSERT INTO kv(key, value, updated) VALUES(?,?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
        (key, json.dumps(value, ensure_ascii=False), now or _now()),
    )


def _now() -> str:
    from utils.time import now_local
    return now_local().isoformat(timespec="minutes")


def backup_to(path: Any) -> Path:
    """Консистентный снимок базы без остановки сервиса (VACUUM INTO).

    Бэкапов у проекта не было вообще: потеря data/ означала потерю всего.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        os.remove(dest)
    connect().execute("VACUUM INTO ?", (str(dest),))
    return dest
