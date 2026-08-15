"""Архив фото и разборов зрения.

Зачем. 12.08.2026 Влад спросил «почему скрин распознался неверно», и ответить
было нечем: картинка нигде не сохранялась, в лог падала одна строка
«Vision: type=shift_schedule, shifts=5». Причину пришлось выяснять по обрывкам
переписки. Теперь хранится и файл, и полный ответ модели, и что из него в итоге
записали в данные.

Побочные выгоды, ради которых это и делалось:
  • появляется корпус для проверки распознавания — иначе тестировать не на чем;
  • «кинь тот график, что я скидывал» становится обычным поиском по описанию;
  • будущему «финансисту» будет из чего считать: скрины заработка перестают
    исчезать бесследно.

Место. На VM 38 ГБ свободно, телеграм жмёт фото до ~100–200 КБ. Даже десять
снимков в день это ~700 МБ в год. Лимит каталога всё равно задан: неограниченно
растущее хранилище на сервере — плохая идея независимо от арифметики.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import db
from utils.time import now_local

logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path("data/vision")
MAX_DIR_BYTES = 5 * 1024 ** 3      # 5 ГБ — потолок каталога
KEEP_ORIGINAL_DAYS = 180           # дальше остаются метаданные и разбор


def _dir() -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR


def _path_for(sha: str, ts) -> Path:
    sub = _dir() / ts.strftime("%Y") / ts.strftime("%m")
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{sha[:16]}.jpg"


def save(image_bytes: bytes, result: Dict[str, Any], chat_id: int = 0,
         model: str = "", applied: str = "") -> Optional[int]:
    """Сохранить фото и разбор. Возвращает id записи (или существующей).

    Никогда не бросает: архив — вспомогательная вещь, из-за него разбор фото
    падать не должен.
    """
    try:
        if not image_bytes:
            return None
        sha = hashlib.sha256(image_bytes).hexdigest()
        existing = db.query_one("SELECT id FROM vision_results WHERE sha256=?", (sha,))
        if existing:
            # То же самое фото прислали повторно — файл на диске уже есть.
            return int(existing["id"])

        ts = now_local()
        path = _path_for(sha, ts)
        path.write_bytes(image_bytes)

        tags = _tags_from(result)
        description = str(result.get("description") or "")
        cur = db.execute(
            "INSERT INTO vision_results(ts, chat_id, sha256, file_path, bytes, kind,"
            " description, tags, model, raw, applied, search_text)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts.isoformat(timespec="minutes"), chat_id, sha, str(path), len(image_bytes),
             str(result.get("type") or ""), description,
             json.dumps(tags, ensure_ascii=False), model,
             json.dumps(result, ensure_ascii=False), applied,
             _search_text(description, tags, "")),
        )
        enforce_limits()
        return cur.lastrowid
    except Exception:
        logger.warning("Архив зрения: не сохранил", exc_info=True)
        return None


def _tags_from(result: Dict[str, Any]) -> List[str]:
    """Теги для поиска. Берём то, что модель уже определила, — отдельный вызов
    ради тегов не нужен."""
    tags = [t for t in (str(result.get("type") or ""), str(result.get("food_kind") or ""))
            if t]
    if result.get("shifts"):
        tags += ["график", "смены", "работа"]
    if str(result.get("type")) == "food":
        tags += ["еда"]
    if result.get("dish"):
        tags.append(str(result["dish"]))
    tags += [str(i) for i in (result.get("items") or [])][:5]
    seen, out = set(), []
    for t in tags:
        low = t.strip().lower()
        if low and low not in seen:
            seen.add(low)
            out.append(t.strip())
    return out


def set_applied(record_id: Optional[int], applied: str) -> None:
    """Дописать, что по итогу записали в данные (смены/приём пищи/ничего)."""
    if not record_id:
        return
    try:
        db.execute("UPDATE vision_results SET applied=? WHERE id=?", (applied, record_id))
    except Exception:
        logger.debug("Архив зрения: applied не записан", exc_info=True)


def _search_text(description: str, tags: List[str], label: str) -> str:
    """Нормализованная строка для поиска.

    Собирается в Python, а не в SQL, потому что SQLite LOWER() опускает только
    ASCII: «Оливье» он оставляет как есть, и LIKE '%оливье%' проходит мимо.
    Кириллица — основной язык этих описаний, так что это не мелочь.
    """
    return " ".join([description, " ".join(tags), label]).lower()


def set_label(record_id: int, label: str) -> None:
    """Как Влад сам назвал фото («сохрани как график августа»)."""
    row = db.query_one("SELECT description, tags FROM vision_results WHERE id=?",
                       (record_id,))
    if row is None:
        return
    tags = json.loads(row["tags"] or "[]")
    db.execute("UPDATE vision_results SET label=?, search_text=? WHERE id=?",
               (label.strip(), _search_text(row["description"], tags, label), record_id))


def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Поиск по описанию, тегам и метке. «Кинь тот график» → файл.

    LIKE по нормализованной колонке, а не FTS5: записей тут сотни, не сотни
    тысяч, а отдельный индекс — лишняя сущность, которую надо синхронизировать.
    """
    words = [w for w in (query or "").lower().split() if len(w) >= 3][:6]
    if not words:
        return recent(limit)
    clauses = " OR ".join(["search_text LIKE ?"] * len(words))
    params: List[Any] = [f"%{w}%" for w in words]
    params.append(limit)
    rows = db.query(
        f"SELECT * FROM vision_results WHERE {clauses} ORDER BY ts DESC LIMIT ?", params)
    return [_row(r) for r in rows]


def recent(limit: int = 5) -> List[Dict[str, Any]]:
    return [_row(r) for r in
            db.query("SELECT * FROM vision_results ORDER BY ts DESC LIMIT ?", (limit,))]


def get(record_id: int) -> Optional[Dict[str, Any]]:
    row = db.query_one("SELECT * FROM vision_results WHERE id=?", (record_id,))
    return _row(row) if row else None


def _row(r) -> Dict[str, Any]:
    return {
        "id": r["id"], "ts": r["ts"], "kind": r["kind"],
        "description": r["description"], "label": r["label"],
        "tags": json.loads(r["tags"] or "[]"), "file_path": r["file_path"],
        "applied": r["applied"], "model": r["model"],
        "raw": json.loads(r["raw"] or "{}"), "exists": Path(r["file_path"]).exists(),
    }


def dir_size() -> int:
    return sum(p.stat().st_size for p in _dir().rglob("*.jpg") if p.is_file())


def enforce_limits() -> int:
    """Ретеншн: старые оригиналы и потолок каталога. Возвращает сколько удалено.

    Метаданные и разбор НЕ трогаем — они весят байты и нужны для истории.
    Удаляется только картинка, запись остаётся с пометкой отсутствия файла.
    """
    removed = 0
    try:
        cutoff = (now_local().date().toordinal() - KEEP_ORIGINAL_DAYS)
        for r in db.query("SELECT id, ts, file_path FROM vision_results ORDER BY ts"):
            path = Path(r["file_path"])
            if not path.exists():
                continue
            try:
                day = now_local().fromisoformat(r["ts"]).date().toordinal()
            except (ValueError, TypeError):
                continue
            if day < cutoff:
                path.unlink(missing_ok=True)
                removed += 1

        # Потолок каталога: чистим самое старое, пока не влезем.
        while dir_size() > MAX_DIR_BYTES:
            oldest = db.query_one(
                "SELECT id, file_path FROM vision_results ORDER BY ts LIMIT 1")
            if oldest is None:
                break
            path = Path(oldest["file_path"])
            if not path.exists():
                break
            path.unlink(missing_ok=True)
            removed += 1
            logger.warning("Архив зрения упёрся в потолок %d МБ — удаляю старое",
                           MAX_DIR_BYTES // 1024 ** 2)
    except Exception:
        logger.debug("Архив зрения: ретеншн не отработал", exc_info=True)
    return removed


def stats() -> Dict[str, Any]:
    total = db.query_one("SELECT COUNT(*) c FROM vision_results")["c"]
    with_file = sum(1 for r in db.query("SELECT file_path FROM vision_results")
                    if Path(r["file_path"]).exists())
    return {"records": total, "files": with_file, "bytes": dir_size()}


def purge_all() -> None:
    """Полная очистка — для тестов и ручного сброса."""
    shutil.rmtree(_dir(), ignore_errors=True)
    db.execute("DELETE FROM vision_results")
