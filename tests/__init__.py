"""Пакет тестов.

Запускать через pytest: изоляцию базы по тестам даёт conftest.py, и без него
состояние течёт между тестами (`python -m unittest` покажет каскад падений —
это не баги кода, а отсутствие изоляции).

Здесь же — страховка на случай любого запуска: база сразу уводится во
временный каталог. Без неё путь по умолчанию (data/memory.sqlite) в рабочем
дереве указывал бы на БОЕВУЮ базу, и прогон тестов писал бы в неё.
"""

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="redmond-tests-")

try:
    from utils import db

    db.set_db_path(Path(_TMP) / "hub.sqlite")
except Exception:  # noqa: BLE001 — тесты сами упадут понятнее, чем импорт
    pass


@atexit.register
def _cleanup() -> None:
    try:
        from utils import db

        db.close_all()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(_TMP, ignore_errors=True)
