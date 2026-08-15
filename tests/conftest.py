"""Общая подготовка тестов.

Хранилище с 15.08.2026 — SQLite, и соединение живёт на уровне процесса. Без
явного закрытия после каждого теста файл остаётся открытым: на Windows это
блокирует удаление временного каталога, а между тестами утекает состояние.
Фикстура ниже снимает это со всех тестов разом, включая старые.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# requests тянут почти все модули логики, а в тестовом окружении его может не
# быть — сеть тут всё равно не нужна.
if "requests" not in sys.modules:
    _stub = types.ModuleType("requests")
    _stub.utils = types.SimpleNamespace(quote=lambda s: s)
    _stub.get = lambda *a, **kw: None
    _stub.post = lambda *a, **kw: None
    sys.modules["requests"] = _stub


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    """Каждому тесту — своя база, вне его собственного временного каталога.

    Старые тесты делают chdir во временный каталог и удаляют его в tearDown,
    который отрабатывает РАНЬШЕ финализации фикстур. Если база лежит внутри
    этого каталога, удаление падает на открытом файле. Путь абсолютный, так
    что chdir на него не влияет; тесты, которым нужна своя база, спокойно
    переопределяют путь сами.
    """
    from utils import db
    db.close_all()
    db.set_db_path(tmp_path / "hub.sqlite")
    yield
    db.close_all()
    db.set_db_path(db.DEFAULT_DB_PATH)
