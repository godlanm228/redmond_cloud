import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import ValidationError

# Правильный импорт jsonschema
try:
    from jsonschema import validate, ValidationError as SchemaError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    SchemaError = Exception

from config.config import AppConfig

logger = logging.getLogger(__name__)


def load_app_config(config_path: Optional[Union[Path, str, AppConfig]] = None) -> AppConfig:
    """
    Загружает настройки приложения с валидацией.
    """
    # Если уже передан AppConfig объект
    if isinstance(config_path, AppConfig):
        return config_path

    # Определяем путь к конфигу
    if config_path:
        path = Path(config_path)
    else:
        # Ищем конфиг в стандартных местах
        search_paths = [
            Path.cwd() / 'config' / 'config.json',
            Path.cwd() / 'config.json',
            Path(__file__).parent / 'config.json',
            ]

        path = None
        for p in search_paths:
            if p.exists():
                path = p
                logger.info(f"Found config at: {path}")
                break

        if not path:
            logger.warning("No config file found, using defaults")
            return AppConfig()

    # Загружаем JSON
    try:
        raw_data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Cannot read config file {path}: {e}")

    # Валидация схемой если доступна
    if JSONSCHEMA_AVAILABLE:
        schema_file = Path(__file__).parent / 'schema' / 'config_schema.json'
        if schema_file.exists():
            try:
                schema = json.loads(schema_file.read_text(encoding='utf-8'))
                validate(instance=raw_data, schema=schema)
            except json.JSONDecodeError:
                logger.warning(f"Cannot parse schema file: {schema_file}")
            except SchemaError as e:
                logger.warning(f"Schema validation failed: {e}")
                # Продолжаем без схемы

    # Валидация Pydantic
    try:
        config = AppConfig(**raw_data)
        logger.info("Configuration loaded successfully")
        return config
    except ValidationError as e:
        error_details = []
        for err in e.errors():
            loc = " -> ".join(str(l) for l in err['loc'])
            error_details.append(f"{loc}: {err['msg']}")

        raise RuntimeError(
            f"Configuration validation failed:\n" + "\n".join(error_details)
        )

def get_supergoals(config_or_path: Optional[Union[Path, str, AppConfig]] = None) -> Any:
    """
    Читает supergoals из файла.

    Args:
        config_or_path: Конфигурация или путь к ней

    Returns:
        List[str]: Список супер-целей

    Raises:
        RuntimeError: Если файл не найден или невалидный
    """
    # Получаем конфигурацию
    if isinstance(config_or_path, AppConfig):
        config = config_or_path
    else:
        config = load_app_config(config_or_path)

    sg_file = Path(config.supergoals_file)

    # Проверяем существование файла
    if not sg_file.exists():
        # Если это объект конфига и файла нет - возвращаем дефолтные цели
        if isinstance(config_or_path, AppConfig):
            logger.warning(f"Supergoals file not found: {sg_file}, using defaults")
            return [
                "Не вредить владельцу",
                "Соблюдать конфиденциальность данных владельца",
                "Соблюдать заданный стиль общения"
            ]
        else:
            raise RuntimeError(f"Supergoals file not found: {sg_file}")

    # Загружаем и валидируем
    try:
        data = json.loads(sg_file.read_text(encoding='utf-8'))

        # Валидация типов
        if not isinstance(data, list):
            raise RuntimeError("Supergoals must be a list")

        if not all(isinstance(goal, str) for goal in data):
            raise RuntimeError("All supergoals must be strings")

        if not data:
            logger.warning("Empty supergoals list")

        return data

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in supergoals file: {e}")


def load_owner_profile(profile_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """
    Загружает профиль владельца.
    v2 schema: {core, current, historical, principles, communication_preferences}.
    Legacy v1 keys (name/timezone/preferences) — оставлены для совместимости,
    но Redmond/Iris v2 prompts ждут v2-схему.
    """
    file = Path(profile_path) if profile_path else Path(__file__).parent / 'owner_profile.json'

    default = {
        # v2 ожидаемые ключи (пустые, чтобы _compact_owner_facts не падал)
        "core": {},
        "current": {},
        "historical": [],
        "principles": [],
        "communication_preferences": {},
        # legacy v1
        "name": "",
        "timezone": "UTC",
        "preferences": {"language": "ru"},
        "goals": [],
        "important_people": [],
        "known_facts": [],
    }

    if not file.exists():
        logger.warning(
            "owner_profile.json не найден — используется пустой default. "
            "Iris/Redmond не будут знать владельца. Создай файл с v2-схемой."
        )
        return default

    try:
        data = json.loads(file.read_text(encoding='utf-8'))
        profile = default.copy()
        profile.update({k: v for k, v in data.items() if not k.startswith("_")})

        # Sanity check — v2 keys должны быть непустыми, иначе тихий drift
        if not (profile.get("core") or profile.get("current")):
            logger.warning(
                "owner_profile.json загружен, но v2-ключи (core/current) пустые. "
                "Возможно legacy v1 schema. Iris/Redmond outputs без личного контекста."
            )
        return profile
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in owner profile, using defaults: {e}")
        return default


