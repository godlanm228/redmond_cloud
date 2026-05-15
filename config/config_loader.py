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

class ConfigurationError(Exception):
    """Ошибка конфигурации"""
    pass

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
        ConfigurationError: Если файл не найден или невалидный
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

def load_rules(rules_path: Optional[Union[Path, str]] = None) -> Any:
    """Загружает правила системы"""
    file = Path(rules_path) if rules_path else Path(__file__).parent / 'rules.json'

    if not file.exists():
        # Возвращаем дефолтные правила
        logger.warning(f"Rules file not found: {file}, using defaults")
        return {
            "rules": [],
            "forbidden_actions": [
                "delete_system_files",
                "modify_own_code",
                "disable_safety"
            ]
        }

    try:
        data = json.loads(file.read_text(encoding='utf-8'))
        return data
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in rules file: {e}")

def load_personality_profile(profile_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Загружает профиль личности"""
    file = Path(profile_path) if profile_path else Path(__file__).parent / 'personality_profile.json'

    default_profile = {
        "name": "Redmond",
        "style": "sarcastic but strict",
        "traits": ["analytical", "protective", "efficient"],
        "tone_variations": {
            "normal": "professional",
            "alert": "urgent",
            "casual": "friendly"
        }
    }

    if not file.exists():
        logger.warning(f"Personality profile not found: {file}, using defaults")
        return default_profile

    try:
        data = json.loads(file.read_text(encoding='utf-8'))

        # Валидация и дополнение дефолтными значениями
        profile = default_profile.copy()
        profile.update(data)

        return profile

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in personality profile: {e}")


def load_owner_profile(profile_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Загружает профиль владельца (имя, тайм-зона, факты)."""
    file = Path(profile_path) if profile_path else Path(__file__).parent / 'owner_profile.json'

    default = {
        "name": "",
        "timezone": "UTC",
        "preferences": {"language": "ru"},
        "goals": [],
        "important_people": [],
        "known_facts": [],
    }

    if not file.exists():
        return default

    try:
        data = json.loads(file.read_text(encoding='utf-8'))
        profile = default.copy()
        profile.update({k: v for k, v in data.items() if not k.startswith("_")})
        return profile
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in owner profile, using defaults: {e}")
        return default


def save_owner_profile(profile: Dict[str, Any], profile_path: Optional[Union[Path, str]] = None) -> None:
    """Сохраняет профиль владельца обратно на диск."""
    file = Path(profile_path) if profile_path else Path(__file__).parent / 'owner_profile.json'
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding='utf-8')

def save_config(config: AppConfig, path: Optional[Union[Path, str]] = None) -> None:
    """
    Сохраняет конфигурацию в файл.

    Args:
        config: Объект конфигурации
        path: Путь для сохранения
    """
    if not path:
        path = Path.cwd() / 'config' / 'config.json'
    else:
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Конвертируем в dict и сохраняем
    data = config.dict()

    # Преобразуем Path объекты в строки
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    logger.info(f"Configuration saved to: {path}")