import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator


class AppConfig(BaseModel):
    """Конфигурация приложения. Секреты подтягиваются из переменных окружения."""

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }

    # ---------- директории ----------
    data_dir: Path = Field(default=Path("data"))

    # ---------- LLM провайдеры (fallback цепочка) ----------
    llm_provider_order: List[str] = Field(
        default=["groq", "gemini"],
        description="Порядок попыток провайдеров — первый успешный отдаёт ответ",
    )

    groq_api_key: str = Field(default="", description="env: REDMOND_GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_fallback_model: str = Field(
        default="qwen/qwen3.6-27b",
        description="Используется при rate_limit / tool_use_failed на primary модели",
    )
    groq_api_base: str = Field(default="https://api.groq.com")

    gemini_api_key: str = Field(default="", description="env: REDMOND_GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash")

    # ---------- ASR ----------
    # Голосовые в TG транскрибируются через Groq Whisper API (free tier) —
    # локальный Whisper на VM с 1 GB RAM невозможен.
    groq_whisper_model: str = Field(default="whisper-large-v3-turbo")

    # ---------- профили и память ----------
    supergoals_file: str = Field(default="config/supergoals.json")
    personality_profile: str = Field(default="config/personality_profile.json")
    owner_profile: str = Field(default="config/owner_profile.json")
    baseline_db_path: str = Field(default="data/memory.sqlite")
    max_memory_records: int = Field(default=50000, ge=100)

    # ---------- общее ----------
    log_level: str = Field(default="INFO")
    max_history: int = Field(default=6, ge=1)
    top_k: int = Field(default=3, ge=1)

    # ---------- валидаторы ----------

    @field_validator("data_dir", mode="before")
    @classmethod
    def ensure_dir(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @model_validator(mode="after")
    def overlay_env_secrets(self):
        """Секреты из окружения имеют приоритет над config.json."""
        env_map = {
            "groq_api_key": "REDMOND_GROQ_API_KEY",
            "gemini_api_key": "REDMOND_GEMINI_API_KEY",
        }
        for field_name, env_var in env_map.items():
            env_val = os.environ.get(env_var)
            if env_val:
                object.__setattr__(self, field_name, env_val)

        return self
