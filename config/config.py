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

    # ---------- сеть / процессы ----------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1, le=32)

    # ---------- директории ----------
    data_dir: Path = Field(default=Path("data"))

    # ---------- LLM провайдеры (fallback цепочка) ----------
    llm_provider_order: List[str] = Field(
        default=["groq", "ollama", "gemini", "transformers"],
        description="Порядок попыток провайдеров — первый успешный отдаёт ответ",
    )
    llm_endpoint: str = Field(default="http://localhost:8001")
    llm_model_path: str = Field(default="EleutherAI/gpt-neo-1.3B")

    groq_api_key: str = Field(default="", description="env: REDMOND_GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    groq_fallback_model: str = Field(
        default="qwen/qwen3-32b",
        description="Используется при rate_limit / tool_use_failed на primary модели",
    )
    groq_api_base: str = Field(default="https://api.groq.com")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b-instruct")

    gemini_api_key: str = Field(default="", description="env: REDMOND_GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash")

    # ---------- ASR / TTS ----------
    # Голосовые в TG транскрибируются через Groq Whisper API (free tier) —
    # локальный Whisper на VM с 1 GB RAM невозможен.
    groq_whisper_model: str = Field(default="whisper-large-v3-turbo")

    tts_engine: str = Field(default="edge-tts", description="'edge-tts' | 'pyttsx3' | 'console'")
    edge_tts_voice: str = Field(
        default="en-US-BrianMultilingualNeural",
        description="Edge TTS voice (BrianMultilingual — мультиязычный Jarvis-стайл)",
    )
    edge_tts_rate: str = Field(default="+0%", description="Edge TTS rate offset, e.g. '+10%'")

    # ---------- внешние сервисы ----------
    google_api_key: str = Field(default="", description="env: REDMOND_GOOGLE_API_KEY")
    google_search_engine_id: str = Field(default="", description="env: REDMOND_GOOGLE_SEARCH_ENGINE_ID")
    google_search_daily_limit: int = Field(default=100, ge=1)

    # ---------- профили и память ----------
    supergoals_file: str = Field(default="config/supergoals.json")
    personality_profile: str = Field(default="config/personality_profile.json")
    owner_profile: str = Field(default="config/owner_profile.json")
    baseline_db_path: str = Field(default="data/memory.sqlite")
    max_memory_records: int = Field(default=50000, ge=100)

    # ---------- общее ----------
    log_level: str = Field(default="INFO")
    idle_timeout_sec: int = Field(default=300, ge=10)
    max_history: int = Field(default=6, ge=1)
    top_k: int = Field(default=3, ge=1)
    search_timeout_sec: int = Field(default=10, ge=1)

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
            "google_api_key": "REDMOND_GOOGLE_API_KEY",
            "google_search_engine_id": "REDMOND_GOOGLE_SEARCH_ENGINE_ID",
            "groq_api_key": "REDMOND_GROQ_API_KEY",
            "gemini_api_key": "REDMOND_GEMINI_API_KEY",
        }
        for field_name, env_var in env_map.items():
            env_val = os.environ.get(env_var)
            if env_val:
                object.__setattr__(self, field_name, env_val)

        return self
