import logging
import os
import sys

# Логгеры, которые при DEBUG-уровне корня превращают вывод в кашу.
# Им принудительно ставим уровень WARNING.
NOISY_LOGGERS = (
    "comtypes",
    "comtypes.client",
    "comtypes.client._code_cache",
    "comtypes.client._create",
    "comtypes.client._events",
    "comtypes.client._generate",
    "comtypes.client._managing",
    "comtypes._comobject",
    "comtypes._post_coinit",
    "comtypes._post_coinit.unknwn",
    "comtypes._vtbl",
    "urllib3",
    "urllib3.connectionpool",
    "asyncio",
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "huggingface_hub",
    "filelock",
    "matplotlib",
    "PIL",
    "websockets",
    "pygame",
)


class RedactingFormatter(logging.Formatter):
    """Маскирует известные секреты в КАЖДОЙ строке лога, включая traceback'и.

    Секреты попадают в лог не через наш код, а через чужие exception-тексты:
    requests вписывает в HTTPError полный URL (так Gemini-ключ из ?key=…
    месяцами лежал в v2.log). Логировать «аккуратнее» в 50 местах ненадёжно —
    маскируем на единственной точке, через которую проходит весь вывод.

    Что маскируем: значения env-переменных с именами *_TOKEN / *_KEY /
    *_SECRET / *_PASSWORD длиной ≥ 8 (все бот-токены, Groq/Gemini-ключи,
    Claude-токен). Снимается снапшотом на старте — setup_logging зовётся
    после load_dotenv.
    """

    _NAME_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        secrets = {
            v for k, v in os.environ.items()
            if v and len(v) >= 8 and k.upper().endswith(self._NAME_SUFFIXES)
        }
        # Длинные первыми: если один секрет — подстрока другого, не оставляем хвост.
        self._secrets = sorted(secrets, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        for secret in self._secrets:
            if secret in out:
                out = out.replace(secret, "***REDACTED***")
        return out


def setup_logging(level: str = "INFO") -> None:
    """
    Корневой логгер пишет в stdout. Шумные модули (comtypes/urllib3/asyncio
    и пр.) принудительно глушатся до WARNING, чтобы DEBUG-режим оставался
    читаемым. Формат — с редакцией секретов (см. RedactingFormatter).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    formatter = RedactingFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
