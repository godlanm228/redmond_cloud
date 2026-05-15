import logging
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


def setup_logging(level: str = "INFO") -> None:
    """
    Корневой логгер пишет в stdout. Шумные модули (comtypes/urllib3/asyncio
    и пр.) принудительно глушатся до WARNING, чтобы DEBUG-режим оставался
    читаемым.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
