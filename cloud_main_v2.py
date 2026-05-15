"""
Cloud Redmond v2 — multi-bot режим.

4 параллельных Telegram Application'а в одном процессе:
  • Redmond  (@redmond_hub_bot)    — router + общий ассистент
  • Iris     (@iris_redberry_bot)   — личный коуч / трекер
  • Newser   (@newser_redmond_bot)  — searcher / новости
  • Cipher   (@cipher_redberry_bot) — Claude Code через Pro подписку

Запуск:
    python cloud_main_v2.py

Замена для cloud_main.py — старый можно держать как fallback пока v2 не оттестирован.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import load_app_config  # noqa: E402
from core.dispatcher import Dispatcher  # noqa: E402
from core.multi_bot_runner import run_multi_bot  # noqa: E402
from utils.logging_config import setup_logging  # noqa: E402


logger = logging.getLogger("redmond-cloud-v2")

# Подавляем шумные библиотеки
for noisy in ("httpx", "httpcore", "telegram", "apscheduler"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    config = load_app_config()
    setup_logging(config.log_level)

    allowed = os.getenv("ALLOWED_USER_IDS") or "(open)"
    logger.info("=== Cloud Redmond v2 starting ===")
    logger.info("Whitelist: %s", allowed)

    # Один Dispatcher на всех — содержит response_generator, intent_recognizer,
    # safety, auth. State в нём stateless или защищён shared dict.
    dispatcher = Dispatcher(config)

    try:
        asyncio.run(run_multi_bot(dispatcher))
    except KeyboardInterrupt:
        logger.info("Stopped by KeyboardInterrupt")


if __name__ == "__main__":
    main()
