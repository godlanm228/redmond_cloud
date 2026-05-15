"""Информация о файле / директории по пути."""

import logging
import os
from datetime import datetime
from typing import Optional

from handlers import register

logger = logging.getLogger(__name__)


@register("get_file_info")
def handle_get_file_info(ctx) -> Optional[str]:
    filepath = ctx.intent.slots.get("path", "").strip()
    if not filepath:
        return "Укажите путь к файлу."
    if not os.path.exists(filepath):
        return f"Файл «{filepath}» не найден."

    try:
        stat = os.stat(filepath)
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        kind = "Директория" if os.path.isdir(filepath) else "Файл"
        return (
            f"Информация о «{filepath}»:\n"
            f"- Размер: {size_mb:.2f} МБ\n"
            f"- Изменён: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- Тип: {kind}"
        )
    except Exception:
        logger.exception("file_info error")
        return f"Ошибка при получении информации о «{filepath}»."
