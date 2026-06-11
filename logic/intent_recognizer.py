import re
from typing import Dict, NamedTuple


class Intent(NamedTuple):
    name: str
    slots: Dict[str, str]


class IntentRecognizer:
    """Rule-based распознавание интентов. Всё неопознанное уходит в 'chat'.

    PC-наследие (login/joke/search/mark_important/отчёты/file_info) снесено
    2026-06-11 вместе с registry-хендлерами: они перехватывали сообщения до
    агентного роутинга. Остался минимум, который реально влияет на пайплайн:
    weather (пропускает memory-enhance) и chat (всё остальное).
    """

    def recognize(self, text: str) -> Intent:
        t = text.lower().strip()

        if "погода" in t:
            return Intent("weather", {})

        return Intent("chat", {"text": text})
