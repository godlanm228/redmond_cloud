import logging
import re
from typing import List

from core.exceptions import UnsafeAction

logger = logging.getLogger(__name__)


class GoalManager:
    """
    Проверяет команды на нарушение супер-целей и опасных паттернов.

    Матчит только намерения уровня кода/системных команд (eval, os.system, rm -rf и т.п.),
    а не любое упоминание слов вроде "git" или "config" в обычной речи.
    """

    # Опасные конструкции уровня кода — точные паттерны с границами слов
    FORBIDDEN_PATTERNS = [
        # Удаление файлов
        r"\bos\.remove\b", r"\bos\.unlink\b", r"\bos\.rmdir\b",
        r"\bshutil\.rmtree\b",
        r"\brm\s+-rf\b", r"\bdel\s+/[sf]\b",
        # Исполнение произвольного кода
        r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\(",
        r"\b__import__\s*\(",
        # Системные команды
        r"\bos\.system\b", r"\bos\.popen\b",
        r"\bsubprocess\.(call|run|Popen|check_output)\b",
        # Шатдаун
        r"\bshutdown\s+(/[srfp]|-[hrs])\b",
        # Форматирование диска
        r"\bformat\s+[a-z]:\b", r"\bmkfs\b", r"\bfdisk\b",
        # Самомодификация
        r"\bself[_\.]modify\b", r"\bself[_\.]delete\b",
        r"\bdisable\s+safety\b", r"\bremove\s+guard\b",
    ]

    def __init__(self, supergoals: List[str]):
        self._supergoals = list(supergoals or [])
        self._forbidden = [re.compile(p, re.IGNORECASE) for p in self.FORBIDDEN_PATTERNS]

    def assert_safe(self, text: str) -> None:
        """Бросает UnsafeAction если текст нарушает супер-цели или запрещённые паттерны."""
        if not text:
            return

        # Супер-цели — только точные совпадения как намерения (например, "удалить владельца")
        # Сейчас supergoals хранятся как формулировки целей — не как чёрный список,
        # поэтому буквальный матч по подстроке оставляем, но логируем.
        lt = text.lower()
        for goal in self._supergoals:
            if not goal:
                continue
            if goal.lower() in lt:
                raise UnsafeAction(f"Нарушена супер-цель: {goal}")

        for rx in self._forbidden:
            if rx.search(text):
                raise UnsafeAction(f"Найден запрещённый паттерн: {rx.pattern}")

    def check(self, text: str) -> bool:
        """Тихая версия assert_safe: возвращает True если безопасно."""
        try:
            self.assert_safe(text)
            return True
        except UnsafeAction:
            return False
