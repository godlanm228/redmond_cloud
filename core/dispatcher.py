import logging
import re
from typing import Optional

from config.config_loader import get_supergoals, load_app_config
from core.exceptions import UnsafeAction
from logic.intent_recognizer import Intent, IntentRecognizer
from logic.response_generator import ResponseGenerator
from safety.goal_manager import GoalManager
from utils.searcher import GoogleSearchLimitExceeded

logger = logging.getLogger(__name__)


class AuthManager:
    """Простая ролевая модель: guest → user → owner по паролям."""

    def __init__(self, basic_password: str, super_password: str):
        self.basic_password = basic_password
        self.super_password = super_password
        self.role = "guest"

    def try_login(self, text: str) -> Optional[str]:
        m = re.match(r"^(?:login|войти)\s+(.+)$", text, flags=re.IGNORECASE)
        if not m:
            return None
        pwd = m.group(1).strip()
        if self.super_password and pwd == self.super_password:
            self.role = "owner"
            return "owner"
        if self.basic_password and pwd == self.basic_password:
            self.role = "user"
            return "user"
        return None

    def is_guest(self) -> bool:
        return self.role == "guest"

    def is_user(self) -> bool:
        return self.role == "user"

    def is_owner(self) -> bool:
        return self.role == "owner"


class Dispatcher:
    """Маршрутизация распознанного текста: auth → safety → intent → handler → TTS."""

    BASIC_INTENTS = {"analyze_sales", "show_report", "get_file_info"}

    def __init__(self, config=None):
        self.config = config or load_app_config()

        supergoals = get_supergoals(self.config)
        self.safety = GoalManager(supergoals)

        self.auth = AuthManager(
            self.config.basic_password,
            self.config.super_password,
        )

        self.intent_recognizer = IntentRecognizer()
        self.response_generator = ResponseGenerator(self.config)

        self.tts = None
        self._sleeping = False

    def bind_tts(self, tts) -> None:
        self.tts = tts

    async def _say(self, text: str) -> str:
        """Вернуть и озвучить ответ. Возвращает строку, чтобы вызывающий мог её использовать."""
        if self.tts is not None and text:
            await self.tts.say(text)
        return text or ""

    async def dispatch(self, text: str) -> str:
        """Обработать пользовательский ввод. Возвращает текст ответа (для логов/Unity)."""
        txt = (text or "").strip().lower()
        if not txt:
            return ""

        logger.info("IN [%s]: %r", self.auth.role, txt)

        # Sleep / wake
        if txt in ("спать", "sleep"):
            self._sleeping = True
            return await self._say("Перехожу в режим сна. Жду команду «проснись».")

        if self._sleeping:
            if txt in ("проснись", "wake up"):
                self._sleeping = False
                return await self._say("Я снова онлайн.")
            return ""

        # Login
        new_role = self.auth.try_login(txt)
        if new_role:
            return await self._say(f"Роль обновлена: {new_role}.")

        # Safety check
        try:
            self.safety.assert_safe(text)
        except UnsafeAction as e:
            logger.warning("Blocked unsafe input: %s", e)
            return await self._say("Не могу выполнить эту команду.")

        # Intent
        intent: Intent = self.intent_recognizer.recognize(text)
        logger.debug("Intent: %s %s", intent.name, intent.slots)

        # Аналитика и поиск — гейт по роли. Сам обработчик живёт в handlers/.
        if intent.name in self.BASIC_INTENTS and self.auth.is_guest():
            return await self._say("Сначала войдите в систему.")

        try:
            resp = self.response_generator.generate(intent, text, self.auth.role)
        except GoogleSearchLimitExceeded:
            return await self._say("Лимит Google Search исчерпан — ответы могут быть неактуальны.")

        if not resp:
            return await self._say("Не знаю, что ответить.")
        return await self._say(resp)
