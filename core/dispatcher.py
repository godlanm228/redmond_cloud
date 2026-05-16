import logging

from config.config_loader import get_supergoals, load_app_config
from logic.intent_recognizer import IntentRecognizer
from logic.response_generator import ResponseGenerator
from safety.goal_manager import GoalManager

logger = logging.getLogger(__name__)


class Dispatcher:
    """Контейнер для общих сервисов v2 (safety / intent / response_generator).

    В v2 multi-bot цепочке метод `dispatch()` не используется — handlers
    обращаются напрямую к `self.response_generator`. Класс остаётся обёрткой
    для удобного создания и shared-access из bot_data. Дальнейшее упрощение
    — см. backlog «Dispatcher refactor pass».
    """

    def __init__(self, config=None):
        self.config = config or load_app_config()

        supergoals = get_supergoals(self.config)
        self.safety = GoalManager(supergoals)

        self.intent_recognizer = IntentRecognizer()
        self.response_generator = ResponseGenerator(self.config)
