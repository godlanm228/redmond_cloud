from .intent_matcher import match_intent

class IntentProcessor:
    def __init__(self):
        self.context = {}

    def process_input(self, text: str) -> str:
        intent = match_intent(text)
        if intent == "productivity_check":
            return "Вы провели сегодня 3 часа за VSCode. Продолжайте в том же духе."
        elif intent == "request_info":
            return "Вот пошаговый рецепт приготовления риса, сэр. Вывести на экран?"
        elif intent == "block_distraction":
            return "Discord будет заблокирован на 30 минут. Сосредоточьтесь, сэр."
        elif intent == "greeting":
            return "Приветствую вас, сэр. Чем займёмся?"
        elif intent == "fallback":
            return "Не понял намерение, сэр. Хотите, чтобы я уточнил?"
        else:
            return f"Обнаружено намерение: {intent}"
