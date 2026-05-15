class EngineStop(Exception):
    """Сигнал для тестов остановить движок."""
    pass

class UnsafeAction(Exception):
    """Нарушение супер-цели или forbidden-паттерна."""
    pass

class SecurityBreach(Exception):
    """Обнаружена угроза или атака — для внутренних нужд."""
    pass