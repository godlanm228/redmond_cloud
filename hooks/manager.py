import logging
from typing import Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

# Тип корутины-хука без аргументов
Hook = Callable[[], Awaitable[None]]

class HookManager:
    """
    Хранит и запускает списки корутин-хуков по имени события.
    """
    def __init__(self):
        self._hooks: Dict[str, List[Hook]] = {}

    def register(self, event: str, hook: Hook) -> None:
        """
        Зарегистрировать корутину hook на событие event.
        """
        self._hooks.setdefault(event, []).append(hook)
        logger.debug("Hook registered: %s -> %s", event, hook)

    async def run(self, event: str) -> None:
        """
        Запустить все корутины, привязанные к событию event.
        """
        hooks = self._hooks.get(event, [])
        logger.debug("Running %d hooks for event '%s'", len(hooks), event)
        for hook in hooks:
            try:
                await hook()
            except Exception:
                logger.exception("Error in hook '%s'", event)