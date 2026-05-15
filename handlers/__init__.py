"""
Registry-based handlers по интентам.

Каждый файл в этом пакете регистрирует один или несколько handler'ов
через декоратор `@register("intent_name")`. Dispatcher вызывает
`run_handler(intent_name, ctx)` — registry находит подходящий и
исполняет.

Handler возвращает строковый ответ или None если не обработал
(тогда вызывающий идёт по обычному LLM-пайплайну).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# intent_name -> handler callable(ctx) -> Optional[str]
_REGISTRY: Dict[str, Callable[[Any], Optional[str]]] = {}


def register(intent_name: str):
    """Декоратор: регистрирует функцию как handler для указанного интента."""
    def decorator(fn: Callable[[Any], Optional[str]]):
        if intent_name in _REGISTRY:
            logger.warning("Handler '%s' переопределён (%s)", intent_name, fn.__name__)
        _REGISTRY[intent_name] = fn
        return fn
    return decorator


def get_handler(intent_name: str) -> Optional[Callable[[Any], Optional[str]]]:
    return _REGISTRY.get(intent_name)


def run_handler(intent_name: str, ctx: Any) -> Optional[str]:
    fn = _REGISTRY.get(intent_name)
    if fn is None:
        return None
    try:
        return fn(ctx)
    except Exception:
        logger.exception("Handler '%s' raised", intent_name)
        return None


def list_handlers() -> Dict[str, str]:
    return {name: fn.__module__ + "." + fn.__name__ for name, fn in _REGISTRY.items()}


def _autodiscover() -> None:
    """Импортирует все модули пакета — декораторы регистрируют себя сами."""
    import handlers as pkg
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name.startswith("_"):
            continue
        importlib.import_module(f"handlers.{mod_info.name}")


_autodiscover()
