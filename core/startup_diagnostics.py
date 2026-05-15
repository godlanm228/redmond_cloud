import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TextIO


class ModuleSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    OPTIONAL = "OPTIONAL"


class ModuleState(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class ModuleHealth:
    name: str
    state: ModuleState
    severity: ModuleSeverity
    details: str = ""


class StartupDiagnostics:
    """Сборщик статусов модулей при старте, печатает в stream и выдаёт сводку."""

    MODULE_NAME_WIDTH = 24

    def __init__(self, stream: Optional[TextIO] = None):
        self._stream = stream or sys.stdout
        self._items: List[ModuleHealth] = []

    def ok(self, name: str, details: str = "", severity: ModuleSeverity = ModuleSeverity.OPTIONAL) -> None:
        self._add(name, ModuleState.OK, severity, details)

    def warn(self, name: str, details: str = "", severity: ModuleSeverity = ModuleSeverity.OPTIONAL) -> None:
        self._add(name, ModuleState.WARN, severity, details)

    def error(self, name: str, details: str = "", severity: ModuleSeverity = ModuleSeverity.OPTIONAL) -> None:
        self._add(name, ModuleState.ERROR, severity, details)

    def _add(self, name: str, state: ModuleState, severity: ModuleSeverity, details: str) -> None:
        item = ModuleHealth(name=name, state=state, severity=severity, details=details.strip())
        self._items.append(item)
        print(self._format_line(item), file=self._stream, flush=True)

    def _format_line(self, item: ModuleHealth) -> str:
        dots = "." * max(2, self.MODULE_NAME_WIDTH - len(item.name))
        line = f"[INIT] {item.name} {dots} {item.state.value}"
        if item.details:
            line += f" ({item.details})"
        return line

    def summary(self) -> None:
        ok = sum(1 for i in self._items if i.state == ModuleState.OK)
        warn = sum(1 for i in self._items if i.state == ModuleState.WARN)
        err = sum(1 for i in self._items if i.state == ModuleState.ERROR)
        print(f"[INIT] SUMMARY: {ok} OK / {warn} WARN / {err} ERROR", file=self._stream, flush=True)

    def has_critical_errors(self) -> bool:
        return any(
            i.state == ModuleState.ERROR and i.severity == ModuleSeverity.CRITICAL
            for i in self._items
        )
