import re
from typing import Dict, NamedTuple


class Intent(NamedTuple):
    name: str
    slots: Dict[str, str]


class IntentRecognizer:
    """Rule-based распознавание интентов. Всё неопознанное уходит в 'chat'."""

    def recognize(self, text: str) -> Intent:
        t = text.lower().strip()

        if re.match(r"(login|войти)\s+\S+", t):
            return Intent("login", {})

        if "анализ продаж" in t:
            return Intent("analyze_sales", {})

        if "показать отчет" in t or "показать отчёт" in t:
            return Intent("show_report", {})

        m = re.search(r"информация о файле\s+(.+)", t)
        if m:
            return Intent("get_file_info", {"path": m.group(1).strip()})

        if "погода" in t:
            return Intent("weather", {})

        if "шутк" in t:
            return Intent("joke", {})

        if t.startswith("сохрани это") or "важно" in t:
            return Intent("mark_important", {})

        if t.startswith("найди ") or t.startswith("поищи ") or t.startswith("search "):
            query = re.sub(r"^(найди|поищи|search)\s+", "", t)
            return Intent("search", {"query": query})

        return Intent("chat", {"text": text})
