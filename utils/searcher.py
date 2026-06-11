import logging
from typing import Dict, List, Tuple

from config.config_loader import load_app_config

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    Fallback-поиск через DuckDuckGo (бесплатный, без ключа).

    Primary-поиск живёт НЕ здесь: logic/tools.py сначала идёт в Gemini
    Google-grounding и зовёт searcher только когда Gemini недоступен.
    Google CSE-ветка снесена 2026-06-11: аккаунт-левел 403 («project does
    not have access»), нелечимо со стороны кода — каждый вызов впустую
    жёг попытку перед DDG.

    Метод `search()` возвращает `(results, source)`, source ∈
    {"duckduckgo", "none"} — вызывающий код предупреждает пользователя
    о менее авторитетном источнике.
    """

    def __init__(self, config=None):
        cfg = config or load_app_config()
        self._ddg_available = False
        try:
            from ddgs import DDGS  # noqa: F401
            self._ddg_available = True
        except ImportError:
            logger.warning("ddgs не установлен — web-поиск fallback недоступен")

    def search(
        self, query: str, top_k: int = 3, region: str = "wt-wt"
    ) -> Tuple[List[Dict[str, str]], str]:
        """region — хинт DDG ("de-de", "ru-ru", …) для локальной выдачи."""
        if not query or not query.strip():
            return [], "none"
        if self._ddg_available:
            results = self._search_ddg(query, top_k, region)
            if results:
                return results, "duckduckgo"
        return [], "none"

    def _search_ddg(self, query: str, top_k: int, region: str = "wt-wt") -> List[Dict[str, str]]:
        try:
            from ddgs import DDGS
            region = (region or "wt-wt").lower()
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, region=region, safesearch="moderate", max_results=top_k))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", "") or r.get("url", ""),
                    "snippet": r.get("body", "") or r.get("snippet", ""),
                }
                for r in raw
            ]
        except Exception:
            logger.exception("DuckDuckGo search error")
            return []
