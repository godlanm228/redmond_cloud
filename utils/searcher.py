import logging
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import requests

from config.config_loader import load_app_config

logger = logging.getLogger(__name__)


class GoogleSearchLimitExceeded(Exception):
    """Google Custom Search дневной лимит исчерпан."""


class WebSearcher:
    """
    Поиск в интернете с двумя источниками:
    - Google Custom Search (основной, требует ключ + cx, ~100 запросов/день)
    - DuckDuckGo (fallback, бесплатный без ключа)

    Метод `search()` возвращает `(results, source)`, где source — "google"
    или "duckduckgo" — чтобы вызывающий код мог сообщить пользователю,
    что данные пришли из менее авторитетного источника.
    """

    GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, config=None):
        cfg = config or load_app_config()
        self.api_key = cfg.google_api_key
        self.cx = cfg.google_search_engine_id
        self.daily_limit = cfg.google_search_daily_limit
        self.timeout = cfg.search_timeout_sec
        self.db_path = "data/search_usage.sqlite"
        self._init_db()

        self._ddg = None
        self._try_init_ddg()

    def _try_init_ddg(self) -> None:
        try:
            from ddgs import DDGS  # noqa: F401
            self._ddg_available = True
        except ImportError:
            logger.warning("ddgs не установлен — DuckDuckGo fallback недоступен")
            self._ddg_available = False

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    day   TEXT PRIMARY KEY,
                    count INT NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _get_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("SELECT count FROM usage WHERE day=?", (self._today(),)).fetchone()
            return cur[0] if cur else 0
        finally:
            conn.close()

    def _increment(self) -> None:
        today = self._today()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO usage(day,count) VALUES(?,1) "
                "ON CONFLICT(day) DO UPDATE SET count = count + 1",
                (today,),
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- публичный API ----------

    def search(self, query: str, top_k: int = 3) -> Tuple[List[Dict[str, str]], str]:
        """
        Поиск с автоматическим fallback. Возвращает (results, source).
        source ∈ {"google", "duckduckgo", "none"}.
        """
        if not query or not query.strip():
            return [], "none"

        # 1. Google — если есть ключ и cx, лимит не превышен
        if self.api_key and self.cx:
            if self._get_count() >= self.daily_limit:
                logger.warning("Google daily limit exceeded — переходим на DuckDuckGo")
            else:
                google_results = self._search_google(query, top_k)
                if google_results is not None:
                    return google_results, "google"

        # 2. DuckDuckGo fallback
        if self._ddg_available:
            ddg_results = self._search_ddg(query, top_k)
            if ddg_results:
                return ddg_results, "duckduckgo"

        return [], "none"

    # ---------- провайдеры ----------

    def _search_google(self, query: str, top_k: int) -> Optional[List[Dict[str, str]]]:
        """Возвращает None если Google недоступен (тогда сработает fallback)."""
        params = {"key": self.api_key, "cx": self.cx, "q": query, "num": top_k}
        try:
            resp = requests.get(self.GOOGLE_ENDPOINT, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("items", [])
            self._increment()
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in data
            ]
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 403):
                logger.warning("Google search %s — switching to fallback", status)
                return None
            logger.exception("Google search HTTP error")
            return None
        except Exception:
            logger.exception("Google search error")
            return None

    def _search_ddg(self, query: str, top_k: int) -> List[Dict[str, str]]:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=top_k))
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
