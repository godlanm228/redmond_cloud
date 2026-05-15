"""
Автонаполнение `owner_profile.known_facts` фактами из диалога.

После каждого пользовательского сообщения вызывается `extract_facts()` —
LLM (Groq) смотрит на пару (user, assistant) и возвращает 0–N фактов о
владельце в формате коротких утверждений. Факты дедуплицируются и
сохраняются обратно в owner_profile.json.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

import requests

from config.config_loader import save_owner_profile

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


EXTRACTION_SYSTEM_PROMPT = (
    "Ты извлекаешь факты о пользователе из его реплики ассистенту. "
    "Анализируй ТОЛЬКО пользовательский текст; ассистент тебе нужен лишь как контекст. "
    "Верни JSON-массив строк — каждая строка это короткое утверждение о пользователе в третьем лице.\n"
    "\n"
    "ИЗВЛЕКАЙ:\n"
    "  • имя, возраст, профессию, город, язык\n"
    "  • проекты, на которых работает\n"
    "  • предпочтения, привычки, ценности\n"
    "  • важных людей (семья, коллеги — только если упомянуты явно)\n"
    "  • цели, планы, договорённости\n"
    "  • технические факты о его системе/инструментах\n"
    "\n"
    "НЕ ИЗВЛЕКАЙ:\n"
    "  • факты о ассистенте, мире, общие знания\n"
    "  • временные состояния («сейчас зол», «спрашивает погоду»)\n"
    "  • вопросы пользователя\n"
    "\n"
    "Если фактов нет — верни `[]`. Если есть — массив максимум из 5 строк.\n"
    "ОТВЕЧАЙ ТОЛЬКО JSON-массивом, без обёрток, без markdown."
)


class FactExtractor:
    """Извлекает факты о владельце из диалога и обновляет owner_profile."""

    MAX_FACTS = 50  # потолок размера профиля

    def __init__(self, config, owner_profile: Dict[str, Any]):
        self.config = config
        self.owner_profile = owner_profile
        self.profile_path = getattr(config, "owner_profile", "config/owner_profile.json")
        self._lock = threading.Lock()

    def extract_async(self, user_text: str, assistant_text: str) -> None:
        """Запустить извлечение в фоновом треде, чтобы не блокировать TTS."""
        if not user_text or not user_text.strip():
            return
        t = threading.Thread(
            target=self._extract_and_persist,
            args=(user_text, assistant_text),
            daemon=True,
        )
        t.start()

    def _extract_and_persist(self, user_text: str, assistant_text: str) -> None:
        try:
            facts = self._extract(user_text, assistant_text)
            if not facts:
                return
            self._merge(facts)
        except Exception:
            logger.exception("Fact extraction failed")

    def _extract(self, user_text: str, assistant_text: str) -> List[str]:
        api_key = getattr(self.config, "groq_api_key", "")
        if not api_key:
            return []

        model = getattr(self.config, "groq_model", "llama-3.3-70b-versatile")
        user_msg = (
            f"Реплика пользователя: «{user_text}»\n"
            f"Ответ ассистента (контекст): «{assistant_text[:300]}»"
        )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw = self._groq_call(api_key, model, messages)
        if not raw:
            return []

        return self._parse_facts(raw)

    def _groq_call(self, api_key: str, model: str, messages: list) -> Optional[str]:
        try:
            if GROQ_SDK_AVAILABLE:
                client = Groq(api_key=api_key)
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,  # детерминированно — нам нужен JSON
                    max_tokens=300,
                )
                return completion.choices[0].message.content

            base = getattr(self.config, "groq_api_base", "https://api.groq.com").rstrip("/")
            resp = requests.post(
                f"{base}/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 300,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug("Fact extractor Groq call failed: %s", e)
            return None

    @staticmethod
    def _parse_facts(raw: str) -> List[str]:
        s = raw.strip()
        # Снимаем markdown-обёртки на случай если модель не послушалась
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
            if s.endswith("```"):
                s = s[:-3].strip()

        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            logger.debug("Fact extractor returned non-JSON: %r", raw[:200])
            return []

        if not isinstance(data, list):
            return []
        return [str(f).strip() for f in data if str(f).strip()]

    def _merge(self, new_facts: List[str]) -> None:
        """Добавить новые факты в профиль, удалив дубликаты и обрезав до лимита."""
        with self._lock:
            existing = list(self.owner_profile.get("known_facts") or [])
            existing_lower = {f.lower() for f in existing}

            added = []
            for fact in new_facts:
                if fact.lower() in existing_lower:
                    continue
                existing.append(fact)
                existing_lower.add(fact.lower())
                added.append(fact)

            if not added:
                return

            # Обрезаем до лимита, удаляя самые старые
            if len(existing) > self.MAX_FACTS:
                existing = existing[-self.MAX_FACTS:]

            self.owner_profile["known_facts"] = existing

            try:
                save_owner_profile(self.owner_profile, self.profile_path)
                logger.info("Memory: added %d facts about owner", len(added))
                for f in added:
                    logger.debug("  + %s", f)
            except Exception:
                logger.exception("Failed to persist owner profile")
