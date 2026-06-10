"""
Транскрипция голосовых через Groq Whisper API (whisper-large-v3-turbo).

Локальный Whisper на VM невозможен (large-v3 хочет ~3 GB RAM, у Oracle
Free Tier — 1 GB), у Groq та же модель доступна как API на том же free tier.
TG voice = ogg/opus — Groq принимает его напрямую, без перекодирования.
"""

from __future__ import annotations

import logging
from typing import Tuple

import requests

logger = logging.getLogger(__name__)

_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def transcribe_voice(
    data: bytes,
    api_key: str,
    model: str = "whisper-large-v3-turbo",
) -> Tuple[str, str]:
    """Голосовое → текст. Возвращает (text, error); error пуст при успехе."""
    if not data:
        return "", "пустой файл"
    if not api_key:
        return "", "Groq API key не настроен"
    try:
        resp = requests.post(
            _TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("voice.ogg", data)},
            data={"model": model, "temperature": "0", "response_format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        text = (resp.json().get("text") or "").strip()
        if not text:
            return "", "пустая транскрипция (тишина?)"
        return text, ""
    except Exception as e:
        logger.exception("Voice transcription failed")
        return "", str(e)
