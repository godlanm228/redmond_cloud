"""
Недельное расписание Влада: рабочие смены (из скрина графика через Groq vision)
+ статичное учебное расписание (SoSe 2026, HRW Campus Bottrop).

Смены: data/coach/shifts.json — {"YYYY-MM-DD": {"start": "HH:MM", "end": "HH:MM"}}.
Учёба: константа в коде (меняется раз в семестр — проще править тут, чем JSON).

Vision-парсинг: llama-4-scout на Groq (free tier, понимает скрины German UI).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

from logic.coach_storage import _load_json, _save_json  # тот же data/coach каталог
from utils.time import now_local

logger = logging.getLogger(__name__)

# Дорога (мин): работа — самокат от дома; универ — дом → Essen Hbf → SB16 (~34 мин) → кампус.
WORK_COMMUTE_MIN = 20
UNI_COMMUTE_MIN = 60

# Учебное расписание: weekday (0=пн) → [(start, end, что)]. Ботроп = дорога UNI_COMMUTE_MIN.
# Среда: туториум 13:15-14:50 Влад скипает осознанно — вместо него домашний блок.
STUDY_TIMETABLE: Dict[int, List[Tuple[str, str, str]]] = {
    0: [("14:05", "15:45", "Лекция Grundlagen der Ingenieurmathematik (Ботроп)")],
    1: [
        ("12:20", "14:00", "Лекция Ingenieurmathematik (Ботроп)"),
        ("14:05", "15:45", "Практика Ingenieurmathematik (Ботроп)"),
    ],
    2: [("13:15", "14:50", "Домашняя учёба (туториум скипается в пользу дома)")],
    4: [
        ("08:00", "09:35", "Лекция Projektmanagement (Ботроп)"),
        ("11:30", "13:05", "Практика Projektmanagement (Ботроп)"),
    ],
}

# Дни без пар и обычно без смен — кандидаты на «день на себя» (CS/Dota/друзья).
HOME_STUDY_WEEKDAYS = (2, 3)  # ср (вместо туториума), чт (пар нет)

_VISION_MODEL = os.getenv("REDMOND_GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


# ---------- смены: storage ----------

def get_shift(d: date) -> Optional[Dict[str, str]]:
    shifts = _load_json("shifts.json", {})
    return shifts.get(d.strftime("%Y-%m-%d"))


def save_shifts(items: List[Dict[str, str]]) -> int:
    """Merge смен в shifts.json (date → start/end). Возвращает сколько сохранено."""
    shifts = _load_json("shifts.json", {})
    n = 0
    for it in items:
        d, start, end = it.get("date"), it.get("start"), it.get("end")
        if not (d and start and end):
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        shifts[d] = {"start": start, "end": end}
        n += 1
    if n:
        _save_json("shifts.json", shifts)
    return n


def format_week(days: int = 8) -> str:
    """Человекочитаемое расписание на N дней вперёд: смены + пары. Для tool/пингов."""
    out: List[str] = []
    today = now_local().date()
    day_names = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    for i in range(days):
        d = today + timedelta(days=i)
        parts: List[str] = []
        shift = get_shift(d)
        if shift:
            parts.append(f"смена {shift['start']}–{shift['end']} (дорога ~{WORK_COMMUTE_MIN} мин)")
        for start, end, what in STUDY_TIMETABLE.get(d.weekday(), []):
            parts.append(f"{start}–{end} {what}")
        label = f"{day_names[d.weekday()]} {d.strftime('%d.%m')}"
        out.append(f"{label}: " + ("; ".join(parts) if parts else "свободен"))
    return "\n".join(out)


# ---------- смены: vision-парсинг скрина ----------

_VISION_PROMPT = """Extract WORK SHIFTS from this screenshot of a German shift-planning app.

Rules:
- Entries titled "Bar" with times like "16:00 – 23:00" are shifts.
- If a day has both a planned entry and a checkmarked actual entry (e.g. "17:14 – 22:59"), use the PLANNED one.
- "Keine Ereignisse" = day off, skip it.
- Dates look like "Sonntag, 14. Juni" (German). Current year: {year}. Current month: {month}.
- End time "00:00" means midnight.

Return ONLY a JSON array, no other text:
[{{"date":"YYYY-MM-DD","start":"HH:MM","end":"HH:MM"}}]"""


def parse_shift_screenshot(image_b64: str, api_key: str) -> Tuple[List[Dict[str, str]], str]:
    """
    Скрин → список смен через Groq vision. Возвращает (items, error).
    items пуст + error заполнен = не разобрали.
    """
    if not api_key:
        return [], "Groq API key не настроен."
    now = now_local()
    prompt = _VISION_PROMPT.format(year=now.year, month=now.strftime("%B"))
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": _VISION_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        }},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 600,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"] or ""
    except Exception as e:
        logger.exception("Vision call failed")
        return [], f"Vision-модель недоступна: {e}"

    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        return [], "Не нашёл JSON в ответе vision-модели."
    try:
        items = json.loads(m.group(0))
        if not isinstance(items, list):
            raise ValueError("not a list")
    except Exception:
        return [], "Vision-модель вернула битый JSON."
    return items, ""


def ingest_shift_screenshot(image_b64: str, api_key: str) -> str:
    """Полный путь: скрин → парс → сохранить → человекочитаемый ответ для чата."""
    items, err = parse_shift_screenshot(image_b64, api_key)
    if err:
        return f"Скрин не разобрал ({err}). Кинь смены текстом — «чт 14:00-23:00, пт 18:00-00:00…»"
    if not items:
        return "На скрине смен не увидел. Если они там есть — кинь текстом."

    n = save_shifts(items)
    day_names = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    lines = []
    for it in sorted(items, key=lambda x: x.get("date", "")):
        try:
            d = datetime.strptime(it["date"], "%Y-%m-%d").date()
            lines.append(f"• {day_names[d.weekday()]} {d.strftime('%d.%m')}: {it['start']}–{it['end']}")
        except (KeyError, ValueError):
            continue
    return (
        f"Принял график, смен сохранено: {n}\n\n" + "\n".join(lines) +
        "\n\nЕсли что-то распознал криво — поправь текстом."
    )
