"""
Единое распознавание изображений: Gemini 2.5 Flash (primary) с fallback
на Groq vision (llama-4-scout, free tier).

Scout — слабейшая vision-модель Groq: путала пасту-ракушки с «ракумаки»,
немецкую колбаску определяла как «кусок колбасы». Gemini распознаёт
радикально лучше и заодно не тратит Groq-лимиты. Scout остаётся аварийным
запасным, если Gemini недоступен.

Один vision-вызов классифицирует фото и достаёт нужное:
  • shift_schedule — скрин графика смен → структурные смены (как раньше)
  • food — еда/тарелка/продукты/холодильник → описание для дневника Iris
  • other — что угодно ещё → описание для Redmond

Раньше ВСЕ фото слепо парсились как график смен (фото ужина → «смен не увидел»),
а на «что ты видишь?» текстовая модель врала «не могу смотреть картинки».
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from utils.time import now_local

logger = logging.getLogger(__name__)

_VISION_MODEL = os.getenv("REDMOND_GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_PROMPT = """Look at the image and return ONLY a JSON object — no other text.

Classify "type":
- "shift_schedule": screenshot of a shift-planning / calendar app. Entries titled
  "Bar" with times like "16:00 – 23:00"; days marked "Keine Ereignisse".
- "food": a meal/plate/dish, OR groceries / fridge contents / a store receipt.
  Set "food_kind": "meal" | "groceries" | "receipt".
  CLASSIFY CAREFULLY: a sealed package / bag / box with a brand label (held in hand or
  on a surface) = "groceries", EVEN if it contains food like vegetables or noodles.
  Only food served on a plate / in a bowl / pan, ready to eat = "meal".
  meal -> "dish" (Russian, what is on the plate) + HONEST estimates "kcal_low"/"kcal_high"
  (a TIGHT range ~15%, null if truly unsure) and "protein_g" (approx grams, null if unsure);
  portions from a photo are rough — give ranges, never fake precision.
  groceries/receipt -> "items": product names (Russian, short; on a receipt read the line
  items, skip prices and totals). Also set "barcode" if an EAN/barcode is clearly readable.
- "other": anything else (people, places, screenshots that are not schedules, etc).

JSON shape:
{{"type": "shift_schedule|food|other",
  "food_kind": "meal|groceries|receipt",
  "shifts": [{{"date":"YYYY-MM-DD","start":"HH:MM","end":"HH:MM"}}],
  "dish": "<for food_kind=meal>",
  "kcal_low": null, "kcal_high": null, "protein_g": null,
  "items": ["<for food_kind=groceries/receipt>"],
  "barcode": "<digits if a barcode is clearly readable, else empty>",
  "description": "<concise Russian description of what is shown; for food list the
   dishes/items; for other describe the scene in one-two sentences>"}}

shift_schedule rules: if a day has both a planned and a checkmarked actual entry,
use the PLANNED one; skip "Keine Ereignisse" days; German dates like
"Sonntag, 14. Juni"; current year {year}, current month {month}; end "00:00" = midnight.
If the image is NOT a schedule, "shifts" MUST be []."""


def _as_int(v: Any) -> Optional[int]:
    """Безопасный каст оценок vision (kcal/protein) в int; None при мусоре/null."""
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _call_gemini_vision(prompt: str, image_b64: str) -> str:
    """Gemini 2.5 Flash vision. '' при любой ошибке (уйдём на Groq scout)."""
    from utils.gemini import extract_text, generate
    return extract_text(generate(
        [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
        ],
        temperature=0.0,
        max_tokens=700,
        timeout=60.0,
    ))


def _call_groq_vision(prompt: str, image_b64: str, api_key: str) -> Dict[str, str]:
    """Groq scout (fallback). Возвращает {content, error}."""
    if not api_key:
        return {"content": "", "error": "no api key"}
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
                "max_tokens": 700,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return {"content": resp.json()["choices"][0]["message"]["content"] or "", "error": ""}
    except Exception as e:
        logger.exception("Groq vision call failed")
        return {"content": "", "error": str(e)}


def analyze_image(image_b64: str, api_key: str) -> Dict[str, Any]:
    """Один vision-вызов: Gemini → fallback Groq scout (api_key — ключ Groq).
    Возвращает {type, shifts, description, error}."""
    now = now_local()
    prompt = _PROMPT.format(year=now.year, month=now.strftime("%B"))

    content = _call_gemini_vision(prompt, image_b64)
    if content:
        logger.info("Vision: Gemini")
    else:
        logger.warning("Gemini vision недоступен — fallback на Groq scout")
        groq = _call_groq_vision(prompt, image_b64, api_key)
        content = groq["content"]
        if not content:
            return {"type": "other", "shifts": [], "description": "", "error": groq["error"] or "empty"}

    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return {"type": "other", "shifts": [], "description": content[:200].strip(), "error": "no json"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"type": "other", "shifts": [], "description": "", "error": "bad json"}

    result: Dict[str, Any] = {
        "type": str(data.get("type", "other")).strip().lower(),
        "shifts": data.get("shifts") if isinstance(data.get("shifts"), list) else [],
        "description": str(data.get("description", "")).strip(),
        "food_kind": str(data.get("food_kind", "")).strip().lower(),
        "dish": str(data.get("dish", "")).strip(),
        "barcode": "".join(ch for ch in str(data.get("barcode", "")) if ch.isdigit()),
        "items": [str(x).strip() for x in data.get("items") if str(x).strip()]
                 if isinstance(data.get("items"), list) else [],
        "kcal_low": _as_int(data.get("kcal_low")),
        "kcal_high": _as_int(data.get("kcal_high")),
        "protein_g": _as_int(data.get("protein_g")),
        "error": "",
    }
    if result["type"] not in ("shift_schedule", "food", "other"):
        result["type"] = "other"
    return result
