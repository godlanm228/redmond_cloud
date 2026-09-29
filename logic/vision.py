"""
Единое распознавание изображений: модели Gemini из пула "vision" по очереди
(utils/llm_gate — у каждой своя дневная квота, исчерпанные пропускаются).

Запасного пути через Groq нет: llama-4-scout (путала пасту-ракушки с
«ракумаки») Groq снял, а других моделей со зрением у него сейчас нет.

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
import re
from typing import Any, Dict, List, Optional


from utils.time import now_local

logger = logging.getLogger(__name__)


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
    """Разбор фото моделями пула "vision" по очереди (utils/llm_gate): у каждой
    Gemini своя дневная квота, исчерпанные шлюз пропускает. '' — не смог никто."""
    from utils import llm_gate
    from utils.gemini import extract_text, generate
    for model in llm_gate.open_models(llm_gate.pool("vision")):
        text = extract_text(generate(
            [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            ],
            model=model,
            temperature=0.0,
            max_tokens=700,
            timeout=60.0,
        ))
        if text:
            return text
    return ""


def analyze_image(image_b64: str, api_key: str) -> Dict[str, Any]:
    """Один разбор фото моделями пула "vision". Возвращает {type, shifts,
    description, error}. api_key не используется: запасной путь через Groq
    scout снят вместе с моделью (llama-4-scout отвечает 404 с сентября 2026,
    других моделей со зрением у Groq нет)."""
    now = now_local()
    prompt = _PROMPT.format(year=now.year, month=now.strftime("%B"))

    content = _call_gemini_vision(prompt, image_b64)
    if not content:
        from utils import llm_gate
        why = "; ".join(llm_gate.describe(llm_gate.pool("vision"))) or "модели не ответили"
        logger.warning("Vision: разбор не удался — %s", why)
        return {"type": "other", "shifts": [], "description": "", "error": why}
    logger.info("Vision: разобрано")

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
