"""
Единое распознавание изображений: модели Gemini из пула "vision" по очереди
(utils/llm_gate — у каждой своя дневная квота, исчерпанные пропускаются).

Запасного пути через Groq нет: llama-4-scout (путала пасту-ракушки с
«ракумаки») Groq снял, а других моделей со зрением у него сейчас нет.

Один vision-вызов классифицирует фото и достаёт нужное:
  • shift_schedule — скрин приложения графика смен → структурные смены
  • calendar — скрин календаря/расписания (пары, встречи) → события, дальше
    тот же разбор по видам, что у .ics (logic/calendar_import)
  • food — еда/тарелка/продукты/холодильник → описание для дневника Iris
  • other — что угодно ещё → описание для Redmond

До 01.10.2026 типа calendar не было, а shift_schedule описывался как «скрин
приложения графика смен / календаря»: шесть скринов учебного расписания
CampusNet ушли в смены бара. Пачка фото (альбом) разбирается ОДНИМ запросом —
бесплатная квота Gemini 20 запросов в сутки на модель, шесть скринов съедали
треть.

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


_SHAPE = """Classify "type":
- "shift_schedule": screenshot of the owner's WORK SHIFT planner (bar job): entries
  are his shifts, e.g. titled "Bar" with times like "16:00 – 23:00"; days marked
  "Keine Ereignisse". ONLY work shifts.
- "calendar": any other calendar / timetable screenshot — university timetable
  (CampusNet, Vorlesung, Übung, Seminar, Hörsaal), personal calendar, appointments,
  training. Put EVERY visible entry into "events" with its own title.
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
{{"type": "shift_schedule|calendar|food|other",
  "food_kind": "meal|groceries|receipt",
  "shifts": [{{"date":"YYYY-MM-DD","start":"HH:MM","end":"HH:MM"}}],
  "events": [{{"date":"YYYY-MM-DD","start":"HH:MM","end":"HH:MM","title":"<as shown>",
              "location":"<room/place if shown>"}}],
  "dish": "<for food_kind=meal>",
  "kcal_low": null, "kcal_high": null, "protein_g": null,
  "items": ["<for food_kind=groceries/receipt>"],
  "barcode": "<digits if a barcode is clearly readable, else empty>",
  "description": "<concise Russian description of what is shown; for food list the
   dishes/items; for other describe the scene in one-two sentences>"}}

shift_schedule rules: if a day has both a planned and a checkmarked actual entry,
use the PLANNED one; skip "Keine Ereignisse" days; end "00:00" = midnight.
Dates for shift_schedule and calendar: German dates like "Sonntag, 14. Juni";
current year {year}, current month {month}. Not a shift planner → "shifts": [];
not a calendar → "events": []."""

_PROMPT = "Look at the image and return ONLY a JSON object — no other text.\n\n" + _SHAPE

_BATCH_PROMPT = (
    "You get {n} images. Look at each one separately and return ONLY a JSON array of "
    "{n} objects — one per image, in the order given, no other text. Each object has "
    "the shape below.\n\n" + _SHAPE
)


def _as_int(v: Any) -> Optional[int]:
    """Безопасный каст оценок vision (kcal/protein) в int; None при мусоре/null."""
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _call_gemini_vision(prompt: str, images_b64: List[str], max_tokens: int = 700) -> str:
    """Разбор фото моделями пула "vision" по очереди (utils/llm_gate): у каждой
    Gemini своя дневная квота, исчерпанные шлюз пропускает. '' — не смог никто."""
    from utils import llm_gate
    from utils.gemini import extract_text, generate
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    parts += [{"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in images_b64]
    for model in llm_gate.open_models(llm_gate.pool("vision")):
        text = extract_text(generate(
            parts,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=90.0,
        ))
        if text:
            return text
    return ""


def _failed(why: str, description: str = "") -> Dict[str, Any]:
    return {"type": "other", "shifts": [], "events": [], "description": description,
            "error": why}


def _normalize(data: Any) -> Dict[str, Any]:
    """Ответ модели по одному фото → словарь, на который опираются хендлеры."""
    if not isinstance(data, dict):
        return _failed("bad json")
    result: Dict[str, Any] = {
        "type": str(data.get("type", "other")).strip().lower(),
        "shifts": data.get("shifts") if isinstance(data.get("shifts"), list) else [],
        "events": [e for e in data.get("events") if isinstance(e, dict)]
                  if isinstance(data.get("events"), list) else [],
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
    if result["type"] not in ("shift_schedule", "calendar", "food", "other"):
        result["type"] = "other"
    return result


def _why_failed() -> str:
    from utils import llm_gate
    return "; ".join(llm_gate.describe(llm_gate.pool("vision"))) or "модели не ответили"


def analyze_image(image_b64: str, api_key: str = "") -> Dict[str, Any]:
    """Один разбор фото моделями пула "vision". Возвращает {type, shifts, events,
    description, error}. api_key не используется: запасной путь через Groq
    scout снят вместе с моделью (llama-4-scout отвечает 404 с сентября 2026,
    других моделей со зрением у Groq нет)."""
    now = now_local()
    prompt = _PROMPT.format(year=now.year, month=now.strftime("%B"))

    content = _call_gemini_vision(prompt, [image_b64])
    if not content:
        why = _why_failed()
        logger.warning("Vision: разбор не удался — %s", why)
        return _failed(why)
    logger.info("Vision: разобрано")

    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return _failed("no json", content[:200].strip())
    try:
        return _normalize(json.loads(m.group(0)))
    except json.JSONDecodeError:
        return _failed("bad json")


def analyze_images(images_b64: List[str]) -> List[Dict[str, Any]]:
    """Пачка фото (альбом) — один запрос на все. Ответ не той длины или не
    массив — разбираем по одному: потерять фото хуже, чем потратить запросы."""
    if len(images_b64) <= 1:
        return [analyze_image(b) for b in images_b64]
    now = now_local()
    prompt = _BATCH_PROMPT.format(n=len(images_b64), year=now.year, month=now.strftime("%B"))
    content = _call_gemini_vision(prompt, images_b64,
                                  max_tokens=min(700 * len(images_b64), 6000))
    if not content:
        why = _why_failed()
        logger.warning("Vision: пачка из %d не разобрана — %s", len(images_b64), why)
        return [_failed(why) for _ in images_b64]
    m = re.search(r"\[.*\]", content, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list) or len(data) != len(images_b64):
        logger.warning("Vision: ответ на пачку из %d не той формы (%s) — разбираю по одному",
                       len(images_b64), type(data).__name__ if data is not None else "не JSON")
        return [analyze_image(b) for b in images_b64]
    logger.info("Vision: пачка из %d разобрана одним запросом", len(images_b64))
    return [_normalize(d) for d in data]
