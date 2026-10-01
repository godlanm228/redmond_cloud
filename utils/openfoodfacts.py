"""
OpenFoodFacts — бесплатная открытая база продуктов (БЕЗ ключа). Точная нутриция
по штрихкоду/названию для ПАКЕТИРОВАННОЙ еды: реальные числа вместо оценки на глаз.
None при промахе → вызывающий честно падает на оценку (см. logic/tools.lookup_food).

Endpoints:
  • штрихкод: /api/v2/product/{barcode}.json  (надёжный путь)
  • название: /cgi/search.pl                   (fallback, менее точен)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://world.openfoodfacts.org"
_UA = {"User-Agent": "RedmondHub/1.0 (personal assistant; contact vlad)"}
_FIELDS = "product_name,brands,nutriments,quantity,serving_size"


def _parse(product: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not product:
        return None
    n = product.get("nutriments") or {}
    kcal = n.get("energy-kcal_100g")
    prot = n.get("proteins_100g")
    name = (product.get("product_name") or "").strip()
    if not name and kcal is None:
        return None
    return {
        "name": name,
        "brands": (product.get("brands") or "").strip(),
        "kcal_100g": round(kcal) if isinstance(kcal, (int, float)) else None,
        "protein_100g": round(prot, 1) if isinstance(prot, (int, float)) else None,
        "quantity": (product.get("quantity") or "").strip(),
        "serving_size": (product.get("serving_size") or "").strip(),
    }


def _why(e: Exception) -> str:
    """Короткая причина сбоя для владельца: «HTTP 503», «таймаут», «нет связи»."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status:
        return f"HTTP {status}"
    name = type(e).__name__
    if "Timeout" in name:
        return "таймаут"
    if "Connection" in name:
        return "нет связи"
    return name


def _found(product: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = _parse(product or {})
    return {**parsed, "found": True} if parsed else {"found": False}


def lookup_barcode(barcode: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    digits = "".join(ch for ch in str(barcode) if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        r = requests.get(
            f"{_BASE}/api/v2/product/{digits}.json",
            params={"fields": _FIELDS}, headers=_UA, timeout=timeout,
        )
        if r.status_code == 404:
            return {"found": False}  # штрихкода нет в базе — это ответ, не сбой
        r.raise_for_status()
        d = r.json()
        if d.get("status") == 1 or d.get("product"):
            return _found(d.get("product"))
        return {"found": False}
    except Exception as e:  # noqa: BLE001 — сервис лёг ≠ продукта нет
        logger.warning("OFF barcode lookup failed (%s): %s", barcode, e)
        return {"found": False, "error": _why(e)}


def search_name(query: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return None
    try:
        r = requests.get(
            f"{_BASE}/cgi/search.pl",
            params={
                "search_terms": query, "json": 1, "page_size": 1,
                "fields": _FIELDS, "sort_by": "popularity_key",
            },
            headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        products = r.json().get("products") or []
        if products:
            return _found(products[0])
        return {"found": False}
    except Exception as e:  # noqa: BLE001 — сервис лёг ≠ продукта нет
        logger.warning("OFF name search failed (%s): %s", query, e)
        return {"found": False, "error": _why(e)}


def lookup(barcode: str = "", name: str = "") -> Optional[Dict[str, Any]]:
    """Штрихкод (точно) → название (запасной путь).

    Три разных ответа, а не два: {"found": True, …} — нашли; {"found": False} —
    в базе нет; {"found": False, "error": "HTTP 503"} — сервис не ответил.
    01.10.2026 OpenFoodFacts лежал (503), а бот говорил «продукт не найден».
    None — искать было нечем."""
    results = []
    if barcode:
        results.append(lookup_barcode(barcode))
    if name and not (results and results[-1] and results[-1].get("found")):
        results.append(search_name(name))
    results = [r for r in results if r]
    for r in results:
        if r.get("found"):
            return r
    errors = [r["error"] for r in results if r.get("error")]
    if errors:
        return {"found": False, "error": "; ".join(errors)}
    return {"found": False} if results else None
