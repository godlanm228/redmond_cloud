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


def lookup_barcode(barcode: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    digits = "".join(ch for ch in str(barcode) if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        r = requests.get(
            f"{_BASE}/api/v2/product/{digits}.json",
            params={"fields": _FIELDS}, headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        d = r.json()
        if d.get("status") == 1 or d.get("product"):
            return _parse(d.get("product") or {})
    except Exception as e:
        logger.debug("OFF barcode lookup failed (%s): %s", barcode, e)
    return None


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
            return _parse(products[0])
    except Exception as e:
        logger.debug("OFF name search failed (%s): %s", query, e)
    return None


def lookup(barcode: str = "", name: str = "") -> Optional[Dict[str, Any]]:
    """Штрихкод (точно) → название (fallback). None если ничего не нашли."""
    res = lookup_barcode(barcode) if barcode else None
    if not res and name:
        res = search_name(name)
    return res
