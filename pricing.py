"""Minimal static price lookup for savings + budget-gate forward estimate.

Replaces ``opensquilla.engine.pricing.lookup_price``. The project maintains a
full live model catalog with per-provider pricing; the standalone package ships
a small static fallback table and lets the operator declare per-model prices
via ``[squilla_router.prices]`` (USD per 1M input tokens). Unknown models
return a zero price (treated as free/local) — the budget gate then treats the
forward estimate as a known-zero cost, never acting on ignorance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Price:
    input_per_m: float = 0.0
    output_per_m: float = 0.0


# Static fallback: a few common models. Operator overrides via config extend this.
_FALLBACK_PRICES: dict[str, Price] = {
    "claude-haiku-4-5-20251001": Price(1.0, 5.0),
    "claude-sonnet-4-6": Price(3.0, 15.0),
    "claude-opus-4-8": Price(15.0, 75.0),
    "gpt-5.5": Price(5.0, 15.0),
    "gpt-5.4-mini": Price(0.5, 1.5),
    "deepseek-v4-flash": Price(0.27, 1.1),
    "deepseek-v4-pro": Price(2.7, 11.0),
    "kimi-k2.7-code": Price(2.0, 8.0),
    "glm-5.2": Price(2.0, 8.0),
}

# Operator-declared prices; populated via ``set_prices`` at config load time.
_OVERRIDE_PRICES: dict[str, Price] = {}


def set_prices(prices: dict[str, Any] | None) -> None:
    """Replace the operator price table (USD per 1M tokens).

    Accepts ``{model: {"input_per_m": x, "output_per_m": y}}`` or
    ``{model: x}`` (input-only, output defaults to 0).
    """
    global _OVERRIDE_PRICES
    if not prices:
        _OVERRIDE_PRICES = {}
        return
    table: dict[str, Price] = {}
    for model, value in prices.items():
        key = str(model).strip()
        if not key:
            continue
        if isinstance(value, dict):
            table[key] = Price(
                float(value.get("input_per_m", 0.0) or 0.0),
                float(value.get("output_per_m", 0.0) or 0.0),
            )
        elif isinstance(value, (int, float)):
            table[key] = Price(float(value), 0.0)
    _OVERRIDE_PRICES = table


def lookup_price(model: str) -> Price:
    """Return the price for *model* (operator table first, then fallback)."""
    key = str(model or "").strip()
    if key in _OVERRIDE_PRICES:
        return _OVERRIDE_PRICES[key]
    # Case-insensitive fallback match.
    lower = key.lower()
    for fb_key, price in _FALLBACK_PRICES.items():
        if fb_key.lower() == lower:
            return price
    return Price()
