"""Standalone prompt cache injection — mark system prompt for Anthropic prompt caching.

Pure function: takes a system prompt and optional dynamic suffix, returns
metrics (hash, char count) for observability. No I/O, no dependencies
beyond stdlib.

Usage:

    from squilla_router_standalone.prompt_cache import cache_system_prompt

    metadata = cache_system_prompt(base_prompt, dynamic_prompt="")
    # metadata: {"cache_base_chars": ..., "cache_base_hash": ..., "cache_enabled": True}
"""

from __future__ import annotations

import hashlib
from typing import Any


def _hash16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_system_prompt(
    base_prompt: str,
    dynamic_prompt: str = "",
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Annotate system prompt for Anthropic prompt caching.

    Returns a metadata dict with cache metrics. When ``enabled`` is False,
    returns ``{"cache_enabled": False}`` — a no-op so the caller's path is
    unchanged.

    The ``base_prompt`` is the static part of the system prompt (tools,
    identity, rules); ``dynamic_prompt`` is the per-turn suffix (context,
    instructions). Cache breakpoint injection (``[CACHE_BREAK]``) is left to
    the caller — this function only records the metrics.
    """
    if not enabled:
        return {"cache_enabled": False}

    meta: dict[str, Any] = {
        "cache_enabled": True,
        "cache_base_chars": len(base_prompt),
        "cache_base_hash": _hash16(base_prompt),
    }
    if dynamic_prompt:
        meta["cache_dynamic_chars"] = len(dynamic_prompt)
        meta["cache_dynamic_hash"] = _hash16(dynamic_prompt)

    return meta