"""Per-session routing history with bounded size and eviction.

Verbatim port of the ``RoutingHistoryStore`` + history-append helpers in
``opensquilla.engine.steps.squilla_router``. The history entry schema is the
critical one for V4 ML: it must carry ``text`` + the full ``extra`` routing
trail (so the V4 adapter can read ``history_user_texts`` and
``prev_route_decisions`` on turn 2+), not just tier labels.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_MAX_ROUTING_HISTORY = 5
_ROUTING_HISTORY_WINDOW = 1800  # seconds


class RoutingHistoryStore:
    """Per-session routing history with bounded size and eviction."""

    def __init__(self, max_entries: int = _MAX_ROUTING_HISTORY) -> None:
        self._entries: dict[str, list[dict]] = {}
        self._max_entries = max_entries

    def get(self, session_key: str) -> list[dict] | None:
        return self._entries.get(session_key)

    def set(self, session_key: str, value: list[dict]) -> None:
        self._entries[session_key] = value

    def setdefault(self, session_key: str, default: list[dict]) -> list[dict]:
        return self._entries.setdefault(session_key, default)

    def length(self, session_key: str) -> int:
        return len(self._entries.get(session_key, []))

    def clear(self) -> None:
        self._entries.clear()

    def evict(self, session_key: str) -> bool:
        return self._entries.pop(session_key, None) is not None


def _routing_history_entry(
    *,
    text: str,
    extra: dict,
    decision: Any,
) -> dict:
    """Build a routing-history entry carrying text + full extra trail."""
    return {
        "text": text,
        **extra,
        "base_tier": extra.get("base_tier", decision.tier),
        "final_tier": extra.get("final_tier", decision.tier),
        "final_route_class": extra.get("final_route_class"),
    }


def append_routing_history(
    store: RoutingHistoryStore,
    session_key: str,
    entry_payload: dict,
) -> list[dict]:
    """Append a routing-history entry, bounded to the store's max size."""
    history = store.setdefault(session_key, [])
    entry = {
        "turn_index": len(history),
        "_ts": time.monotonic(),
        **entry_payload,
    }
    history.append(entry)
    if len(history) > store._max_entries:  # noqa: SLF001
        trimmed = history[-store._max_entries :]  # noqa: SLF001
        store.set(session_key, trimmed)
        history = trimmed
    log.debug(
        "squilla_router.history_appended",
        session=session_key,
        turn_index=entry["turn_index"],
        route_class=entry.get("route_class"),
        total_history=store.length(session_key),
    )
    return store.get(session_key) or []


def seed_routing_history(
    store: RoutingHistoryStore,
    entries_by_session: dict[str, list[dict]],
) -> int:
    """Seed the in-process history store from persisted decision records.

    Boot-time rehydration hook: sessions that already accumulated live
    in-process history are never clobbered. Returns the number of sessions seeded.
    """
    seeded = 0
    for session_key, entries in entries_by_session.items():
        if not session_key or not entries:
            continue
        if store.get(session_key):
            continue
        store.set(
            session_key,
            [dict(entry) for entry in entries][-store._max_entries :],  # noqa: SLF001
        )
        seeded += 1
    return seeded


def windowed_history(
    history: list[dict] | None,
    *,
    now: float | None = None,
    window: float = _ROUTING_HISTORY_WINDOW,
    max_entries: int = _MAX_ROUTING_HISTORY,
) -> list[dict]:
    """Apply the age window + size cap to a loaded history list."""
    if not history:
        return []
    cutoff = (now if now is not None else time.monotonic()) - window
    filtered = [e for e in history if e.get("_ts", 0) > cutoff]
    return filtered[-max_entries:]
