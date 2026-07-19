"""File-based router decision records (replaces the SQLite writer).

Ports ``opensquilla.engine.steps.router_decision_record``: the trail builder
(``build_trail``), the stage/flush lifecycle, and boot-time rehydration — but
the writer is a per-session JSONL file under the state dir instead of a
SQLite table. Records carry tier/route-class/model tokens, numbers, and
booleans only; free text (complaint terms, prompt hints, errors) never enters
a record (the trail whitelist enforces this).

With no writer registered (the default unless the caller calls
``set_decision_writer``), every function here is a no-op — the router step's
behavior is unchanged. Calibration aggregation (``aggregate_calibration``)
can consume these JSONL records offline.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import structlog

from squilla_router_standalone.engine.routing.calibration import state_dir
from squilla_router_standalone.engine.routing.policy import route_class_for_tier
from squilla_router_standalone.router_tiers import normalize_text_tier

log = structlog.get_logger(__name__)

PENDING_RECORD_KEY = "_pending_router_decision_record"
DECISION_ID_METADATA_KEY = "router_decision_id"
FALLBACK_HOPS_METADATA_KEY = "router_fallback_hops"

# Safe-token charset for persisted tokens (tier/route_class/model/provider).
_SAFE = set("abcdefghijklmnopqrstuvwxyz0123456789._-:/")


def sanitize_token(value: object) -> str | None:
    """Lowercase + charset-clamp a token; reject non-strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if all(c in _SAFE for c in text):
        return text
    clamped = "".join(c if c in _SAFE else "_" for c in text)
    return clamped or None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    if as_float != as_float or as_float in (float("inf"), float("-inf")):
        return None
    return as_float


def _trail_entry(stage: str, **fields: object) -> dict[str, Any]:
    entry: dict[str, Any] = {"stage": stage}
    for key, value in fields.items():
        if value is None:
            continue
        entry[key] = value
    return entry


def build_trail(extra: Mapping[str, Any], *, final_tier: str | None) -> list[dict[str, Any]]:
    """Rebuild the policy-stage trail from ``routing_extra`` as safe entries."""
    trail: list[dict[str, Any]] = []
    base_tier = sanitize_token(extra.get("base_tier"))
    if base_tier is not None:
        trail.append(
            _trail_entry(
                "classify",
                tier=base_tier,
                route_class=sanitize_token(extra.get("route_class")),
            )
        )
    if "confidence_gate_applied" in extra:
        trail.append(
            _trail_entry(
                "confidence_gate",
                applied=bool(extra.get("confidence_gate_applied")),
                threshold=_number(extra.get("confidence_threshold")),
                default_tier=sanitize_token(extra.get("confidence_default_tier")),
            )
        )
    if "complaint_upgrade_applied" in extra:
        terms = extra.get("complaint_terms")
        trail.append(
            _trail_entry(
                "complaint_upgrade",
                applied=bool(extra.get("complaint_upgrade_applied")),
                terms_count=len(terms) if isinstance(terms, (list, tuple)) else 0,
            )
        )
    if "anti_downgrade_applied" in extra:
        trail.append(
            _trail_entry(
                "anti_downgrade",
                applied=bool(extra.get("anti_downgrade_applied")),
                previous_tier=sanitize_token(extra.get("previous_tier")),
                window_seconds=_number(extra.get("kv_cache_window_seconds")),
            )
        )
    if extra.get("capability_gate_applied"):
        trail.append(_trail_entry("capability_gate", applied=True))
    if extra.get("large_context_floor_applied"):
        trail.append(
            _trail_entry(
                "large_context_floor",
                applied=True,
                from_tier=sanitize_token(extra.get("large_context_floor_from_tier")),
                min_tier=sanitize_token(extra.get("large_context_floor_min_tier")),
                material_tokens=_number(extra.get("large_context_material_tokens")),
            )
        )
    if extra.get("budget_gate_applied"):
        trail.append(
            _trail_entry(
                "budget_gate",
                applied=True,
                rule=sanitize_token(extra.get("budget_gate_outcome")),
            )
        )
    final_token = sanitize_token(extra.get("final_tier")) or sanitize_token(final_tier)
    if final_token is not None:
        trail.append(
            _trail_entry(
                "final",
                tier=final_token,
                route_class=sanitize_token(extra.get("final_route_class")),
            )
        )
    return trail


class FileDecisionWriter:
    """Per-session JSONL router decision writer (replaces the SQLite writer).

    One JSONL file per session under ``<state_dir>/router_decisions/``.
    Best-effort: never raises into the turn loop. Each record is one line.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self._root = Path(root) if root is not None else state_dir("router_decisions")

    def _session_path(self, session_key: str) -> Path:
        # Sanitize the session key into a safe filename stem.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_key))[:128]
        if not safe:
            safe = "session"
        return self._root / f"{safe}.jsonl"

    def record_decision(self, record: dict[str, Any]) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            path = self._session_path(str(record.get("session_key") or "session"))
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 — decision records must never fail a turn
            log.warning("router_decision_record.write_failed", exc_info=True)

    def load_recent_history(
        self,
        *,
        window_seconds: int = 1800,
        per_session: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Load recent decision rows grouped by session (mirror of SQLite writer)."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        if not self._root.exists():
            return grouped
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_seconds * 1000
        try:
            for path in self._root.glob("*.jsonl"):
                rows: list[dict[str, Any]] = []
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            rows.append(json.loads(line))
                        except (ValueError, TypeError):
                            continue
                except OSError:
                    continue
                recent = [r for r in rows if int(r.get("ts_ms") or 0) >= cutoff]
                if recent:
                    grouped[path.stem] = recent[-per_session:]
        except Exception:  # noqa: BLE001 — rehydration must never block boot
            log.warning("router_decision_record.load_failed", exc_info=True)
        return grouped


_decision_writer: FileDecisionWriter | None = None


def set_decision_writer(writer: FileDecisionWriter | None) -> None:
    """Install (or, with ``None``, clear) the process-wide decision writer."""
    global _decision_writer
    _decision_writer = writer


def get_decision_writer() -> FileDecisionWriter | None:
    return _decision_writer


def stage_router_decision(
    ctx: Any,
    *,
    decision: Any,
    routing_extra: Mapping[str, Any] | None = None,
) -> None:
    """Stage one decision record on the turn if a writer is registered."""
    writer = _decision_writer
    if writer is None:
        return
    try:
        extra: Mapping[str, Any] = routing_extra if isinstance(routing_extra, Mapping) else {}
        metadata: dict[str, Any] = ctx.metadata
        decision_id = uuid.uuid4().hex
        turn_index: int | None = None
        history = metadata.get("routing_history")
        if isinstance(history, list) and history:
            last = history[-1]
            if isinstance(last, dict) and isinstance(last.get("turn_index"), int):
                turn_index = last["turn_index"]
        record: dict[str, Any] = {
            "decision_id": decision_id,
            "session_key": str(ctx.session_key),
            "turn_index": turn_index,
            "ts_ms": int(time.time() * 1000),
            "classifier": sanitize_token(extra.get("model_version")),
            "proposed_tier": sanitize_token(extra.get("base_tier")) or sanitize_token(decision.tier),
            "confidence": _number(getattr(decision, "confidence", None)),
            "probs": extra.get("probabilities"),
            "flags": extra.get("flags"),
            "final_tier": sanitize_token(extra.get("final_tier")) or sanitize_token(decision.tier),
            "provider": sanitize_token(metadata.get("routed_provider")),
            "model": sanitize_token(getattr(decision, "model", None)),
            "thinking_level": sanitize_token(metadata.get("thinking_level")),
            "source": sanitize_token(getattr(decision, "source", None)),
            "trail": build_trail(extra, final_tier=getattr(decision, "tier", None)),
            "baseline_model": sanitize_token(metadata.get("baseline_model")),
            "savings_pct": _number(metadata.get("savings_pct")),
        }
        metadata[DECISION_ID_METADATA_KEY] = decision_id
        metadata[PENDING_RECORD_KEY] = record
    except Exception:  # noqa: BLE001 — decision records must never fail a turn
        log.warning("router_decision_record.stage_failed", exc_info=True)


def flush_router_decision(
    metadata: dict[str, Any],
    *,
    ensemble_trace: Mapping[str, Any] | None = None,
) -> None:
    """Complete the staged record with executed facts and write it (inline)."""
    writer = _decision_writer
    record: Any = None
    try:
        record = metadata.pop(PENDING_RECORD_KEY, None)
    except Exception:  # noqa: BLE001 — tolerate read-only mappings
        return
    if not isinstance(record, dict) or writer is None:
        return
    try:
        record["executed_kind"] = "ensemble" if bool(metadata.get("ensemble_enabled")) else "single"
        record["ensemble_profile"] = None
        record["fallback_hops"] = int(metadata.get(FALLBACK_HOPS_METADATA_KEY) or 0)
        if not bool(metadata.get("ensemble_enabled")):
            executed_model = sanitize_token(metadata.get("routed_model"))
            if executed_model is not None:
                record["model"] = executed_model
    except Exception:  # noqa: BLE001 — decision records must never fail a turn
        log.warning("router_decision_record.flush_failed", exc_info=True)
        return
    try:
        writer.record_decision(record)
    except Exception:  # noqa: BLE001 — decision records must never fail a turn
        log.warning("router_decision_record.flush_failed", exc_info=True)


def rehydrate_history_from_writer(
    writer: FileDecisionWriter,
    store: Any,
    *,
    window_seconds: int = 1800,
    per_session: int = 5,
) -> int:
    """Seed ``RoutingHistoryStore`` from persisted decisions (best-effort)."""
    try:
        from squilla_router_standalone.history import seed_routing_history

        grouped = writer.load_recent_history(
            window_seconds=window_seconds,
            per_session=per_session,
        )
        if not grouped:
            return 0
        now_ms = int(time.time() * 1000)
        now_mono = time.monotonic()
        entries_by_session: dict[str, list[dict[str, Any]]] = {}
        for session_key, rows in grouped.items():
            entries: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                final_tier = normalize_text_tier(row.get("final_tier"))
                proposed_tier = normalize_text_tier(row.get("proposed_tier"))
                if final_tier is None and proposed_tier is None:
                    continue
                age_seconds = max(0.0, (now_ms - int(row.get("ts_ms") or 0)) / 1000.0)
                turn_index = row.get("turn_index")
                entry: dict[str, Any] = {
                    "turn_index": turn_index if isinstance(turn_index, int) else index,
                    "_ts": max(0.0, now_mono - age_seconds),
                    "base_tier": proposed_tier or final_tier,
                    "final_tier": final_tier or proposed_tier,
                    "route_class": route_class_for_tier(proposed_tier or final_tier or ""),
                    "final_route_class": route_class_for_tier(final_tier or proposed_tier or ""),
                    "rehydrated": True,
                }
                entries.append(entry)
            if entries:
                entries_by_session[session_key] = entries
        return seed_routing_history(store, entries_by_session)
    except Exception:  # noqa: BLE001 — rehydration must never block boot
        log.warning("router_decision_record.rehydrate_failed", exc_info=True)
        return 0
