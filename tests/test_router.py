"""Tests for the standalone router package.

Covers: import smoke, the 8-gate policy engine (parity with the project's
post-classifier stages), the history-entry schema fix (text + full extra
trail, so V4 ML keeps history-awareness on turn 2+), and end-to-end heuristic
routing through the full step pipeline.
"""

from __future__ import annotations

import sys
import types
import time
from types import SimpleNamespace

import pytest

from squilla_router_standalone import Router, SquillaRouterConfig
from squilla_router_standalone.engine.routing import (
    BudgetGateInput,
    PolicyInputs,
    RoutingDecision,
    RoutingPolicyEngine,
    TierCapability,
    anti_downgrade,
    budget_gate,
    capability_gate,
    complaint_upgrade,
    confidence_gate,
    large_context_floor,
    large_context_min_tier,
    previous_final_entry,
    previous_final_tier,
    route_class_for_tier,
)
from squilla_router_standalone.history import (
    RoutingHistoryStore,
    _routing_history_entry,
    append_routing_history,
)
from squilla_router_standalone.router_tiers import TEXT_TIERS, normalize_text_tier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _router_cfg(**overrides) -> SimpleNamespace:
    """A minimal router_cfg object whose attributes the policy engine reads."""
    base = SimpleNamespace(
        confidence_threshold=0.5,
        confidence_high_tier_margin=0.05,
        default_tier="c1",
        complaint_upgrade_enabled=True,
        complaint_upgrade_steps=1,
        complaint_upgrade_max_chars=160,
        kv_cache_anti_downgrade_enabled=True,
        kv_cache_anti_downgrade_window_seconds=600,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def _tiers() -> dict:
    # Project default: text tiers are non-vision; image_model is the vision tier.
    return {
        "c0": {"model": "haiku", "supports_image": False, "context_window": 200_000},
        "c1": {"model": "sonnet", "supports_image": False, "context_window": 200_000},
        "c2": {"model": "sonnet", "supports_image": False, "context_window": 200_000},
        "c3": {"model": "opus", "supports_image": False, "context_window": 200_000},
    }


def _heuristic_config() -> SquillaRouterConfig:
    cfg = SquillaRouterConfig()
    cfg.strategy = "heuristic"
    cfg.tiers = _tiers() | {"image_model": {"model": "sonnet", "supports_image": True, "image_only": True}}
    cfg.rollout_phase = "full"
    return cfg


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------


def test_imports_smoke():
    """The package imports cleanly without ML extras (V4 degrades to heuristic)."""
    import squilla_router_standalone  # noqa: F401
    import squilla_router_standalone.api  # noqa: F401
    import squilla_router_standalone.engine.steps.squilla_router  # noqa: F401
    assert Router is not None
    assert SquillaRouterConfig is not None


# ---------------------------------------------------------------------------
# History-entry schema (the critical fix)
# ---------------------------------------------------------------------------


def test_history_entry_carries_text_and_extra_trail():
    """History entries must carry `text` + the full `extra` trail so V4 ML
    can read history_user_texts and prev_route_decisions on turn 2+. The old
    docs proxy stored only tier labels — this is the regression test for that."""
    store = RoutingHistoryStore()
    extra = {
        "route_class": "R1",
        "final_tier": "c1",
        "final_route_class": "R1",
        "difficulty": 0.3,
        "margin": 0.4,
        "probabilities": {"R1": 0.9},
    }
    decision = RoutingDecision(tier="c1", model="sonnet", confidence=0.9, source="v4_phase3")
    payload = _routing_history_entry(text="重写这个函数", extra=extra, decision=decision)
    history = append_routing_history(store, "s1", payload)

    assert len(history) == 1
    entry = history[0]
    # The schema fix: text + extra fields are present.
    assert entry["text"] == "重写这个函数"
    assert entry["final_tier"] == "c1"
    assert entry["final_route_class"] == "R1"
    assert entry["difficulty"] == 0.3
    assert entry["margin"] == 0.4
    # V4 _build_request reads entry["text"] -> history_user_texts; reads
    # final_route_class/difficulty/margin -> prev_route_decisions. Those keys
    # must survive append (which spreads **extra into the entry).
    assert "_ts" in entry and "turn_index" in entry


def test_previous_final_tier_reads_history_schema():
    """previous_final_tier must resolve from the full history entry."""
    entry = {"final_tier": "c2", "final_route_class": "R2", "_ts": time.monotonic()}
    assert previous_final_tier(entry) == "c2"
    # Fallback to final_route_class when final_tier absent.
    entry2 = {"final_route_class": "R3", "_ts": time.monotonic()}
    assert previous_final_tier(entry2) == "c3"


def test_previous_final_entry_window():
    now = time.monotonic()
    old = {"final_tier": "c3", "_ts": now - 1000}
    recent = {"final_tier": "c2", "_ts": now - 10}
    assert previous_final_entry([old, recent], now, 600) is recent
    assert previous_final_entry([old], now, 600) is None


# ---------------------------------------------------------------------------
# confidence_gate
# ---------------------------------------------------------------------------


def test_confidence_gate_falls_back_to_default_for_high_tier():
    cfg = _router_cfg(default_tier="c1")
    # c3 above default c1, confidence 0.40 < 0.45 cutoff (0.5 - 0.05) -> fallback.
    result = confidence_gate("c3", confidence=0.40, router_cfg=cfg,
                             valid_tiers=list(TEXT_TIERS), tiers=_tiers())
    assert result.applied is True
    assert result.tier == "c1"


def test_confidence_gate_keeps_confident_high_tier():
    cfg = _router_cfg(default_tier="c1")
    result = confidence_gate("c3", confidence=0.60, router_cfg=cfg,
                             valid_tiers=list(TEXT_TIERS), tiers=_tiers())
    assert result.applied is False
    assert result.tier == "c3"


# ---------------------------------------------------------------------------
# complaint_upgrade
# ---------------------------------------------------------------------------


def test_complaint_upgrade_zh():
    cfg = _router_cfg()
    # "重写" is a complaint term; c1 -> upgrade 1 step -> c2.
    result = complaint_upgrade("c1", message="重写这个回答", router_cfg=cfg,
                                valid_tiers=list(TEXT_TIERS),
                                pre_confidence_tier="c1", previous_tier=None)
    assert result.applied is True
    assert "重写" in result.terms
    assert result.tier == "c2"


def test_complaint_upgrade_skips_long_message():
    cfg = _router_cfg()
    long_msg = "请帮我看看这段很长的内容 " + "x" * 200 + " 重写"
    result = complaint_upgrade("c1", message=long_msg, router_cfg=cfg,
                               valid_tiers=list(TEXT_TIERS),
                               pre_confidence_tier="c1", previous_tier=None)
    # max_chars=160 truncates -> no complaint detected on a too-long message.
    assert result.applied is False


# ---------------------------------------------------------------------------
# anti_downgrade
# ---------------------------------------------------------------------------


def test_anti_downgrade_holds_previous_higher_tier():
    cfg = _router_cfg()
    result = anti_downgrade("c1", router_cfg=cfg, valid_tiers=list(TEXT_TIERS),
                            previous_tier="c3")
    assert result.applied is True
    assert result.tier == "c3"


def test_anti_downgrade_allows_upgrade():
    cfg = _router_cfg()
    result = anti_downgrade("c2", router_cfg=cfg, valid_tiers=list(TEXT_TIERS),
                            previous_tier="c1")
    assert result.applied is False
    assert result.tier == "c2"


# ---------------------------------------------------------------------------
# capability_gate (config-driven)
# ---------------------------------------------------------------------------


def test_capability_gate_vision_walk_up():
    # c0 model is non-vision, turn carries an image, c2 is vision-capable.
    caps = {
        "c0": TierCapability(supports_vision=False, context_window=200_000),
        "c1": TierCapability(supports_vision=False, context_window=200_000),
        "c2": TierCapability(supports_vision=True, context_window=200_000),
        "c3": TierCapability(supports_vision=True, context_window=200_000),
    }
    result = capability_gate("c0", valid_tiers=list(TEXT_TIERS),
                             tier_capabilities=caps, turn_has_image=True, material_tokens=100)
    assert result.tier == "c2"
    assert any(a.rule == "vision_walk_up" for a in result.actions)


def test_capability_gate_context_walk_up():
    # c0 context_window=1000, material_tokens=5000 -> walk to a tier that fits.
    caps = {
        "c0": TierCapability(supports_vision=True, context_window=1000),
        "c1": TierCapability(supports_vision=True, context_window=1000),
        "c2": TierCapability(supports_vision=True, context_window=200_000),
        "c3": TierCapability(supports_vision=True, context_window=200_000),
    }
    result = capability_gate("c0", valid_tiers=list(TEXT_TIERS),
                             tier_capabilities=caps, turn_has_image=False, material_tokens=5000)
    assert result.tier == "c2"
    assert any(a.rule == "context_walk_up" for a in result.actions)


def test_capability_gate_noop_without_definite_facts():
    # All None -> strict no-op (parity with pre-gate pipeline).
    caps = {t: TierCapability() for t in TEXT_TIERS}
    result = capability_gate("c0", valid_tiers=list(TEXT_TIERS),
                             tier_capabilities=caps, turn_has_image=True, material_tokens=999_999)
    assert result.tier == "c0"
    assert result.actions == ()


# ---------------------------------------------------------------------------
# large_context_floor
# ---------------------------------------------------------------------------


def test_large_context_min_tier():
    assert large_context_min_tier(25_000, 200_000) == "c2"
    assert large_context_min_tier(80_000, 200_000) == "c3"
    assert large_context_min_tier(90_000, 200_000) == "c3"  # ratio 0.45 > 0.40
    assert large_context_min_tier(1_000, 200_000) is None


def test_large_context_floor_raises_to_c2():
    decision = RoutingDecision(tier="c0", model="haiku", confidence=0.9, source="v4_phase3")
    extra: dict = {}
    meta: dict = {}
    floored = large_context_floor(decision, tiers=_tiers(), valid_tiers=list(TEXT_TIERS),
                                  material_tokens=30_000, context_window_tokens=200_000,
                                  extra=extra, metadata_updates=meta)
    assert floored.tier == "c2"
    assert extra["large_context_floor_applied"] is True
    assert extra["final_tier"] == "c2"


# ---------------------------------------------------------------------------
# budget_gate
# ---------------------------------------------------------------------------


def test_budget_gate_cap_lowers_tier():
    budget = BudgetGateInput(action="cap", limit_usd=1.0, spend_usd=2.0,
                             estimate_usd=0.0, cap_tier="c1", session_key="s")
    result = budget_gate("c3", valid_tiers=list(TEXT_TIERS), budget=budget)
    assert result.outcome == "cap"
    assert result.tier == "c1"


def test_budget_gate_none_is_off():
    assert budget_gate("c3", valid_tiers=list(TEXT_TIERS), budget=None).outcome == "off"


# ---------------------------------------------------------------------------
# Full pipeline via Router (heuristic strategy, no ML deps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heuristic_routes_by_length():
    router = Router(_heuristic_config())
    short = await router.route("hi", session_key="s1")
    assert short.source == "heuristic"
    assert short.tier == "c0"  # <= 240 chars -> short_plain -> c0

    long_code = "```python\n" + "x = 1\n" * 400 + "```"  # fenced -> code_or_material
    routed = await router.route(long_code, session_key="s2")
    assert routed.tier == "c2"

    heavy = "x" * 13_000  # >= 12000 chars -> heavy -> c3
    heavy_routed = await router.route(heavy, session_key="s3")
    assert heavy_routed.tier == "c3"


@pytest.mark.asyncio
async def test_image_turn_routed_to_image_tier():
    router = Router(_heuristic_config())
    result = await router.route("what is this", session_key="s-img",
                                attachments=[{"type": "image/png"}])
    assert result.source == "image_route"
    assert result.tier == "image_model"


@pytest.mark.asyncio
async def test_runtime_status_reports_heuristic():
    router = Router(_heuristic_config())
    status = router.runtime_status()
    # Heuristic strategy loads even without ML extras.
    assert status["strategy"] in ("heuristic", "v4_phase3")
    assert status["initialized"] is True
