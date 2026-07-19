"""Post-classifier routing policy: named heuristic stages over one decision.

Verbatim port of ``opensquilla.engine.routing.policy`` (only imports are
repointed to the standalone package). Contains the full 8-gate pipeline:
confidence_gate, complaint_upgrade, anti_downgrade, capability_gate, bind,
large_context_floor, budget_gate, provider_mismatch/veto.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from squilla_router_standalone.engine.routing.calibration import (
    CalibrationState,
    apply_bias,
    effective_threshold,
)
from squilla_router_standalone.engine.routing.policy_data import (
    COMPLAINT_TERMS,
    LARGE_CONTEXT_T2_FLOOR_TOKENS,
    LARGE_CONTEXT_T3_CONTEXT_RATIO,
    LARGE_CONTEXT_T3_FLOOR_TOKENS,
    THINKING_MODE_ORDER,
)
from squilla_router_standalone.router_tiers import (
    DEFAULT_TEXT_TIER,
    HIGHEST_TEXT_TIER,
    ROUTE_CLASS_TO_TIER,
    TIER_TO_ROUTE_CLASS,
    TierConfig,
    normalize_text_tier,
    tier_index,
)
from squilla_router_standalone.squilla_router.controller import normalize_decisions


def _canonical_order(valid_tiers: list[str]) -> list[str]:
    """Order tiers by the canonical c0<c1<c2<c3 ladder, not dict/TOML order."""
    return sorted(
        valid_tiers,
        key=lambda name: (0, tier_index(name)) if tier_index(name) >= 0 else (1, 0),
    )

log = structlog.get_logger(__name__)

_TIER_TO_ROUTE_CLASS = dict(TIER_TO_ROUTE_CLASS)
_ROUTE_CLASS_TO_TIER = dict(ROUTE_CLASS_TO_TIER)


@dataclass
class RoutingDecision:
    """Result of squilla router classification."""

    tier: str
    model: str
    confidence: float
    source: str  # "image_route" | "v4_phase3" | "v4_unavailable" | "default"


# ---------------------------------------------------------------------------
# Shared helpers (moved verbatim from the step module)
# ---------------------------------------------------------------------------


def _tier_index(tier: str, valid_tiers: list[str]) -> int:
    normalized = normalize_text_tier(tier) or tier
    ordered = _canonical_order(valid_tiers)
    return ordered.index(normalized) if normalized in ordered else -1


def _upgrade_tier(tier: str, valid_tiers: list[str], steps: int) -> str:
    ordered = _canonical_order(valid_tiers)
    normalized = normalize_text_tier(tier) or tier
    idx = ordered.index(normalized) if normalized in ordered else -1
    if idx < 0:
        return tier
    return ordered[min(idx + max(steps, 0), len(ordered) - 1)]


def _tier_config_value(tier_cfg: object, key: str, default: object = None) -> object:
    if isinstance(tier_cfg, dict):
        return tier_cfg.get(key, default)
    return getattr(tier_cfg, key, default)


def detect_complaint(message: str, max_chars: int | None = None) -> list[str]:
    text = message.strip()
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return []
    lowered = text.lower()
    return [term for term in COMPLAINT_TERMS if term in lowered]


def route_class_for_tier(tier: str) -> str | None:
    normalized = normalize_text_tier(tier) or tier
    return _TIER_TO_ROUTE_CLASS.get(normalized)


def tier_for_route_class(route_class: object) -> str | None:
    if route_class is None:
        return None
    return _ROUTE_CLASS_TO_TIER.get(str(route_class))


def _min_thinking_mode_for_tier(tier: str | None) -> str | None:
    tier = normalize_text_tier(tier)
    if tier == HIGHEST_TEXT_TIER:
        return "T3"
    if tier == "c2":
        return "T2"
    if tier == DEFAULT_TEXT_TIER:
        return "T1"
    return None


def _promote_thinking_mode(current: str | None, minimum: str | None) -> str | None:
    if minimum is None:
        return current
    if current not in THINKING_MODE_ORDER:
        return minimum
    if THINKING_MODE_ORDER[current] < THINKING_MODE_ORDER[minimum]:
        return minimum
    return current


def previous_final_entry(
    routing_history: list[dict] | None,
    now: float,
    window: float,
) -> dict | None:
    if not routing_history:
        return None
    cutoff = now - window
    for entry in reversed(routing_history):
        if entry.get("_ts", now) >= cutoff:
            return entry
    return None


def previous_final_tier(entry: dict | None) -> str | None:
    if not entry:
        return None
    tier = entry.get("final_tier")
    if tier:
        return normalize_text_tier(tier) or str(tier)
    return tier_for_route_class(entry.get("final_route_class") or entry.get("route_class"))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceGateResult:
    tier: str
    applied: bool
    threshold: float
    default_tier: str | None


def confidence_gate(
    tier: str,
    *,
    confidence: float,
    router_cfg: object,
    valid_tiers: list[str],
    tiers: dict | None = None,
    calibration: CalibrationState | None = None,
) -> ConfidenceGateResult:
    """Fall back to the default tier when classifier confidence is too low."""
    base_threshold = float(getattr(router_cfg, "confidence_threshold", 0.5))
    high_tier_margin = float(getattr(router_cfg, "confidence_high_tier_margin", 0.05))
    default_tier = getattr(router_cfg, "default_tier", None)
    threshold = effective_threshold(base_threshold, calibration)
    if default_tier is None:
        return ConfidenceGateResult(tier, False, threshold, None)
    default_tier = normalize_text_tier(default_tier) or str(default_tier)
    selected_cfg = tiers.get(tier, {}) if isinstance(tiers, dict) else {}
    if bool(_tier_config_value(selected_cfg, "image_only", False)):
        return ConfidenceGateResult(tier, False, threshold, default_tier)
    gate_confidence = apply_bias(confidence, tier, calibration)
    tier_rank = _tier_index(tier, valid_tiers)
    default_rank = _tier_index(default_tier, valid_tiers)
    cutoff = threshold - high_tier_margin if tier_rank > default_rank else threshold
    if gate_confidence < cutoff and tier_rank >= 0 and default_rank >= 0 and tier != default_tier:
        return ConfidenceGateResult(default_tier, True, threshold, default_tier)
    return ConfidenceGateResult(tier, False, threshold, default_tier)


@dataclass(frozen=True)
class ComplaintUpgradeResult:
    tier: str
    terms: list[str]
    applied: bool
    steps: int
    max_chars: int


def complaint_upgrade(
    tier: str,
    *,
    message: str,
    router_cfg: object,
    valid_tiers: list[str],
    pre_confidence_tier: str | None,
    previous_tier: str | None,
) -> ComplaintUpgradeResult:
    """Upgrade the tier when a short message contains a known complaint term."""
    steps = int(getattr(router_cfg, "complaint_upgrade_steps", 1))
    max_chars = int(getattr(router_cfg, "complaint_upgrade_max_chars", 160))
    if not getattr(router_cfg, "complaint_upgrade_enabled", True):
        return ComplaintUpgradeResult(tier, [], False, steps, max_chars)
    terms = detect_complaint(message, max_chars=max_chars)
    if not terms:
        return ComplaintUpgradeResult(tier, [], False, steps, max_chars)
    upgrade_start_tier = tier
    if pre_confidence_tier in valid_tiers and _tier_index(
        pre_confidence_tier or "", valid_tiers
    ) > _tier_index(upgrade_start_tier, valid_tiers):
        upgrade_start_tier = pre_confidence_tier or upgrade_start_tier
    if previous_tier in valid_tiers and _tier_index(
        previous_tier or "", valid_tiers
    ) > _tier_index(upgrade_start_tier, valid_tiers):
        upgrade_start_tier = previous_tier or upgrade_start_tier
    upgraded_tier = _upgrade_tier(upgrade_start_tier, valid_tiers, steps)
    return ComplaintUpgradeResult(upgraded_tier, terms, upgraded_tier != tier, steps, max_chars)


@dataclass(frozen=True)
class AntiDowngradeResult:
    tier: str
    applied: bool


def anti_downgrade(
    tier: str,
    *,
    router_cfg: object,
    valid_tiers: list[str],
    previous_tier: str | None,
) -> AntiDowngradeResult:
    """Hold the previous turn's tier when routing would drop below it."""
    if (
        getattr(router_cfg, "kv_cache_anti_downgrade_enabled", True)
        and previous_tier in valid_tiers
        and _tier_index(tier, valid_tiers) >= 0
        and _tier_index(previous_tier or "", valid_tiers) > _tier_index(tier, valid_tiers)
    ):
        return AntiDowngradeResult(previous_tier or tier, True)
    return AntiDowngradeResult(tier, False)


@dataclass(frozen=True)
class TierCapability:
    """Definite catalog facts for one tier's model.

    In the standalone package these facts come from operator-declared tier
    config (``supports_vision`` / ``context_window``) rather than the project's
    ``provider.model_catalog``. ``None`` on either field means no definite
    signal was declared — the capability gate never acts on a ``None``.
    """

    supports_vision: bool | None = None
    context_window: int | None = None


@dataclass(frozen=True)
class CapabilityGateAction:
    rule: str  # "vision_walk_up" | "context_walk_up"
    from_tier: str
    to_tier: str


@dataclass(frozen=True)
class CapabilityGateResult:
    tier: str
    actions: tuple[CapabilityGateAction, ...] = ()


def capability_gate(
    tier: str,
    *,
    valid_tiers: list[str],
    tier_capabilities: dict[str, TierCapability] | None,
    turn_has_image: bool,
    material_tokens: int,
) -> CapabilityGateResult:
    """Walk the working tier UP when the declared capabilities say its model
    cannot serve the turn (vision input, context fit)."""
    if not tier_capabilities:
        return CapabilityGateResult(tier)
    ordered = _canonical_order(valid_tiers)
    normalized = normalize_text_tier(tier) or tier
    idx = ordered.index(normalized) if normalized in ordered else -1
    if idx < 0:
        return CapabilityGateResult(tier)
    current = ordered[idx]
    actions: list[CapabilityGateAction] = []

    def _caps(name: str) -> TierCapability:
        capability = tier_capabilities.get(name)
        return capability if capability is not None else TierCapability()

    if turn_has_image and _caps(current).supports_vision is False:
        for candidate in ordered[idx + 1 :]:
            if _caps(candidate).supports_vision is True:
                actions.append(CapabilityGateAction("vision_walk_up", current, candidate))
                current = candidate
                idx = ordered.index(current)
                break

    window = _caps(current).context_window
    if material_tokens > 0 and window is not None and material_tokens > window:
        target: str | None = None
        for candidate in ordered[idx + 1 :]:
            candidate_window = _caps(candidate).context_window
            if candidate_window is not None and material_tokens <= candidate_window:
                target = candidate
                break
        if target is None and idx < len(ordered) - 1:
            target = ordered[-1]  # nothing definitely fits: saturate at the top
        if target is not None and target != current:
            actions.append(CapabilityGateAction("context_walk_up", current, target))
            current = target

    return CapabilityGateResult(current, tuple(actions))


def record_capability_gate_trail(extra: dict, result: CapabilityGateResult) -> None:
    """Append the gate's actions to the ``routing_extra`` trail."""
    if not result.actions:
        return
    trail = extra.setdefault("routing_trail", [])
    for action in result.actions:
        trail.append(
            {
                "stage": "capability_gate",
                "rule": action.rule,
                "from_tier": action.from_tier,
                "to_tier": action.to_tier,
            }
        )
    extra["capability_gate_applied"] = True


def bind(
    decision: RoutingDecision,
    *,
    final_tier: str,
    tiers: dict,
    extra: dict,
    base_tier: str,
    pre_confidence_tier: str,
    gate: ConfidenceGateResult,
    complaint: ComplaintUpgradeResult,
    downgrade: AntiDowngradeResult,
    previous_tier: str | None,
    previous_route_class: object,
    window: float,
) -> RoutingDecision:
    """Record the finalized routing trail and rebind to the final tier's model."""
    final_route_class = route_class_for_tier(final_tier)
    extra.update(
        {
            "base_tier": base_tier,
            "pre_confidence_tier": normalize_text_tier(pre_confidence_tier)
            or pre_confidence_tier,
            "confidence_threshold": gate.threshold,
            "confidence_default_tier": gate.default_tier,
            "confidence_gate_applied": gate.applied,
            "final_tier": final_tier,
            "final_route_class": final_route_class,
            "complaint_detected": bool(complaint.terms),
            "complaint_terms": complaint.terms,
            "complaint_upgrade_applied": complaint.applied,
            "complaint_upgrade_steps": complaint.steps,
            "complaint_upgrade_max_chars": complaint.max_chars,
            "anti_downgrade_applied": downgrade.applied,
            "previous_tier": normalize_text_tier(previous_tier) or previous_tier,
            "previous_route_class": previous_route_class,
            "kv_cache_window_seconds": window,
        }
    )

    return RoutingDecision(
        tier=final_tier,
        model=tiers[final_tier].get("model", decision.model),
        confidence=decision.confidence,
        source=decision.source,
    )


def reconcile_controller_with_final_tier(
    thinking_mode: str | None,
    prompt_policy: str | None,
    extra: dict,
) -> tuple[str | None, str | None]:
    """Keep controller output consistent with OpenSquilla's final tier overrides."""
    final_tier = normalize_text_tier(extra.get("final_tier")) or extra.get("final_tier")
    base_tier = normalize_text_tier(extra.get("base_tier")) or extra.get("base_tier")
    if not final_tier or final_tier == base_tier:
        return thinking_mode, prompt_policy

    original_thinking = thinking_mode
    original_prompt = prompt_policy

    thinking_mode = _promote_thinking_mode(
        thinking_mode,
        _min_thinking_mode_for_tier(str(final_tier)),
    )
    if prompt_policy == "P0" and (
        str(final_tier) in {"c2", HIGHEST_TEXT_TIER} or extra.get("complaint_detected")
    ):
        prompt_policy = "P1"
    if thinking_mode is not None and prompt_policy is not None:
        thinking_mode, prompt_policy = normalize_decisions(thinking_mode, prompt_policy)

    if thinking_mode != original_thinking or prompt_policy != original_prompt:
        extra.setdefault("base_thinking_mode", original_thinking)
        extra.setdefault("base_prompt_policy", original_prompt)
        extra["thinking_mode"] = thinking_mode
        extra["prompt_policy"] = prompt_policy
        extra["controller_reconciled"] = True
    else:
        extra.setdefault("controller_reconciled", False)
    return thinking_mode, prompt_policy


def large_context_min_tier(material_tokens: int, context_window_tokens: int) -> str | None:
    """Minimum tier a turn with this much material context may run on."""
    if (
        material_tokens >= LARGE_CONTEXT_T3_FLOOR_TOKENS
        or material_tokens >= int(context_window_tokens * LARGE_CONTEXT_T3_CONTEXT_RATIO)
    ):
        return HIGHEST_TEXT_TIER
    if material_tokens >= LARGE_CONTEXT_T2_FLOOR_TOKENS:
        return "c2"
    return None


def large_context_floor(
    decision: RoutingDecision,
    *,
    tiers: dict,
    valid_tiers: list[str],
    material_tokens: int,
    context_window_tokens: int,
    extra: dict | None,
    metadata_updates: dict,
) -> RoutingDecision:
    """Floor the routed tier for turns carrying large material contexts."""
    if decision.tier not in valid_tiers:
        return decision

    min_tier = large_context_min_tier(material_tokens, context_window_tokens)
    if min_tier is None:
        return decision
    if min_tier not in valid_tiers:
        return decision
    if _tier_index(decision.tier, valid_tiers) >= _tier_index(min_tier, valid_tiers):
        return decision

    floored = RoutingDecision(
        tier=min_tier,
        model=tiers[min_tier].get("model", decision.model),
        confidence=decision.confidence,
        source="large_context_floor",
    )
    metadata_updates["large_context_floor_from_tier"] = decision.tier
    metadata_updates["large_context_material_tokens"] = material_tokens

    if extra is not None:
        extra.setdefault("base_tier", decision.tier)
        extra["large_context_floor_applied"] = True
        extra["large_context_floor_from_tier"] = decision.tier
        extra["large_context_floor_min_tier"] = min_tier
        extra["large_context_material_tokens"] = material_tokens
        extra["large_context_pre_floor_source"] = decision.source
        extra["final_tier"] = min_tier
        extra["final_route_class"] = route_class_for_tier(min_tier)

    return floored


# ---------------------------------------------------------------------------
# Budget gate (additive, default-off)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetGateInput:
    action: str  # "warn" | "cap"
    limit_usd: float
    spend_usd: float | None
    estimate_usd: float | None = None
    cap_tier: str | None = None
    spend_source: str = "unknown"
    session_key: str | None = None


@dataclass(frozen=True)
class BudgetGateResult:
    tier: str
    outcome: str  # "off" | "suspended" | "under_limit" | "warn" | "cap"
    spend_usd: float | None = None
    projected_usd: float | None = None
    limit_usd: float | None = None
    action: str = "off"
    from_tier: str = ""
    spend_source: str = "unknown"
    session_key: str | None = None


def budget_gate(
    tier: str,
    *,
    valid_tiers: list[str],
    budget: BudgetGateInput | None,
) -> BudgetGateResult:
    """Warn or cap when accumulated session spend crosses the configured limit."""
    if budget is None:
        return BudgetGateResult(tier, "off")
    if budget.spend_usd is None:
        return BudgetGateResult(
            tier,
            "suspended",
            limit_usd=budget.limit_usd,
            action=budget.action,
            spend_source=budget.spend_source,
            session_key=budget.session_key,
        )
    projected = budget.spend_usd + (budget.estimate_usd or 0.0)
    common: dict[str, object] = {
        "spend_usd": budget.spend_usd,
        "projected_usd": projected,
        "limit_usd": budget.limit_usd,
        "spend_source": budget.spend_source,
        "session_key": budget.session_key,
    }
    if projected <= budget.limit_usd:
        return BudgetGateResult(tier, "under_limit", action=budget.action, **common)  # type: ignore[arg-type]
    if budget.action == "cap":
        target = normalize_text_tier(budget.cap_tier) if budget.cap_tier else None
        if (
            target is not None
            and target in valid_tiers
            and _tier_index(target, valid_tiers) < _tier_index(tier, valid_tiers)
        ):
            return BudgetGateResult(target, "cap", action="cap", from_tier=tier, **common)  # type: ignore[arg-type]
        return BudgetGateResult(tier, "warn", action="warn", from_tier=tier, **common)  # type: ignore[arg-type]
    return BudgetGateResult(tier, "warn", action="warn", from_tier=tier, **common)  # type: ignore[arg-type]


def record_budget_gate_trail(extra: dict, result: BudgetGateResult) -> None:
    """Append the budget gate's action to the routing trail (warn/cap only)."""
    if result.outcome not in ("warn", "cap"):
        return
    entry: dict[str, object] = {
        "stage": "budget_gate",
        "rule": result.outcome,
        "spend_usd": result.spend_usd,
        "limit_usd": result.limit_usd,
        "spend_source": result.spend_source,
    }
    if result.outcome == "cap":
        entry["from_tier"] = result.from_tier
        entry["to_tier"] = result.tier
    extra.setdefault("routing_trail", []).append(entry)
    extra["budget_gate_applied"] = True
    extra["budget_gate_outcome"] = result.outcome


def apply_budget_gate(
    decision: RoutingDecision,
    result: BudgetGateResult,
    *,
    tiers: dict,
    extra: dict | None,
    metadata_updates: dict,
) -> RoutingDecision:
    """Apply a :func:`budget_gate` result to the decision + turn metadata."""
    if result.outcome not in ("warn", "cap"):
        return decision

    metadata_updates["router_budget_applied"] = True
    metadata_updates["router_budget_outcome"] = result.outcome
    metadata_updates["router_budget_action"] = result.action
    metadata_updates["router_budget_limit_usd"] = result.limit_usd
    metadata_updates["router_budget_spend_source"] = result.spend_source
    if result.spend_usd is not None:
        metadata_updates["router_budget_spend_usd"] = result.spend_usd
    if result.projected_usd is not None and result.projected_usd != result.spend_usd:
        metadata_updates["router_budget_projected_usd"] = result.projected_usd
    if extra is not None:
        record_budget_gate_trail(extra, result)

    if result.outcome == "cap":
        metadata_updates["router_budget_from_tier"] = result.from_tier
        metadata_updates["router_budget_to_tier"] = result.tier
        capped = RoutingDecision(
            tier=result.tier,
            model=tiers[result.tier].get("model", decision.model),
            confidence=decision.confidence,
            source="budget_cap",
        )
        if extra is not None:
            extra["final_tier"] = result.tier
            extra["final_route_class"] = route_class_for_tier(result.tier)
        return capped

    return decision  # warn: tier UNCHANGED


@dataclass(frozen=True)
class ProviderMismatchOutcome:
    outcome: str  # "skipped" | "match" | "cross_provider" | "mismatch"
    routed_provider: str | None
    tier_provider: str
    tier_model: str
    active_provider: str


def provider_mismatch(
    *,
    tiers: dict,
    tier_name: str,
    routing_applied: bool,
    active_provider: str,
    cross_provider_tiers: bool,
) -> ProviderMismatchOutcome:
    """Assess the routed tier's provider; never vetoes or alters the decision."""
    if not routing_applied:
        return ProviderMismatchOutcome("skipped", None, "", "", "")
    tier = TierConfig.from_value(tiers.get(tier_name))
    routed_provider = tier.provider.lower() if tier.provider else None
    active = str(active_provider or "").strip().lower()
    if not tier.provider or not active:
        return ProviderMismatchOutcome("match", routed_provider, tier.provider, tier.model, active)
    if tier.provider.lower() == active:
        return ProviderMismatchOutcome("match", routed_provider, tier.provider, tier.model, active)
    if cross_provider_tiers:
        return ProviderMismatchOutcome(
            "cross_provider", routed_provider, tier.provider, tier.model, active
        )
    return ProviderMismatchOutcome("mismatch", routed_provider, tier.provider, tier.model, active)


@dataclass(frozen=True)
class ProviderMismatchVeto:
    applied: bool
    from_tier: str = ""
    to_tier: str = ""


def provider_mismatch_veto(
    *,
    tiers: dict,
    tier_name: str,
    valid_tiers: list[str],
    routing_applied: bool,
    active_provider: str,
    cross_provider_tiers: bool,
    default_tier: object = None,
) -> ProviderMismatchVeto:
    """Pick the rebind target when a provider mismatch must be vetoed."""
    outcome = provider_mismatch(
        tiers=tiers,
        tier_name=tier_name,
        routing_applied=routing_applied,
        active_provider=active_provider,
        cross_provider_tiers=cross_provider_tiers,
    )
    if outcome.outcome != "mismatch":
        return ProviderMismatchVeto(False)

    current = normalize_text_tier(tier_name) or tier_name
    idx = _tier_index(current, valid_tiers)
    active = str(active_provider or "").strip().lower()

    def _executes_on_active(name: str) -> bool:
        tier = TierConfig.from_value(tiers.get(name))
        return not tier.provider or tier.provider.lower() == active

    if idx >= 0:
        candidates = sorted(
            (name for name in valid_tiers if name != current and _executes_on_active(name)),
            key=lambda name: (
                abs(_tier_index(name, valid_tiers) - idx),
                _tier_index(name, valid_tiers),
            ),
        )
        if candidates:
            return ProviderMismatchVeto(True, current, candidates[0])

    fallback = normalize_text_tier(default_tier) or (
        str(default_tier) if default_tier else None
    )
    if fallback and fallback in valid_tiers and fallback != current:
        return ProviderMismatchVeto(True, current, fallback)
    return ProviderMismatchVeto(False, current, "")


def record_provider_mismatch_veto_trail(extra: dict, veto: ProviderMismatchVeto) -> None:
    """Append a veto rebind to the ``routing_extra`` trail (only when applied)."""
    if not veto.applied:
        return
    extra.setdefault("routing_trail", []).append(
        {
            "stage": "provider_mismatch",
            "rule": "veto_rebind",
            "from_tier": veto.from_tier,
            "to_tier": veto.to_tier,
        }
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class PolicyInputs:
    """Classifier output plus turn facts, as plain data."""

    decision: RoutingDecision
    message: str
    router_cfg: object
    tiers: dict
    valid_tiers: list[str]
    routing_history: list[dict] | None
    extra: dict | None
    thinking_mode: str | None
    prompt_policy: str | None
    history_strategy: bool
    material_estimated_tokens: int
    context_window_tokens: int
    now: float | None = None
    turn_has_image: bool = False
    tier_capabilities: dict[str, TierCapability] | None = None
    calibration: CalibrationState | None = None
    budget: BudgetGateInput | None = None


@dataclass
class PolicyResult:
    decision: RoutingDecision
    thinking_mode: str | None
    prompt_policy: str | None
    metadata_updates: dict = field(default_factory=dict)


class RoutingPolicyEngine:
    """Runs the post-classifier stages in the exact legacy order."""

    def run(self, inputs: PolicyInputs) -> PolicyResult:
        decision = inputs.decision
        thinking_mode = inputs.thinking_mode
        prompt_policy = inputs.prompt_policy
        metadata_updates: dict = {}
        extra = inputs.extra if isinstance(inputs.extra, dict) else None

        if inputs.history_strategy and extra is not None:
            decision = self._finalize(decision, inputs, extra)
            thinking_mode, prompt_policy = reconcile_controller_with_final_tier(
                thinking_mode,
                prompt_policy,
                extra,
            )

        decision = large_context_floor(
            decision,
            tiers=inputs.tiers,
            valid_tiers=inputs.valid_tiers,
            material_tokens=inputs.material_estimated_tokens,
            context_window_tokens=inputs.context_window_tokens,
            extra=extra,
            metadata_updates=metadata_updates,
        )
        if decision.source == "large_context_floor" and extra is not None:
            thinking_mode, prompt_policy = reconcile_controller_with_final_tier(
                thinking_mode,
                prompt_policy,
                extra,
            )

        if inputs.budget is not None:
            budget_result = budget_gate(
                decision.tier,
                valid_tiers=inputs.valid_tiers,
                budget=inputs.budget,
            )
            decision = apply_budget_gate(
                decision,
                budget_result,
                tiers=inputs.tiers,
                extra=extra,
                metadata_updates=metadata_updates,
            )

        return PolicyResult(
            decision=decision,
            thinking_mode=thinking_mode,
            prompt_policy=prompt_policy,
            metadata_updates=metadata_updates,
        )

    def _finalize(
        self,
        decision: RoutingDecision,
        inputs: PolicyInputs,
        extra: dict,
    ) -> RoutingDecision:
        base_tier = normalize_text_tier(decision.tier) or decision.tier
        final_tier = base_tier
        base_route_class = extra.get("route_class") or route_class_for_tier(base_tier)
        if base_route_class is not None:
            extra["route_class"] = base_route_class
            extra.setdefault("top1_label", base_route_class)

        pre_confidence_tier = final_tier
        gate = confidence_gate(
            final_tier,
            confidence=decision.confidence,
            router_cfg=inputs.router_cfg,
            valid_tiers=inputs.valid_tiers,
            tiers=inputs.tiers,
            calibration=inputs.calibration,
        )
        final_tier = gate.tier

        now = inputs.now if inputs.now is not None else time.monotonic()
        window = float(
            getattr(inputs.router_cfg, "kv_cache_anti_downgrade_window_seconds", 600)
        )
        previous_entry = previous_final_entry(inputs.routing_history, now, window)
        previous_tier = previous_final_tier(previous_entry)
        previous_route_class = None
        if previous_entry:
            previous_route_class = previous_entry.get("final_route_class") or previous_entry.get(
                "route_class"
            )

        complaint = complaint_upgrade(
            final_tier,
            message=inputs.message,
            router_cfg=inputs.router_cfg,
            valid_tiers=inputs.valid_tiers,
            pre_confidence_tier=pre_confidence_tier,
            previous_tier=previous_tier,
        )
        final_tier = complaint.tier

        downgrade = anti_downgrade(
            final_tier,
            router_cfg=inputs.router_cfg,
            valid_tiers=inputs.valid_tiers,
            previous_tier=previous_tier,
        )
        final_tier = downgrade.tier

        gate_capabilities = capability_gate(
            final_tier,
            valid_tiers=inputs.valid_tiers,
            tier_capabilities=inputs.tier_capabilities,
            turn_has_image=inputs.turn_has_image,
            material_tokens=inputs.material_estimated_tokens,
        )
        record_capability_gate_trail(extra, gate_capabilities)
        final_tier = gate_capabilities.tier

        return bind(
            decision,
            final_tier=final_tier,
            tiers=inputs.tiers,
            extra=extra,
            base_tier=base_tier,
            pre_confidence_tier=pre_confidence_tier,
            gate=gate,
            complaint=complaint,
            downgrade=downgrade,
            previous_tier=previous_tier,
            previous_route_class=previous_route_class,
            window=window,
        )
