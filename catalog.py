"""Config-driven tier-capability facts.

Replaces ``opensquilla.provider.model_catalog`` (shared_catalog /
resolve_entry / get_capabilities / resolve_context_window_with_source). The
project reads vision/context-window facts from a live model catalog; in the
standalone package the operator declares them per-tier in TOML
(``supports_vision``, ``context_window``) — which the project already treats as
"definite knowledge" via its ``[models.*]`` override path. ``None`` on either
field means no definite signal was declared, so the capability gate never acts
on ignorance (byte-identical no-op when nothing is declared).
"""

from __future__ import annotations

from squilla_router_standalone.engine.routing.policy import TierCapability
from squilla_router_standalone.router_tiers import TierConfig


def tier_capability_facts(
    tiers: dict,
    valid_tiers: list[str],
    active_provider: str = "",
) -> dict[str, TierCapability]:
    """Return per-tier capability facts declared in tier config.

    ``supports_vision`` defaults to ``None`` (unknown) unless the tier
    declares ``supports_image = true`` (vision) or ``supports_image = false``
    with an explicit ``supports_vision = false`` (non-vision). ``context_window``
    is read straight off the tier entry when the operator declares it.
    """
    _ = active_provider  # kept for signature parity; not used (no catalog)
    facts: dict[str, TierCapability] = {}
    for name in valid_tiers:
        tier = TierConfig.from_value(tiers.get(name))
        if not tier.model:
            facts[name] = TierCapability()
            continue

        supports_vision: bool | None = None
        raw = tiers.get(name)
        declared_vision = None
        if isinstance(raw, dict):
            if "supports_vision" in raw:
                declared_vision = bool(raw.get("supports_vision"))
        # supports_image=true is a definite vision signal; an explicit
        # supports_vision declaration overrides it.
        if declared_vision is not None:
            supports_vision = declared_vision
        elif tier.supports_image:
            supports_vision = True

        context_window = tier.context_window

        facts[name] = TierCapability(
            supports_vision=supports_vision,
            context_window=context_window,
        )
    return facts
