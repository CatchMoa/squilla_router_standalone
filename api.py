"""High-level routing API: ``Router.route()``.

Wraps :func:`apply_squilla_router` so library users get a single awaitable
that takes plain inputs (message + session + optional context) and returns a
flat :class:`RoutingResult`. Owns the process-wide decision writer and the
rehydration seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from squilla_router_standalone.config import SquillaRouterConfig
from squilla_router_standalone.decision_record import (
    FileDecisionWriter,
    flush_router_decision,
    rehydrate_history_from_writer,
    set_decision_writer,
)
from squilla_router_standalone.engine.pipeline import LlmConfig, StandaloneConfig, TurnContext
from squilla_router_standalone.engine.steps.squilla_router import (
    apply_squilla_router,
    get_history_store,
    preload_strategy,
    router_runtime_status,
)
from squilla_router_standalone.pricing import set_prices
from squilla_router_standalone.router_tiers import DEFAULT_TEXT_TIER

log = structlog.get_logger(__name__)


@dataclass
class RoutingResult:
    """Flat result of routing one turn."""

    tier: str
    model: str
    confidence: float
    source: str
    thinking_mode: str | None = None
    thinking_level: str | None = None
    prompt_policy: str | None = None
    prompt_hint: str | None = None
    routing_applied: bool = False
    routing_extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Router:
    """The standalone routing engine entry point.

    Construct with a :class:`SquillaRouterConfig` (or call
    :meth:`from_toml`). ``route()`` runs the full squilla-router pipeline
    (classify -> 8-gate policy -> controller -> history append) for one turn.
    """

    def __init__(
        self,
        config: SquillaRouterConfig,
        *,
        active_provider: str = "",
        state_dir: str | None = None,
        persist_decisions: bool = False,
        base_model: str | None = None,
    ) -> None:
        from squilla_router_standalone.engine.routing.calibration import set_state_dir

        self.config = config
        self.active_provider = active_provider
        if state_dir:
            set_state_dir(state_dir)
        # Wire operator-declared prices into the pricing table.
        if config.prices:
            set_prices(config.prices)

        self._llm = LlmConfig(provider=active_provider)
        self._app_config = StandaloneConfig(squilla_router=config, llm=self._llm)
        self._writer: FileDecisionWriter | None = None
        if persist_decisions:
            self._writer = FileDecisionWriter()
            set_decision_writer(self._writer)
            # Rehydrate history from any persisted decisions.
            rehydrate_history_from_writer(self._writer, get_history_store())

        # Default model = the default tier's model (or base_model override).
        self._default_model = base_model or self._tier_model(DEFAULT_TEXT_TIER)

        # Eagerly preload the strategy so router_runtime_status() reports truth.
        try:
            preload_strategy(config)
        except Exception:  # noqa: BLE001 — preload never blocks construction
            log.warning("squilla_router.preload_failed", exc_info=True)

    @classmethod
    def from_toml(cls, path: str, **kwargs: Any) -> Router:
        return cls(SquillaRouterConfig.load_toml(path), **kwargs)

    def _tier_model(self, tier: str) -> str:
        entry = self.config.tiers.get(tier) or self.config.tiers.get(DEFAULT_TEXT_TIER) or {}
        return str(entry.get("model") or "")

    def runtime_status(self) -> dict[str, Any]:
        return router_runtime_status()

    async def route(
        self,
        message: str,
        *,
        session_key: str,
        attachments: list[dict[str, Any]] | None = None,
        base_model: str | None = None,
        prev_assistant_text: str | None = None,
        prev_assistant_usage: dict[str, Any] | None = None,
        history_user_texts: list[str] | None = None,
        flags_text_override: str | None = None,
        session_spend_usd: float | None = None,
        session_cost_source: str = "unknown",
        material_estimated_tokens: int | None = None,
        hold_store: Any = None,
        semantic_message: str | None = None,
    ) -> RoutingResult:
        """Route one turn. Returns the final tier/model + controller decisions."""
        metadata: dict[str, Any] = {}
        if prev_assistant_text is not None:
            metadata["router_prev_assistant_text"] = prev_assistant_text
        if prev_assistant_usage is not None:
            metadata["router_prev_assistant_usage"] = prev_assistant_usage
        if history_user_texts is not None:
            metadata["router_history_user_texts"] = history_user_texts
        if flags_text_override is not None:
            metadata["router_flags_text_override"] = flags_text_override
        if session_spend_usd is not None:
            metadata["session_billed_cost_usd"] = session_spend_usd
            metadata["session_cost_source"] = session_cost_source
        if material_estimated_tokens is not None:
            metadata["material_estimated_tokens"] = material_estimated_tokens
        if hold_store is not None:
            metadata["router_control_hold_store"] = hold_store

        ctx = TurnContext(
            message=semantic_message if semantic_message is not None else message,
            session_key=session_key,
            model=base_model or self._default_model,
            config=self._app_config,
            attachments=list(attachments or []),
            metadata=metadata,
            raw_message=message,
            semantic_message=semantic_message,
        )
        await apply_squilla_router(ctx)

        # Persist the staged decision record (best-effort, no-op without writer).
        flush_router_decision(ctx.metadata)

        extra = ctx.metadata.get("routing_extra") or {}
        return RoutingResult(
            tier=ctx.metadata.get("routed_tier", ""),
            model=ctx.metadata.get("routed_model", ctx.model),
            confidence=float(ctx.metadata.get("routing_confidence", 0.0)),
            source=ctx.metadata.get("routing_source", ""),
            thinking_mode=ctx.metadata.get("thinking_mode"),
            thinking_level=ctx.metadata.get("thinking_level"),
            prompt_policy=ctx.metadata.get("prompt_policy"),
            prompt_hint=extra.get("prompt_hint"),
            routing_applied=bool(ctx.metadata.get("routing_applied", False)),
            routing_extra=dict(extra),
            metadata={
                k: v
                for k, v in ctx.metadata.items()
                if k
                in (
                    "routed_tier",
                    "routed_model",
                    "routing_applied",
                    "rollout_phase",
                    "applied_model",
                    "routing_confidence",
                    "routing_source",
                    "router_fallback_chain",
                    "savings_pct",
                    "savings_max_price_per_m",
                    "savings_routed_price_per_m",
                    "thinking_requested",
                    "thinking_level",
                    "thinking_mode",
                    "prompt_policy",
                    "routed_provider",
                    "image_route_reason",
                    "route_max_history_turns",
                    "router_budget_outcome",
                    "provider_mismatch_veto_applied",
                    "routing_history",
                    "routing_train_features",
                )
            },
        )
