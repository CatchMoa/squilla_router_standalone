"""Standalone squilla_router config.

Ports ``opensquilla.gateway.config.SquillaRouterConfig`` and
``RouterBudgetConfig``. The ``tier_profile`` / packaged preset registry is
dropped: tiers are declared explicitly in TOML (the proxy's setup wizard
already does this). A ``context_window`` field is added to each tier so the
config-driven capability gate has definite facts without a model catalog.

Loadable from TOML via :meth:`SquillaRouterConfig.load_toml`, from a dict, or
constructed directly. ``env_prefix=SQUILLA_ROUTER_`` mirrors the project's
``OPENSQUILLA_SQUILLA_ROUTER_`` env hook (shorter namespace for standalone use).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from squilla_router_standalone.router_tiers import (
    DEFAULT_TEXT_TIER,
    normalize_text_tier,
    normalize_tier_mapping,
)


class RouterBudgetConfig(BaseSettings):
    """Additive, opt-in per-session spend gate for the router."""

    model_config = SettingsConfigDict(extra="ignore", env_prefix="SQUILLA_ROUTER_BUDGET_")

    action: Literal["off", "warn", "cap"] = "warn"
    limit_usd: float | None = None
    cap_tier: str | None = None
    include_next_turn_estimate: bool = False


class SquillaRouterConfig(BaseSettings):
    """Standalone squilla router config (no preset registry)."""

    model_config = SettingsConfigDict(
        env_prefix="SQUILLA_ROUTER_",
        extra="ignore",
    )

    enabled: bool = True
    auto_thinking: bool = True
    rollout_phase: str = "full"  # "observe" | "prompt_only" | "full"
    strategy: str = "v4_phase3"
    cross_provider_tiers: bool = False
    tier_provider_mismatch: Literal["route", "veto"] = "route"
    tiers: dict = Field(default_factory=lambda: _default_tiers())
    default_tier: str = DEFAULT_TEXT_TIER
    confidence_threshold: float = 0.5
    confidence_high_tier_margin: float = Field(default=0.05, ge=0.0)
    v4_bundle_dir: str | None = None
    v4_use_aux_head: bool | None = True
    routing_timeout_seconds: float = Field(default=5.0, gt=0.0)
    kv_cache_anti_downgrade_enabled: bool = True
    kv_cache_anti_downgrade_window_seconds: int = 600
    complaint_upgrade_enabled: bool = True
    complaint_upgrade_steps: int = 1
    complaint_upgrade_max_chars: int = 160
    require_router_runtime: bool = False  # standalone default: degrade, not fail
    calibration_enabled: bool = False
    budget: RouterBudgetConfig = Field(default_factory=RouterBudgetConfig)
    estimated_output_savings_pct: float = 0.03
    upgrade_to_c3_compaction_enabled: bool = True
    vision_history_lookback_turns: int = Field(default=8, ge=0)
    context_window_tokens: int = 200_000
    # Operator-declared per-model USD prices (USD per 1M input tokens).
    prices: dict = Field(default_factory=dict)

    @field_validator("rollout_phase")
    @classmethod
    def _check_rollout(cls, value: str) -> str:
        if value not in ("observe", "prompt_only", "full"):
            raise ValueError("rollout_phase must be one of: observe, prompt_only, full")
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_tiers(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if "default_tier" in values:
            values["default_tier"] = normalize_text_tier(values.get("default_tier")) or values.get(
                "default_tier"
            )
        if isinstance(values.get("tiers"), dict):
            values["tiers"] = normalize_tier_mapping(values["tiers"])
        return values

    @classmethod
    def load_toml(cls, path: str | None = None) -> SquillaRouterConfig:
        """Load config from a TOML file with a ``[squilla_router]`` section."""
        import pathlib
        import tomllib

        if path is None:
            return cls()
        raw = tomllib.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        section = raw.get("squilla_router", raw)
        return cls(**section) if isinstance(section, dict) else cls()


def _default_tiers() -> dict:
    """Default tier config (Claude family; override via TOML).

    Text tiers are non-vision (matches the project default); only
    ``image_model`` carries ``supports_image=true`` so the image bypass
    routes image turns there.
    """
    return {
        "c0": {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "description": "简单任务：快/便宜",
            "supports_image": False,
            "context_window": 200_000,
        },
        "c1": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "description": "默认任务：平衡",
            "supports_image": False,
            "context_window": 200_000,
        },
        "c2": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "description": "中等复杂度：标准",
            "supports_image": False,
            "context_window": 200_000,
        },
        "c3": {
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "description": "最复杂任务：最强推理",
            "supports_image": False,
            "context_window": 200_000,
        },
        "image_model": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "description": "图像模型：视觉问答",
            "supports_image": True,
            "image_only": True,
            "context_window": 200_000,
        },
    }
