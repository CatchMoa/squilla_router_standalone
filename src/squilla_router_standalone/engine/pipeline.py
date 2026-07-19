"""Lightweight turn context for the standalone router step.

Replaces ``opensquilla.engine.pipeline.TurnContext``. The project's TurnContext
carries the full agent turn state; the router step only reads a handful of
fields (message, attachments, metadata, config, session_key, model) and writes
back ``model`` / ``message`` / ``metadata``. This dataclass exposes exactly
those, plus a ``semantic_message`` resolution that mirrors the step's fallback
chain (semantic_message -> raw_message -> message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LlmConfig:
    """Minimal LLM config: the router step only reads ``provider``."""

    provider: str = ""


@dataclass
class StandaloneConfig:
    """Minimal app config: ``squilla_router`` + ``llm``.

    The real ``opensquilla`` config has dozens of sub-sections; only the two
    the router step touches are surfaced here. ``squilla_router`` is the
    :class:`~squilla_router_standalone.config.SquillaRouterConfig`.
    """

    squilla_router: Any = None  # SquillaRouterConfig
    llm: LlmConfig = field(default_factory=LlmConfig)


@dataclass
class TurnContext:
    """Mutable per-turn context consumed by ``apply_squilla_router``."""

    message: str
    session_key: str
    model: str
    config: StandaloneConfig
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_message: str | None = None
    semantic_message: str | None = None

    def resolved_semantic_message(self) -> str:
        """Mirror the step's fallback: semantic -> raw -> message."""
        if self.semantic_message is not None:
            return self.semantic_message
        if self.raw_message is not None:
            return self.raw_message
        return self.message
