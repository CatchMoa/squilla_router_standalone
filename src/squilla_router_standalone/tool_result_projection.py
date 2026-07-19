"""Standalone tool result projection — truncate large tool results to save tokens.

Pure function: takes a tool result string, returns a truncated ``[tool_result_projection]``
block with head/tail preview, sha256 fingerprint, and handle for retrieval.

The projection replaces the original tool result content in the LLM request
with a compact summary, keeping only a preview (head ~65% + tail ~35% of
``max_preview_chars``). The full result is stored separately (caller's
responsibility) and retrievable via the handle.

Usage:

    from squilla_router_standalone.tool_result_projection import ProjectionConfig, project_tool_result

    config = ProjectionConfig(max_preview_chars=60_000)
    projected = project_tool_result(content, tool_name="read", tool_use_id="tu_abc", config=config)
    # projected.block: "[tool_result_projection]\\ntool: read\\n..."
    # projected.full_result_stored: True/False
    # projected.omitted_chars: 12345
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectionConfig:
    """Configuration for tool result projection.

    ``max_preview_chars`` is the maximum characters to keep in the preview
    (head ~65% + tail ~35%). When ``max_preview_chars <= 0``, the preview
    is empty and only the metadata block is sent. The caller is responsible
    for storing the full result and making it retrievable via the ``handle``.
    """

    max_preview_chars: int = 60_000
    store_dir: str | None = None


@dataclass
class ProjectionResult:
    """Result of projecting a tool result."""

    block: str
    omitted_chars: int
    full_result_stored: bool
    handle: str | None = None
    sha256: str = ""


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _store_snapshot(content: str, *, tool_use_id: str, tool_name: str, store_dir: str | None) -> str | None:
    """Store the full tool result to disk and return a handle. None = not stored."""
    import os
    import time
    from pathlib import Path

    if not store_dir:
        return None
    root = Path(store_dir)
    root.mkdir(parents=True, exist_ok=True)
    digest = _sha256(content)
    ts = int(time.time() * 1000)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_name)[:32]
    filename = f"{ts}_{safe_name}_{tool_use_id}_{digest[:12]}.txt"
    path = root / filename
    try:
        path.write_text(content, encoding="utf-8")
        return filename
    except OSError:
        return None


def _result_retrieve_hint() -> str:
    return (
        "To retrieve the full result, call the tool_result_retrieve tool with "
        "the handle from the `tool_result_handle` field above.\n"
    )


def _result_search_hints(content: str) -> str:
    max_hints = 3
    lines = content.split("\n")
    hints: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 80:
            continue
        if stripped.startswith("```") or stripped in ("---",):
            continue
        hints.append(stripped)
        if len(hints) >= max_hints:
            break
    if hints:
        return "search_hints:\n  " + "\n  ".join(hints) + "\n"
    return ""


def project_tool_result(
    content: str,
    *,
    tool_name: str,
    tool_use_id: str,
    config: ProjectionConfig | None = None,
    reason: str = "tool_result_too_large",
    store_full: bool = True,
) -> ProjectionResult:
    """Project a tool result into a compact ``[tool_result_projection]`` block.

    *content* is the raw tool result. *tool_name* and *tool_use_id* identify
    it for retrieval. *config* controls the preview size. *reason* is logged
    in the projection block for debugging. *store_full* controls whether the
    full result is saved to disk (when ``store_dir`` is set in config).
    """
    cfg = config or ProjectionConfig()
    max_preview_chars = max(0, int(cfg.max_preview_chars))
    if max_preview_chars > 0:
        max_preview_chars = max(1, min(max_preview_chars, 4_000))

    digest = _sha256(content)
    handle = None
    if store_full:
        handle = _store_snapshot(
            content,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            store_dir=cfg.store_dir,
        )

    handle_line = f"tool_result_handle: {handle}\n" if handle is not None else ""
    retrieve_hint = _result_retrieve_hint() if handle is not None else ""
    search_hints = _result_search_hints(content) if handle is not None else ""

    if max_preview_chars <= 0:
        head = ""
        tail = ""
    elif len(content) <= max_preview_chars:
        head = content
        tail = ""
    else:
        head_chars = max(1, int(max_preview_chars * 0.65))
        tail_chars = max(0, max_preview_chars - head_chars)
        head = content[:head_chars]
        tail = content[-tail_chars:] if tail_chars else ""

    omitted = max(0, len(content) - len(head) - len(tail))
    block = (
        "[tool_result_projection]\n"
        f"tool: {tool_name}\n"
        f"tool_use_id: {tool_use_id}\n"
        f"original_chars: {len(content)}\n"
        f"sha256: {digest}\n"
        f"{handle_line}"
        f"{retrieve_hint}"
        f"{search_hints}"
        f"omitted_chars: {omitted}\n"
        f"preview_complete: {str(omitted == 0).lower()}\n"
        f"reason: {reason}.\n"
        f"head:\n{head}"
    )
    if tail:
        block += f"\ntail:\n{tail}"

    return ProjectionResult(
        block=block,
        omitted_chars=omitted,
        full_result_stored=handle is not None,
        handle=handle,
        sha256=digest,
    )