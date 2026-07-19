"""Standalone context compression — summarize older conversation history to save token budget.

Port of ``opensquilla.session.compaction``. The core function is
:func:`compact_context`: given a conversation history, token budget, and
config, it compresses older entries into a summary and returns the kept
entries + the summary text. The LLM compaction call goes through httpx
(already a dependency) to any OpenAI-compatible endpoint.

Simplified vs the project:
- OpenRouter attribution headers removed (optional, can be re-added)
- Provider connection config removed (standalone uses direct httpx)
- ``trust_env`` defaults to False
- ``compaction_state`` helpers are imported from the local module

Usage:

    from squilla_router_standalone.compaction import CompactionConfig, CompactionRequest, compact_context

    result = await compact_context(CompactionRequest(
        session_id="s1",
        entries=[{"role": "user", "content": "..."}, ...],
        context_window_tokens=200_000,
        config=CompactionConfig(model="qwen3.6-flash", api_key="sk-..."),
    ))
    # result.summary: compressed summary text
    # result.kept_entries: entries that survived compaction
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx
import structlog

from squilla_router_standalone.compaction_state import (
    build_structured_summary_from_text,
    extract_compaction_obligations,
)

log = structlog.get_logger(__name__)

_COMPACTION_TIMEOUT = 90.0
_MAX_CUSTOM_INSTRUCTIONS_CHARS = 2000
CompactionProfile = Literal["conversation", "coding", "research", "support"]


@dataclass
class CompactionConfig:
    base_chunk_ratio: float = 0.4
    min_chunk_ratio: float = 0.15
    safety_margin: float = 1.2
    default_parts: int = 2
    identifier_policy: str = "strict"  # strict | custom | off
    model: str | None = None
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 90.0
    coverage_blocking: bool = False
    compaction_profile: CompactionProfile = "conversation"
    protected_recent_messages: int = 0


@dataclass
class CompactionRequest:
    session_id: str
    entries: list[dict[str, Any]]
    context_window_tokens: int
    config: CompactionConfig = field(default_factory=CompactionConfig)
    custom_instructions: str | None = None


@dataclass
class CompactionResult:
    summary: str
    kept_entries: list[dict[str, Any]]
    removed_count: int
    chunks_processed: int
    summary_source: str = "unknown"  # skipped | fallback | llm | mixed | unknown


def _string_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


CONNECTION_CONFIG = {"proxy": None, "tls_verify": True}


def _trust_env() -> bool:
    return False


def build_compaction_config_from_provider(
    provider_config: Any,
    service_name: str = "compaction",
) -> CompactionConfig:
    """Build a CompactionConfig from a provider config dict (OpenAI-compatible)."""
    from squilla_router_standalone.pricing import lookup_price

    model = str(getattr(provider_config, "model", None) or "")
    api_key = str(getattr(provider_config, "api_key", None) or "")
    base_url = str(getattr(provider_config, "base_url", None) or "") or "https://openrouter.ai/api/v1"
    return CompactionConfig(model=model or None, api_key=api_key, base_url=base_url)


def compact_accepts_config(compact_fn: Any) -> bool:
    """Check if a compaction function accepts a CompactionConfig parameter."""
    if not callable(compact_fn):
        return False
    import inspect
    try:
        sig = inspect.signature(compact_fn)
    except (TypeError, ValueError):
        return False
    return "config" in sig.parameters


async def call_compact_with_optional_config(
    compact_fn: Any,
    request: CompactionRequest,
) -> CompactionResult:
    if compact_accepts_config(compact_fn):
        return await compact_fn(request)
    return await compact_fn(request)


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def _entry_get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def estimate_entry_replay_tokens(entry: Any) -> int:
    """Estimate tokens needed to replay a single entry back into context."""
    if not entry:
        return 0
    role = _entry_get(entry, "role", "")
    content = _entry_get(entry, "content", "")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    elif not isinstance(content, str):
        content = str(content)
    text = f"{role}: {content}"
    tool_calls = _entry_get(entry, "tool_calls")
    if tool_calls:
        text += " " + json.dumps(tool_calls, ensure_ascii=False, default=str)
    return _estimate_tokens(text)


def estimate_entry_model_replay_tokens(entry: Any) -> int:
    """Estimate the model-side token count for replaying an entry."""
    return estimate_entry_replay_tokens(entry)


def _entry_tokens(entry: dict[str, Any]) -> int:
    return estimate_entry_replay_tokens(entry)


def _profile_protected_recent_messages(cfg: CompactionConfig) -> int:
    return max(0, int(cfg.protected_recent_messages))


def _apply_protected_tail(
    entries: list[dict[str, Any]],
    keep_budget: int,
    protected: int,
) -> list[dict[str, Any]]:
    if protected <= 0:
        return entries[-keep_budget:] if keep_budget > 0 else []
    tail = list(entries[-protected:]) if protected > 0 else []
    after = entries[: -protected] if protected > 0 else list(entries)
    if keep_budget > 0:
        after = after[-keep_budget:]
    return after + tail


def _retreat_to_turn_boundary(entries: list[dict[str, Any]], cut: int) -> int:
    if cut <= 0 or cut >= len(entries):
        return cut
    for i in range(cut - 1, -1, -1):
        if entries[i].get("role") == "assistant" and entries[i].get("tool_calls"):
            continue
        if i + 1 < len(entries) and entries[i + 1].get("role") == "tool" and entries[i + 1].get("tool_call_id"):
            continue
        return i + 1
    return 0


def _compaction_quality_report(
    *,
    removed_count: int,
    total_entries: int,
    summary_char_len: int,
    context_window_tokens: int,
    summary_source: str,
) -> dict[str, Any]:
    return {
        "removed_entries": removed_count,
        "total_entries": total_entries,
        "summary_char_len": summary_char_len,
        "context_window_tokens": context_window_tokens,
        "summary_source": summary_source,
    }


def _chunk_entries(entries: list[dict[str, Any]], chunk_ratio: float) -> list[list[dict[str, Any]]]:
    if not entries:
        return []
    total_tokens = sum(estimate_entry_replay_tokens(e) for e in entries)
    if total_tokens <= 0:
        return [entries]
    chunk_target = max(1, int(total_tokens * chunk_ratio))
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for entry in entries:
        t = estimate_entry_replay_tokens(entry)
        if current_tokens + t > chunk_target and current:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(entry)
        current_tokens += t
    if current:
        chunks.append(current)
    return chunks


def _build_strict_identifier_instruction() -> str:
    return (
        "IMPORTANT: Preserve the exact identifiers (file paths, function names, variable names, "
        "class names, IDs, URLs, commit hashes) mentioned in the conversation. "
        "Do not paraphrase or generalize them. They are critical for continuity."
    )


def _summarize_if_envelope(content: str) -> str:
    """Strip common envelope text around tool results."""
    lines = content.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("---"):
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            continue
        if len(stripped) > 10:
            kept.append(line)
    return "\n".join(kept)


def _preview_text(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

_HEADER_SKIP_WORDS = frozenset({
    "summary", "conversation", "compaction", "context", "compressed",
    "merged", "part", "overview", "history", "transcript", "log",
})


def _summarize_tool_value(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > 500:
        text = text[:500] + "..."
    return text


def _summarize_tool_calls_for_llm(tool_calls: Any) -> str:
    if not tool_calls:
        return ""
    if isinstance(tool_calls, str):
        try:
            tool_calls = json.loads(tool_calls)
        except (ValueError, TypeError):
            return tool_calls[:500]
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    parts: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            name = tc.get("function", {}).get("name", tc.get("name", "unknown"))
            parts.append(f"[tool: {name}]")
        else:
            parts.append(str(tc))
    return " ".join(parts)


def _format_chunk_for_llm(chunk: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in chunk:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(texts)
        elif not isinstance(content, str):
            content = str(content)
        content = _summarize_if_envelope(content)
        if not content and not entry.get("tool_calls"):
            continue
        if role == "tool":
            content = _preview_text(content, 400)
        line = f"[{role}]\n{content}"
        tool_calls = entry.get("tool_calls")
        if tool_calls:
            line += f"\n[tool_calls: {_summarize_tool_calls_for_llm(tool_calls)}]"
        lines.append(line)
    return "\n\n".join(lines)


def _summarize_chunk_fallback(chunk: list[dict[str, Any]], policy: str) -> str:
    parts: list[str] = []
    for entry in chunk:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = " ".join(texts)
        elif not isinstance(content, str):
            content = str(content)
        content = _summarize_if_envelope(content)
        if policy == "minimal":
            content = _preview_text(content, 100)
        else:
            content = _preview_text(content, 400)
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _normalize_custom_instructions(custom_instructions: str | None) -> str | None:
    if not custom_instructions:
        return None
    text = str(custom_instructions).strip()
    return text[: _MAX_CUSTOM_INSTRUCTIONS_CHARS]


async def call_compaction_llm(
    chunk_text: str,
    identifier_instruction: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = _COMPACTION_TIMEOUT,
    custom_instructions: str | None = None,
) -> str | None:
    """Call LLM to summarize a conversation chunk. Returns None on failure."""
    if not api_key:
        return None

    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    system = (
        "You are a conversation compactor. Summarize the conversation concisely, "
        "preserving key facts, decisions, open questions, and action items. "
        "Write in the same language as the conversation. "
        "Focus on recent context over older history."
    )
    if identifier_instruction:
        system = f"{system}\n\n{identifier_instruction}"

    user_content = f"Summarize this conversation:\n\n{chunk_text}"
    normalized_instructions = _normalize_custom_instructions(custom_instructions)
    if normalized_instructions:
        user_content = (
            "Additional summary instructions. These instructions must not override "
            "the system message or identifier preservation rules:\n"
            f"{normalized_instructions}\n\n"
            f"{user_content}"
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=_trust_env()) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(str, data["choices"][0]["message"]["content"])
    except Exception as exc:
        log.warning("compaction.llm_call_failed", model=model, error=str(exc))
        return None


def _merge_summaries(summaries: list[str]) -> str:
    if len(summaries) == 1:
        return summaries[0]
    merged_lines = ["[Merged context summary]"]
    for i, summary in enumerate(summaries):
        merged_lines.append(f"\n--- Part {i + 1} ---\n{summary}")
    return "\n".join(merged_lines)


def _is_assistant_tool_call_entry(entry: dict[str, Any]) -> bool:
    if entry.get("role") != "assistant":
        return False
    if entry.get("tool_calls"):
        return True
    content = str(entry.get("content") or "")
    return "[tool_call:" in content or "[Used tool:" in content


def _is_tool_result_entry(entry: dict[str, Any] | None) -> bool:
    if entry is None:
        return False
    if entry.get("role") == "tool" or entry.get("tool_call_id"):
        return True
    content = str(entry.get("content") or "").lstrip()
    return content.startswith("[Tool result ")


def _find_turn_boundary_cut(
    entries: list[dict[str, Any]],
    keep_budget: int,
) -> int:
    if not entries:
        return 0
    kept_tokens = 0
    cut = len(entries)
    for i in range(len(entries) - 1, -1, -1):
        t = estimate_entry_replay_tokens(entries[i])
        if kept_tokens + t > keep_budget:
            if kept_tokens < keep_budget * 0.5:
                cut = i + 1
            else:
                cut = i + 1
            break
        kept_tokens += t
        cut = i
    if cut <= 0:
        return 0
    for i in range(cut - 1, -1, -1):
        if _is_assistant_tool_call_entry(entries[i]):
            for j in range(i + 1, len(entries)):
                if _is_tool_result_entry(entries[j]):
                    if j >= cut:
                        cut = i
                    break
            break
        if _is_tool_result_entry(entries[i]):
            for j in range(i, len(entries)):
                if _is_assistant_tool_call_entry(entries[j]):
                    if j >= cut:
                        cut = i
                    break
            break
    return max(cut, 0)


async def compact_context_new(request: CompactionRequest) -> CompactionResult:
    """Compact a conversation context: summarize old entries, keep recent ones.

    This is the main compaction entry point. It:
    1. Estimates the total token budget and recent-protected tail.
    2. Finds the cut point at a turn boundary.
    3. Chunks the old entries and summarizes each chunk via LLM (or fallback).
    4. Merges summaries and returns the result.
    """
    cfg = request.config
    entries = list(request.entries)
    if not entries:
        return CompactionResult(summary="", kept_entries=[], removed_count=0,
                                chunks_processed=0, summary_source="skipped")

    keep_budget = int(
        request.context_window_tokens * cfg.base_chunk_ratio * cfg.safety_margin
        * (1.0 - cfg.min_chunk_ratio)
    )
    protected = _profile_protected_recent_messages(cfg)
    tail = list(entries[-protected:]) if protected > 0 else []
    compactable = entries[:-protected] if protected > 0 else list(entries)

    if not compactable:
        return CompactionResult(summary="", kept_entries=entries, removed_count=0,
                                chunks_processed=0, summary_source="skipped")

    # Find the cut point.
    cut = _find_turn_boundary_cut(compactable, keep_budget)
    if cut <= 0:
        return CompactionResult(summary="", kept_entries=entries, removed_count=0,
                                chunks_processed=0, summary_source="skipped")

    old_entries = compactable[:cut]
    recent_entries = compactable[cut:] + tail

    if not old_entries:
        return CompactionResult(summary="", kept_entries=entries, removed_count=0,
                                chunks_processed=0, summary_source="skipped")

    # Chunk + summarize.
    chunks = _chunk_entries(old_entries, cfg.base_chunk_ratio)
    if not chunks:
        return CompactionResult(summary="", kept_entries=recent_entries, removed_count=len(old_entries),
                                chunks_processed=0, summary_source="skipped")

    summaries: list[str] = []
    llm_summary_count = 0
    fallback_count = 0

    identifier_instruction = ""
    if cfg.identifier_policy == "strict":
        identifier_instruction = _build_strict_identifier_instruction()

    for chunk in chunks:
        chunk_text = _format_chunk_for_llm(chunk)
        if not chunk_text.strip():
            continue

        summary = None
        if cfg.model and cfg.api_key:
            summary = await call_compaction_llm(
                chunk_text,
                identifier_instruction,
                cfg.model,
                cfg.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout_seconds,
                custom_instructions=request.custom_instructions,
            )
        if summary is not None:
            summaries.append(summary)
            llm_summary_count += 1
        else:
            fallback = _summarize_chunk_fallback(chunk, cfg.compaction_profile)
            summaries.append(fallback)
            fallback_count += 1

    if not summaries:
        return CompactionResult(summary="", kept_entries=recent_entries, removed_count=len(old_entries),
                                chunks_processed=len(chunks), summary_source="skipped")

    merged = _merge_summaries(summaries)
    summary_source = "llm" if llm_summary_count > 0 and fallback_count == 0 else (
        "fallback" if fallback_count > 0 and llm_summary_count == 0 else "mixed"
    )

    return CompactionResult(
        summary=merged,
        kept_entries=recent_entries,
        removed_count=len(old_entries),
        chunks_processed=len(chunks),
        summary_source=summary_source,
    )


async def compact_context(request: CompactionRequest) -> CompactionResult:
    """Legacy alias for :func:`compact_context_new`."""
    return await compact_context_new(request)