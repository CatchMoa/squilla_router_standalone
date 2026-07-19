"""Structured compaction state helpers.

Port of ``opensquilla.session.compaction_state``. Pure data models + helpers
for building/extracting structured compaction summaries. No I/O, no external
deps beyond pydantic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field


class StructuredCompactionSummary(BaseModel):
    """Portable, inspectable task state produced by local compaction."""

    schema_version: int = 1
    user_goal: str = ""
    current_status: str = ""
    next_action: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    open_steps: list[str] = Field(default_factory=list)
    files_and_artifacts: list[dict[str, str]] = Field(default_factory=list)
    tool_results_to_remember: list[dict[str, str]] = Field(default_factory=list)
    decisions_and_rationale: list[dict[str, str]] = Field(default_factory=list)
    known_failures: list[dict[str, str]] = Field(default_factory=list)
    important_identifiers: list[str] = Field(default_factory=list)
    constraints_and_preferences: list[str] = Field(default_factory=list)
    do_not_repeat: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    critical_carry_forward: list[str] = Field(default_factory=list)
    source_coverage: dict[str, Any] = Field(default_factory=dict)


class CompactionObligation(BaseModel):
    """Small continuity fact that should survive transcript compaction."""

    kind: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _compact_instructions(text: str) -> str:
    """Remove repeated empty lines, strip lines."""
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in cleaned.split("\n") if line.strip())


def extract_compaction_obligations(
    text: str,
    *,
    enforce_source: bool = False,
    source_filter: str = "",
) -> list[CompactionObligation]:
    """Extract structured compaction obligations from free-text summary.

    Each obligation is a small continuity fact (identifiers, decisions,
    carry-forward items) that should survive compaction.
    """
    obligations: list[CompactionObligation] = []
    seen = set()
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 8:
            continue
        # Deduplicate by normalized text.
        key = stripped.lower().strip(".:;!?")
        if key in seen:
            continue
        seen.add(key)
        # Classify by prefix.
        kind = "fact"
        if any(stripped.startswith(p) for p in ("identifier:", "ID:", "id:", "path:")):
            kind = "identifier"
        elif any(stripped.startswith(p) for p in ("decision:", "decided:", "chose:")):
            kind = "decision"
        elif any(stripped.startswith(p) for p in ("carry:", "carry_forward:", "todo:")):
            kind = "carry_forward"
        elif any(stripped.startswith(p) for p in ("question:", "open:", "unresolved:")):
            kind = "open_question"
        obligations.append(
            CompactionObligation(
                kind=kind,
                payload={"text": stripped},
                metadata={"source_line": line},
            )
        )
    if enforce_source and source_filter:
        obligations = [
            o for o in obligations if source_filter in str(o.payload.get("text", ""))
        ]
    return obligations


def build_structured_summary_from_text(
    text: str,
    *,
    source: str = "text",
    schema_version: int = 1,
) -> StructuredCompactionSummary:
    """Build a StructuredCompactionSummary from free-text compaction output.

    This is a best-effort parser: it extracts structured fields from the
    text by scanning for known section headers. The text is the output of
    the LLM compaction call.
    """
    summary = StructuredCompactionSummary(schema_version=schema_version)
    if not text:
        return summary

    summary.source_coverage["source"] = source
    summary.source_coverage["char_len"] = len(text)

    sections = _parse_sections(text)
    for header, content in sections.items():
        cleaned = _compact_instructions(content)
        _populate_field(summary, header, cleaned)

    return summary


_SECTION_HEADERS = {
    "user_goal": ("user goal", "goal", "user goal/objective"),
    "current_status": ("current status", "status", "progress"),
    "next_action": ("next action", "next step", "next"),
    "completed_steps": ("completed steps", "completed", "done", "accomplished"),
    "open_steps": ("open steps", "pending", "remaining", "todo"),
    "files_and_artifacts": (
        "files and artifacts", "files", "artifacts", "artifacts/files",
        "created files", "modified files",
    ),
    "tool_results_to_remember": (
        "tool results to remember", "tool results", "important results",
    ),
    "decisions_and_rationale": (
        "decisions and rationale", "decisions", "key decisions",
    ),
    "known_failures": ("known failures", "failures", "issues", "problems"),
    "important_identifiers": (
        "important identifiers", "identifiers", "ids", "paths",
        "important ids",
    ),
    "constraints_and_preferences": (
        "constraints and preferences", "constraints", "preferences",
    ),
    "do_not_repeat": ("do not repeat", "don't repeat", "avoid", "pitfalls"),
    "unresolved_questions": (
        "unresolved questions", "questions", "open questions",
    ),
    "critical_carry_forward": (
        "critical carry forward", "carry forward", "carry_over",
        "critical items",
    ),
}


def _parse_sections(text: str) -> dict[str, str]:
    """Split text into sections by known header lines."""
    sections: dict[str, str] = {}
    current_header = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip().lower().rstrip(":")
        matched = False
        for field, aliases in _SECTION_HEADERS.items():
            if stripped in aliases:
                if current_header:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = field
                current_lines = []
                matched = True
                break
        if not matched:
            if current_header:
                current_lines.append(line)
            else:
                current_lines.append(line)

    if current_header:
        sections[current_header] = "\n".join(current_lines).strip()
    return sections


def _populate_field(summary: StructuredCompactionSummary, field: str, content: str) -> None:
    """Populate one field of the summary from parsed section content."""
    if not content:
        return
    if field in ("user_goal", "current_status", "next_action"):
        setattr(summary, field, content[:2000])
    elif field in (
        "completed_steps", "open_steps", "constraints_and_preferences",
        "do_not_repeat", "unresolved_questions", "critical_carry_forward",
    ):
        items = [line.strip("-* ").strip() for line in content.split("\n") if line.strip()]
        getattr(summary, field).extend(items[:50])
    elif field in ("files_and_artifacts", "tool_results_to_remember"):
        items = [line.strip("-* ").strip() for line in content.split("\n") if line.strip()]
        for item in items[:30]:
            getattr(summary, field).append({"path": item, "description": ""})
    elif field == "decisions_and_rationale":
        items = [line.strip("-* ").strip() for line in content.split("\n") if line.strip()]
        for item in items[:30]:
            getattr(summary, field).append({"decision": item, "rationale": ""})
    elif field == "known_failures":
        items = [line.strip("-* ").strip() for line in content.split("\n") if line.strip()]
        for item in items[:20]:
            getattr(summary, field).append({"issue": item, "resolution": ""})
    elif field == "important_identifiers":
        items = [line.strip("-* ").strip() for line in content.split("\n") if line.strip()]
        getattr(summary, field).extend(items[:30])