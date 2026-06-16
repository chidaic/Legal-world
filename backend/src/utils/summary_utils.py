"""Helpers for storing and rendering case-history summary entries."""

from __future__ import annotations

import re
from typing import Any, Iterable


_PROMPT_EXCLUDED_STAGES = {
}


_STAGE_ENTRY_RE = re.compile(r"^\[(?P<stage>[^\]]+)\]\s*(?P<summary>.+)$", re.DOTALL)
_STAGE_NAME_ALIASES = {
    "法庭调查": "民事一审",
}


def normalize_stage_name(stage_name: Any) -> str | None:
    """Normalize persisted stage names for display and dedupe."""
    stage = str(stage_name or "").strip()
    if not stage:
        return None
    return _STAGE_NAME_ALIASES.get(stage, stage)


def split_stage_summary_entry(entry: Any) -> tuple[str | None, str]:
    """Parse either a text or dict summary entry into ``(stage, summary)``."""
    if isinstance(entry, dict):
        stage = normalize_stage_name(entry.get("stage"))
        summary = str(entry.get("summary", "") or "").strip()
        return stage, summary

    text = str(entry or "").strip()
    if not text:
        return None, ""

    match = _STAGE_ENTRY_RE.match(text)
    if match:
        stage = normalize_stage_name(match.group("stage"))
        summary = match.group("summary").strip()
        return stage, summary

    return None, text


def format_stage_summary_text(stage: str | None, summary: str) -> str:
    """Format a summary as ``[阶段] 内容`` when a stage name is available."""
    clean_summary = str(summary or "").strip()
    if not clean_summary:
        return ""

    clean_stage = normalize_stage_name(stage)
    return f"[{clean_stage}] {clean_summary}" if clean_stage else clean_summary


def dedupe_stage_summary_text(entries: Iterable[Any]) -> list[str]:
    """Keep one latest summary per stage while preserving stage order."""
    order: list[tuple[str, str]] = []
    latest: dict[tuple[str, str], str] = {}

    for entry in entries or []:
        stage, summary = split_stage_summary_entry(entry)
        if not summary:
            continue

        key = ("stage", stage) if stage else ("text", summary)
        if key not in latest:
            order.append(key)
        latest[key] = format_stage_summary_text(stage, summary)

    return [latest[key] for key in order if latest.get(key)]


def filter_prompt_stage_summary_text(entries: Iterable[Any]) -> list[str]:
    """Return only summaries that should be reused in later prompts."""
    filtered: list[str] = []
    for entry in dedupe_stage_summary_text(entries):
        stage, summary = split_stage_summary_entry(entry)
        if not summary:
            continue
        if stage and normalize_stage_name(stage) in _PROMPT_EXCLUDED_STAGES:
            continue
        filtered.append(format_stage_summary_text(stage, summary))
    return filtered


def upsert_stage_summary_text(
    entries: Iterable[Any],
    stage_name: str,
    summary: str,
) -> list[str]:
    """Insert or replace a stage summary in text-list storage."""
    order: list[tuple[str, str]] = []
    latest: dict[tuple[str, str], str] = {}

    for entry in entries or []:
        stage, existing_summary = split_stage_summary_entry(entry)
        if not existing_summary:
            continue

        key = ("stage", stage) if stage else ("text", existing_summary)
        if key not in latest:
            order.append(key)
        latest[key] = format_stage_summary_text(stage, existing_summary)

    clean_stage = normalize_stage_name(stage_name)
    clean_summary = str(summary or "").strip()
    if clean_summary:
        key = ("stage", clean_stage) if clean_stage else ("text", clean_summary)
        if key not in latest:
            order.append(key)
        latest[key] = format_stage_summary_text(clean_stage, clean_summary)

    return [latest[key] for key in order if latest.get(key)]


def dedupe_stage_summary_dicts(entries: Iterable[Any]) -> list[dict[str, str]]:
    """Keep one latest summary per stage in dict-list storage."""
    order: list[tuple[str, str]] = []
    latest: dict[tuple[str, str], dict[str, str]] = {}

    for entry in entries or []:
        stage, summary = split_stage_summary_entry(entry)
        if not summary:
            continue

        key = ("stage", stage) if stage else ("text", summary)
        if key not in latest:
            order.append(key)

        payload = {"summary": summary}
        if stage:
            payload["stage"] = stage
        latest[key] = payload

    return [latest[key] for key in order if latest.get(key)]


def upsert_stage_summary_dicts(
    entries: Iterable[Any],
    stage_name: str,
    summary: str,
) -> list[dict[str, str]]:
    """Insert or replace a stage summary in dict-list storage."""
    order: list[tuple[str, str]] = []
    latest: dict[tuple[str, str], dict[str, str]] = {}

    for entry in entries or []:
        stage, existing_summary = split_stage_summary_entry(entry)
        if not existing_summary:
            continue

        key = ("stage", stage) if stage else ("text", existing_summary)
        if key not in latest:
            order.append(key)

        payload = {"summary": existing_summary}
        if stage:
            payload["stage"] = stage
        latest[key] = payload

    clean_stage = normalize_stage_name(stage_name)
    clean_summary = str(summary or "").strip()
    if clean_summary:
        key = ("stage", clean_stage) if clean_stage else ("text", clean_summary)
        if key not in latest:
            order.append(key)

        payload = {"summary": clean_summary}
        if clean_stage:
            payload["stage"] = clean_stage
        latest[key] = payload

    return [latest[key] for key in order if latest.get(key)]


def has_stage_summary(entries: Iterable[Any], stage_name: str) -> bool:
    """Return True when a stage summary exists in either storage format."""
    target = normalize_stage_name(stage_name)
    if not target:
        return False

    for entry in entries or []:
        stage, summary = split_stage_summary_entry(entry)
        if stage == target and summary:
            return True
    return False
