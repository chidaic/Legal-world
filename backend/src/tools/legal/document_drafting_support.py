"""Minimal helpers for legal document drafting tool payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw[3:-3].strip()
        if inner.lower().startswith("markdown"):
            inner = inner[8:].strip()
        elif inner.lower().startswith("text"):
            inner = inner[4:].strip()
        elif inner.lower().startswith("json"):
            inner = inner[4:].strip()
        return inner.strip()
    return raw


def extract_document_body(
    message_text: Any,
    *,
    document_title: str,
    end_marker: str = "",
) -> str:
    content = strip_code_fences(str(message_text or ""))
    if not content.strip():
        return ""

    normalized_end_marker = str(end_marker or "").strip()
    if not normalized_end_marker:
        return ""

    marker_index = content.rfind(normalized_end_marker)
    if marker_index < 0:
        return ""

    search_space = content[:marker_index]
    if not search_space.strip():
        return ""

    title_pattern = re.compile(
        rf"(^|\n)\s*{re.escape(str(document_title or '').strip())}\s*(?=\n|$)",
        re.MULTILINE,
    )
    matches = list(title_pattern.finditer(search_space))
    if not matches:
        return ""
    start_index = matches[-1].start()
    if search_space[start_index : start_index + 1] == "\n":
        start_index += 1
    extracted = search_space[start_index:]
    return extracted.strip()


def extract_json_payload(raw_result: Any) -> Dict[str, Any]:
    if isinstance(raw_result, dict):
        return raw_result

    raw = str(raw_result or "").strip()
    if not raw:
        raise ValueError("Empty tool result.")

    code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if code_block_match:
        raw = code_block_match.group(1).strip()

    return json.loads(raw)


def resolve_case_output_dir(agent: Any) -> Path:
    scenario_data = getattr(agent, "scenario_data", {}) or {}
    explicit = str(scenario_data.get("case_output_dir", "") or "").strip()
    if explicit:
        path = Path(explicit).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    trace_recorder = getattr(agent, "_simlaw_trace_recorder", None)
    trace_case_output_dir = str(getattr(trace_recorder, "case_output_dir", "") or "").strip()
    if trace_case_output_dir:
        path = Path(trace_case_output_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    storage = getattr(agent, "storage", None)
    case_id = str(
        getattr(agent, "current_handling_case", None) or scenario_data.get("case_id", "") or ""
    ).strip()
    base_dir = getattr(storage, "base_dir", None)
    if base_dir and case_id:
        path = (Path(base_dir) / "output" / case_id).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    return Path.cwd().resolve()


__all__ = [
    "extract_document_body",
    "extract_json_payload",
    "resolve_case_output_dir",
    "strip_code_fences",
]
