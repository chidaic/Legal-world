"""Optional document comparison tool for drafted legal documents."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from camel.toolkits import FunctionTool


DOCUMENT_COMPARE_TOOL_NAME = "compare_documents"
SECTION_TITLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("title", ("民事起诉状", "民事答辩状", "民事上诉状", "民事上诉答辩状")),
    ("requests", ("诉讼请求", "上诉请求")),
    ("defense_opinions", ("答辩意见",)),
    ("facts_and_reasons", ("事实和理由", "事实与理由", "上诉理由", "答辩理由")),
    ("evidence", ("证据", "新证据", "新证据，证人证言等(如有)", "新证据，证人证言等（如有）")),
    ("closing", ("此致",)),
)
SENTENCE_SPLIT_RE = re.compile(r"[。\n；;]+")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_line(value: Any) -> str:
    return _normalize_text(value).rstrip("：:")


def _normalize_section_key(line: str) -> str | None:
    normalized = _normalize_line(line)
    for key, patterns in SECTION_TITLE_PATTERNS:
        if normalized in {_normalize_line(pattern) for pattern in patterns}:
            return key
    return None


def _split_sections(document_text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key = "body"
    sections[current_key] = []

    for raw_line in str(document_text or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        section_key = _normalize_section_key(line)
        if section_key is not None:
            current_key = section_key
            sections.setdefault(current_key, [])
            continue
        sections.setdefault(current_key, []).append(line)

    return {key: "\n".join(lines).strip() for key, lines in sections.items() if "\n".join(lines).strip()}


def _split_points(text: str) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()
    for chunk in SENTENCE_SPLIT_RE.split(str(text or "")):
        normalized = _normalize_text(chunk).strip("，, ")
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        points.append(normalized)
    return points


def _take_top(items: list[str], limit: int = 5) -> list[str]:
    return items[:limit]


class DocumentCompareTool:
    """Compare two legal documents and summarize section-level deltas."""

    def compare_documents(
        self,
        left_document: str,
        right_document: str,
        left_label: str = "document_a",
        right_label: str = "document_b",
    ) -> str:
        left_text = str(left_document or "").strip()
        right_text = str(right_document or "").strip()
        if not left_text or not right_text:
            raise ValueError("left_document and right_document are required.")

        left_label = str(left_label or "document_a").strip() or "document_a"
        right_label = str(right_label or "document_b").strip() or "document_b"

        left_sections = _split_sections(left_text)
        right_sections = _split_sections(right_text)
        all_sections = sorted(set(left_sections) | set(right_sections))

        section_diffs: list[dict[str, Any]] = []
        focus_shifts: list[str] = []
        for section_name in all_sections:
            left_section = left_sections.get(section_name, "")
            right_section = right_sections.get(section_name, "")
            similarity = round(
                SequenceMatcher(None, _normalize_text(left_section), _normalize_text(right_section)).ratio(),
                4,
            )

            left_points = _split_points(left_section)
            right_points = _split_points(right_section)
            right_point_set = set(right_points)
            left_point_set = set(left_points)
            shared_points = [point for point in left_points if point in right_point_set]
            left_only_points = [point for point in left_points if point not in right_point_set]
            right_only_points = [point for point in right_points if point not in left_point_set]

            if similarity < 0.65 and (left_section or right_section):
                focus_shifts.append(section_name)

            section_diffs.append(
                {
                    "section": section_name,
                    "similarity": similarity,
                    "shared_points": _take_top(shared_points, limit=3),
                    f"{left_label}_only_points": _take_top(left_only_points, limit=3),
                    f"{right_label}_only_points": _take_top(right_only_points, limit=3),
                }
            )

        payload = {
            "tool_name": DOCUMENT_COMPARE_TOOL_NAME,
            "status": "ok",
            "left_label": left_label,
            "right_label": right_label,
            "overall_similarity": round(
                SequenceMatcher(None, _normalize_text(left_text), _normalize_text(right_text)).ratio(),
                4,
            ),
            "shared_sections": sorted(set(left_sections) & set(right_sections)),
            f"{left_label}_only_sections": sorted(set(left_sections) - set(right_sections)),
            f"{right_label}_only_sections": sorted(set(right_sections) - set(left_sections)),
            "focus_shift_sections": focus_shifts,
            "section_diffs": section_diffs,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": DOCUMENT_COMPARE_TOOL_NAME,
            "description": (
                "比较两份法律文书的差异，提炼共同点、各自主张和争点变化。"
                "默认不启用，适合起诉状/答辩状/上诉状之间做对照分析。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "left_document": {
                        "type": "string",
                        "description": "左侧文书全文。",
                    },
                    "right_document": {
                        "type": "string",
                        "description": "右侧文书全文。",
                    },
                    "left_label": {
                        "type": "string",
                        "description": "左侧文书标签，默认 document_a。",
                    },
                    "right_label": {
                        "type": "string",
                        "description": "右侧文书标签，默认 document_b。",
                    },
                },
                "required": ["left_document", "right_document"],
                "additionalProperties": False,
            },
        },
    }


def create_document_compare_tool(agent: Any | None = None) -> FunctionTool:
    del agent
    impl = DocumentCompareTool()
    return FunctionTool(
        impl.compare_documents,
        openai_tool_schema=_build_schema(),
    )


__all__ = [
    "DOCUMENT_COMPARE_TOOL_NAME",
    "DocumentCompareTool",
    "create_document_compare_tool",
]
