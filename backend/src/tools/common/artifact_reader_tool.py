"""Common case artifact reader tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from camel.toolkits import FunctionTool

from src.utils.file_io import safe_read_json


_READABLE_SUFFIXES = {".json", ".md", ".txt"}
_ARTIFACT_DESCRIPTION_MAP = {
    "pipeline_result.json": "全流程模拟输出汇总，包含案件元信息、已完成阶段和阶段结果。",
    "eval_result/summary.json": "评测汇总，包含总分、阶段分和理由摘要。",
    "eval_result/eval_result_full.json": "评测完整结果，包含细项评分、GT 和待评内容。",
    "lawyer/stage_summaries.json": "律师在各阶段沉淀的阶段摘要。",
    "lawyer/long_term_memory.json": "律师在该案中的长期记忆。",
    "PLC_result.json": "原告咨询阶段结果，通常含原告与原告律师的对话历史。",
    "LC_result.json": "法律咨询阶段兼容结果，当前通常等同于原告咨询 PLC_result.json。",
    "DLC_result.json": "被告咨询阶段结果，通常含被告与被告律师的对话历史。",
    "CD_result.json": "起诉状草拟阶段结果。",
    "DD_result.json": "答辩状草拟阶段结果。",
    "CI_result.json": "一审庭审调查或判决结果。",
    "SD_result.json": "是否上诉判定结果。",
    "AD_result.json": "上诉状草拟结果。",
    "AR_result.json": "上诉答辩状草拟结果。",
    "CIA_result.json": "二审庭审调查或判决结果。",
}


def _normalize_rel_path(rel_path: str) -> str:
    raw = str(rel_path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("rel_path is required.")

    path_obj = Path(raw)
    if path_obj.is_absolute():
        raise ValueError("rel_path must be relative to the current case directory.")

    parts = [part for part in path_obj.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("rel_path cannot escape the current case directory.")
    return "/".join(parts)


def _extract_field_value(payload: Any, field: Optional[str]) -> Any:
    if not field:
        return payload

    current = payload
    for part in str(field).split("."):
        part = part.strip()
        if not part:
            continue
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Field '{field}' not found.")
            current = current[part]
            continue
        if isinstance(current, list):
            if not part.isdigit():
                raise KeyError(f"Field '{field}' not found.")
            current = current[int(part)]
            continue
        raise KeyError(f"Field '{field}' not found.")
    return current


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_structured_for_llm(payload: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(payload, dict):
        if not payload:
            return f"{prefix}(empty object)"
        lines: list[str] = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_render_structured_for_llm(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {_stringify_scalar(value)}")
        return "\n".join(lines)
    if isinstance(payload, list):
        if not payload:
            return f"{prefix}(empty list)"
        lines: list[str] = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}- item_{index}:")
                lines.append(_render_structured_for_llm(item, indent + 1))
            else:
                lines.append(f"{prefix}- {_stringify_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{_stringify_scalar(payload)}"


def _describe_artifact(rel_path: str) -> str:
    normalized = str(rel_path).replace("\\", "/")
    if normalized in _ARTIFACT_DESCRIPTION_MAP:
        return _ARTIFACT_DESCRIPTION_MAP[normalized]

    filename = Path(normalized).name
    if filename in _ARTIFACT_DESCRIPTION_MAP:
        return _ARTIFACT_DESCRIPTION_MAP[filename]
    if filename.endswith("_result.json"):
        return "阶段结果文件，通常包含对话、结构化输出或最终文书。"
    if normalized.endswith(".md"):
        return "Markdown 文档。"
    if normalized.endswith(".txt"):
        return "纯文本文件。"
    if normalized.endswith(".json"):
        return "结构化 JSON 文件。"
    return "当前案件目录下的可读文件。"


@dataclass
class ArtifactReader:
    """Whitelisted artifact reader for a single case directory."""

    case_dir: Path
    _allowed_artifacts: dict[str, Path] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.case_dir = Path(self.case_dir).resolve()
        self._allowed_artifacts = self._discover_allowed_artifacts()

    def _discover_allowed_artifacts(self) -> dict[str, Path]:
        allowed: dict[str, Path] = {}
        for path in sorted(self.case_dir.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(self.case_dir).parts):
                continue
            if path.suffix.lower() not in _READABLE_SUFFIXES:
                continue
            rel_path = str(path.relative_to(self.case_dir)).replace("\\", "/")
            allowed[rel_path] = path
        return allowed

    def list_catalog_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for rel_path, path in self._allowed_artifacts.items():
            entry: dict[str, Any] = {
                "path": rel_path,
                "suffix": path.suffix.lower(),
                "description": _describe_artifact(rel_path),
            }
            if path.suffix.lower() == ".json":
                try:
                    payload = safe_read_json(path, default=None)
                    if isinstance(payload, dict):
                        entry["top_level_keys"] = sorted(payload.keys())
                except Exception:
                    entry["top_level_keys"] = ["<invalid_json_file>"]
            entries.append(entry)
        return entries

    def render_catalog(self) -> str:
        lines = [
            "Current case artifacts available for selective reading via `read_case_artifact`:",
        ]
        for entry in self.list_catalog_entries():
            keys = entry.get("top_level_keys")
            description = entry.get("description")
            if keys:
                lines.append(f"- {entry['path']} | {description} | top_level_keys={keys}")
            else:
                lines.append(f"- {entry['path']} | {description}")
        return "\n".join(lines)

    def _build_tool(self) -> FunctionTool:
        tool = FunctionTool(self.read_case_artifact)
        schema = tool.get_openai_tool_schema()
        schema["function"]["description"] = (
            "Read one whitelisted artifact under the current case directory by relative path. "
            "First inspect the artifact catalog from the prompt, which explains what each file contains. "
            "For JSON files you may optionally pass a dotted field path such as 'dialog_history' or "
            "'stage_eval_results.CI.metrics'. The tool returns plain text, not raw JSON."
        )
        properties = schema["function"]["parameters"].setdefault("properties", {})
        if "rel_path" in properties:
            properties["rel_path"]["description"] = (
                "Relative path under the current case directory. "
                "It must exactly match one path listed in the artifact catalog."
            )
        if "field" in properties:
            properties["field"]["description"] = (
                "Optional dotted field path for JSON files, such as "
                "'dialog_history' or 'stage_eval_results.CI.metrics'."
            )
        tool.set_openai_tool_schema(schema)
        return tool

    def get_tool(self) -> FunctionTool:
        return self._build_tool()

    def read_case_artifact(self, rel_path: str, field: Optional[str] = None) -> str:
        """Read one whitelisted artifact under the current case directory."""
        try:
            normalized = _normalize_rel_path(rel_path)
            target = self._allowed_artifacts.get(normalized)
            if target is None:
                return f"Error: rel_path '{normalized}' is not in the current artifact catalog."

            if target.suffix.lower() == ".json":
                payload = safe_read_json(target, default=None)
                selected = _extract_field_value(payload, field)
                rendered = _render_structured_for_llm(selected)
                return (
                    f"[artifact]\npath: {normalized}\nfield: {field or '<root>'}\n"
                    f"description: {_describe_artifact(normalized)}\n\n"
                    f"{rendered}"
                )

            if field:
                return "Error: field is only supported for JSON artifacts."
            return target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: {exc}"


__all__ = ["ArtifactReader"]
