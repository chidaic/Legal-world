"""Shared YAML memory runtime for lawyer/client live-card memory."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

CLIENT_MEMORY_OWNER = "client"
LAWYER_MEMORY_OWNER = "lawyer"

CLIENT_LOAD_TOOL_NAME = "load_client_memory"
CLIENT_SAVE_TOOL_NAME = "save_client_memory"
LAWYER_LOAD_TOOL_NAME = "load_lawyer_memory"
LAWYER_SAVE_TOOL_NAME = "save_lawyer_memory"

CLIENT_MEMORY_SCHEMA: dict[str, Any] = {
    "case_knowledge": {
        "self_narrative": "",
        "case_stage": "",
    },
    "demands": {
        "core_demands": "",
    },
}

LAWYER_MEMORY_SCHEMA: dict[str, Any] = {
    "case_facts": {
        "case_summary": "",
        "evidence_ledger": "",
    },
    "legal_analysis": {
        "legal_frame": "",
        "dispute_focus": "",
    },
    "client_brief": {
        "client_profile": "",
        "client_demand_list": "",
    },
}

MID_FLOW_STAGE_CODES = {"LC", "CD", "DD", "AD", "AR"}
LAWYER_DRAFTING_STAGE_CODES = {"CD", "DD", "AD", "AR"}
LAWYER_DRAFTING_REPLACE_MEMORY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("case_facts", "evidence_ledger"),
    ("client_brief", "client_demand_list"),
)
MEMORY_OPERATION_TYPES = {"revise", "expand"}
APPEND_ONLY_MEMORY_FIELDS: dict[str, tuple[tuple[str, ...], ...]] = {
    CLIENT_MEMORY_OWNER: (("case_knowledge", "case_stage"),),
    LAWYER_MEMORY_OWNER: (),
}

MEMORY_SECTION_FIELD_ALIASES: dict[str, dict[str, dict[str, str]]] = {
    CLIENT_MEMORY_OWNER: {
        "case_knowledge": {
            "narrative": "self_narrative",
            "stage": "case_stage",
        },
        "demands": {
            "demand_list": "core_demands",
            "demands": "core_demands",
        },
    },
    LAWYER_MEMORY_OWNER: {
        "case_facts": {
            "stage": "case_summary",
            "case_stage": "case_summary",
            "summary": "case_summary",
            "case_background": "case_summary",
            "facts": "case_summary",
            "core_facts": "case_summary",
            "timeline": "case_summary",
            "fact_timeline": "case_summary",
            "background": "case_summary",
            "evidence": "evidence_ledger",
            "ledger": "evidence_ledger",
        },
        "legal_analysis": {
            "frame": "legal_frame",
            "focus": "dispute_focus",
        },
        "client_brief": {
            "profile": "client_profile",
            "demand_list": "client_demand_list",
            "demands": "client_demand_list",
        },
    },
}


@dataclass(slots=True)
class ResolvedMemoryPaths:
    memory_path: Path
    history_path: Path
    owner: str
    slot: str
    case_output_dir: Path


def _build_schema_key_index(schema: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key, template in schema.items():
        keys.add(key)
        if isinstance(template, dict):
            keys.update(_build_schema_key_index(template))
    return keys


def _deepcopy_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(schema)


def get_memory_schema(memory_owner: str) -> dict[str, Any]:
    normalized = str(memory_owner or "").strip().lower()
    if normalized == CLIENT_MEMORY_OWNER:
        return _deepcopy_schema(CLIENT_MEMORY_SCHEMA)
    if normalized == LAWYER_MEMORY_OWNER:
        return _deepcopy_schema(LAWYER_MEMORY_SCHEMA)
    raise ValueError(f"Unsupported memory owner: {memory_owner}")


def get_empty_memory_payload(memory_owner: str) -> dict[str, Any]:
    return get_memory_schema(memory_owner)


def _normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        raise ValueError("字段值必须是纯文本字符串，不能是对象或数组。")
    return str(value).strip()


def _normalize_by_schema(
    payload: Any,
    schema: dict[str, Any],
    *,
    path: str = "",
) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        label = path or "<root>"
        raise ValueError(f"{label} 必须是 YAML 映射对象。")

    unknown_keys = sorted(set(payload.keys()) - set(schema.keys()))
    if unknown_keys:
        label = path or "<root>"
        raise ValueError(f"{label} 存在未知字段: {unknown_keys}")

    normalized: dict[str, Any] = {}
    for key, template in schema.items():
        field_path = f"{path}.{key}" if path else key
        value = payload.get(key)
        if isinstance(template, dict):
            normalized[key] = _normalize_by_schema(value, template, path=field_path)
            continue
        normalized[key] = _normalize_scalar(value)
    return normalized


def _split_memory_text_units(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []

    units = [unit.strip() for unit in re.split(r"\n\s*\n+", value) if unit.strip()]
    if len(units) > 1:
        return units

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    sentence_units = [
        unit.strip()
        for unit in re.split(r"(?<=[。！？.!?])\s+", value)
        if unit.strip()
    ]
    return sentence_units or [value]


def _memory_unit_dedupe_key(unit: str) -> str:
    key = re.sub(r"\s+", "", str(unit or ""))
    key = re.sub(r"^[0-9一二三四五六七八九十]+[.、)）．]+", "", key)
    return key


def _dedupe_memory_text_units(text: str) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for unit in _split_memory_text_units(text):
        key = _memory_unit_dedupe_key(unit)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(unit)
    return "\n\n".join(deduped)


def _compact_memory_text(memory_owner: str, path: tuple[str, ...], value: str) -> str:
    return _dedupe_memory_text_units(value)


def _compact_memory_payload(
    memory_owner: str,
    payload: dict[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            compacted[key] = _compact_memory_payload(memory_owner, value, prefix=path)
            continue
        compacted[key] = _compact_memory_text(memory_owner, path, str(value or ""))
    return compacted


def _collect_schema_leaf_targets(
    schema: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, list[tuple[str, ...]]]:
    targets: dict[str, list[tuple[str, ...]]] = {}
    for key, template in schema.items():
        current_path = tuple(filter(None, [*prefix.split("."), key])) if prefix else (key,)
        if isinstance(template, dict):
            nested = _collect_schema_leaf_targets(template, prefix=".".join(current_path))
            for leaf_key, leaf_paths in nested.items():
                targets.setdefault(leaf_key, []).extend(leaf_paths)
            continue
        targets.setdefault(key, []).append(current_path)
    return targets




def _repair_common_multiline_yaml_omissions(
    raw: str,
    schema: dict[str, Any],
    memory_owner: str,
) -> str:
    lines = raw.splitlines()
    if not lines:
        return raw

    known_keys = _build_schema_key_index(schema)
    leaf_keys = set(_collect_schema_leaf_targets(schema).keys())

    def _parse_key_line(line: str) -> tuple[int, str, str] | None:
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
        if not match:
            return None
        return len(match.group(1)), match.group(2), match.group(3)

    repaired: list[str] = []
    active_multiline_indent: int | None = None

    for index, line in enumerate(lines):
        parsed = _parse_key_line(line)

        if active_multiline_indent is not None:
            if not line.strip():
                repaired.append(" " * (active_multiline_indent + 2))
                continue

            is_boundary = False
            if parsed is not None:
                current_indent, current_key, _current_rest = parsed
                if current_indent < active_multiline_indent or (
                    current_indent <= active_multiline_indent and current_key in known_keys
                ):
                    is_boundary = True

            if not is_boundary:
                content = line.lstrip()
                repaired.append(" " * (active_multiline_indent + 2) + content)
                continue

            active_multiline_indent = None

        if parsed is None:
            repaired.append(line)
            continue

        indent, key, rest = parsed
        if key not in leaf_keys:
            repaired.append(line)
            continue

        next_non_empty_line = None
        for look_ahead in lines[index + 1 :]:
            if look_ahead.strip():
                next_non_empty_line = look_ahead
                break

        if next_non_empty_line is None:
            repaired.append(line)
            continue

        next_parsed = _parse_key_line(next_non_empty_line)
        should_open_multiline = False
        normalized_rest = rest.strip()
        if normalized_rest in {"|", "|-", ">", ">-"}:
            repaired.append(line)
            continue

        if not normalized_rest:
            should_open_multiline = True
        elif next_parsed is None:
            should_open_multiline = True
        else:
            next_indent, next_key, _next_rest = next_parsed
            if next_indent > indent:
                should_open_multiline = True
            elif next_indent <= indent and next_key not in known_keys:
                should_open_multiline = True

        if not should_open_multiline:
            repaired.append(line)
            continue

        repaired.append(" " * indent + f"{key}: |")
        if normalized_rest:
            repaired.append(" " * (indent + 2) + normalized_rest)
        active_multiline_indent = indent

    return "\n".join(repaired)


def _autofix_root_level_leaf_misplacements(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    fixed = deepcopy(payload)
    leaf_targets = _collect_schema_leaf_targets(schema)
    root_keys = list(fixed.keys())

    for key in root_keys:
        if key in schema:
            continue
        target_paths = leaf_targets.get(key) or []
        if len(target_paths) != 1:
            continue
        target_path = target_paths[0]
        if len(target_path) < 2:
            continue

        cursor: dict[str, Any] = fixed
        blocked = False
        for section_key in target_path[:-1]:
            nested_value = cursor.get(section_key)
            if nested_value is None:
                cursor[section_key] = {}
                nested_value = cursor[section_key]
            if not isinstance(nested_value, dict):
                blocked = True
                break
            cursor = nested_value
        if blocked:
            continue

        canonical_key = target_path[-1]
        existing_value = cursor.get(canonical_key)
        misplaced_value = fixed.get(key)
        if str(existing_value or "").strip():
            continue
        cursor[canonical_key] = misplaced_value
        fixed.pop(key, None)

    return fixed


def _has_meaningful_nested_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_meaningful_nested_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_nested_value(item) for item in value)
    return bool(str(value or "").strip())


def _autofix_nested_root_section_misplacements(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    fixed = deepcopy(payload)
    top_level_sections = set(schema.keys())

    for parent_key in list(top_level_sections):
        parent_payload = fixed.get(parent_key)
        parent_schema = schema.get(parent_key)
        if not isinstance(parent_payload, dict) or not isinstance(parent_schema, dict):
            continue

        nested_keys = list(parent_payload.keys())
        allowed_nested_keys = set(parent_schema.keys())
        for nested_key in nested_keys:
            if nested_key in allowed_nested_keys:
                continue
            if nested_key not in top_level_sections:
                continue

            misplaced_value = parent_payload.get(nested_key)
            root_value = fixed.get(nested_key)
            if _has_meaningful_nested_value(root_value):
                continue

            fixed[nested_key] = misplaced_value
            parent_payload.pop(nested_key, None)

    return fixed


def _autofix_section_field_aliases(
    payload: dict[str, Any],
    memory_owner: str,
) -> dict[str, Any]:
    fixed = deepcopy(payload)
    section_aliases = MEMORY_SECTION_FIELD_ALIASES.get(str(memory_owner or "").strip().lower(), {})
    for section_key, aliases in section_aliases.items():
        section_payload = fixed.get(section_key)
        if not isinstance(section_payload, dict):
            continue
        for alias_key, canonical_key in aliases.items():
            if alias_key not in section_payload:
                continue
            alias_value = section_payload.get(alias_key)
            canonical_value = section_payload.get(canonical_key)
            if not str(canonical_value or "").strip():
                section_payload[canonical_key] = alias_value
            section_payload.pop(alias_key, None)
    return fixed


def _autofix_cross_section_leaf_misplacements(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    fixed = deepcopy(payload)
    top_level_sections = set(schema.keys())
    leaf_targets = _collect_schema_leaf_targets(schema)

    for parent_key, parent_schema in schema.items():
        parent_payload = fixed.get(parent_key)
        if not isinstance(parent_payload, dict) or not isinstance(parent_schema, dict):
            continue

        allowed_nested_keys = set(parent_schema.keys())
        for nested_key in list(parent_payload.keys()):
            if nested_key in allowed_nested_keys or nested_key in top_level_sections:
                continue

            target_paths = leaf_targets.get(nested_key) or []
            if len(target_paths) != 1:
                continue

            target_path = target_paths[0]
            if len(target_path) < 2 or target_path[0] == parent_key:
                continue

            misplaced_value = parent_payload.get(nested_key)
            if isinstance(misplaced_value, (dict, list, tuple, set)):
                continue

            cursor: dict[str, Any] = fixed
            blocked = False
            for section_key in target_path[:-1]:
                nested_value = cursor.get(section_key)
                if nested_value is None:
                    cursor[section_key] = {}
                    nested_value = cursor[section_key]
                if not isinstance(nested_value, dict):
                    blocked = True
                    break
                cursor = nested_value
            if blocked:
                continue

            canonical_key = target_path[-1]
            existing_value = cursor.get(canonical_key)
            if str(existing_value or "").strip():
                continue

            cursor[canonical_key] = misplaced_value
            parent_payload.pop(nested_key, None)

    return fixed


def normalize_memory_payload(memory_owner: str, payload: Any) -> dict[str, Any]:
    schema = get_memory_schema(memory_owner)
    if isinstance(payload, dict):
        payload = deepcopy(payload)
        payload = _upgrade_legacy_memory_payload(payload, memory_owner)
        payload = _autofix_nested_root_section_misplacements(payload, schema)
        payload = _autofix_section_field_aliases(payload, memory_owner)
        payload = _autofix_cross_section_leaf_misplacements(payload, schema)
        payload = _autofix_root_level_leaf_misplacements(payload, schema)
    normalized = _normalize_by_schema(payload, schema)
    return _compact_memory_payload(memory_owner, normalized)


def render_memory_yaml(memory_owner: str, payload: Any) -> str:
    normalized = normalize_memory_payload(memory_owner, payload)
    rendered = yaml.safe_dump(
        normalized,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    return rendered.rstrip() + "\n"


def parse_memory_yaml(memory_owner: str, content: str) -> dict[str, Any]:
    raw = str(content or "").strip()
    if not raw:
        raise ValueError("memory.yaml \u5185\u5bb9\u4e0d\u80fd\u4e3a\u7a7a\u3002")
    schema = get_memory_schema(memory_owner)
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        repaired = _repair_common_multiline_yaml_omissions(raw, schema, memory_owner)
        if repaired == raw:
            raise ValueError(f"YAML \u89e3\u6790\u5931\u8d25: {exc}") from exc
        try:
            payload = yaml.safe_load(repaired)
        except yaml.YAMLError as repaired_exc:
            raise ValueError(f"YAML \u89e3\u6790\u5931\u8d25: {repaired_exc}") from repaired_exc
    return normalize_memory_payload(memory_owner, payload)


def flatten_memory_payload(payload: Any, *, prefix: str = "") -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, str] = {}
    for key, value in payload.items():
        field_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_memory_payload(value, prefix=field_path))
            continue
        flattened[field_path] = str(value or "").strip()
    return flattened


def _compose_case_summary_text(*parts: str) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    for raw_part in parts:
        text = str(raw_part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        blocks.append(text)
    return "\n\n".join(blocks)


def _upgrade_legacy_memory_payload(payload: dict[str, Any], memory_owner: str) -> dict[str, Any]:
    if str(memory_owner or "").strip().lower() != LAWYER_MEMORY_OWNER:
        return payload

    fixed = deepcopy(payload)
    fixed.pop("strategy", None)

    case_facts = fixed.get("case_facts")
    if isinstance(case_facts, dict):
        case_summary = _compose_case_summary_text(
            case_facts.get("case_summary", ""),
            case_facts.get("case_stage", ""),
            case_facts.get("case_background", ""),
            case_facts.get("core_facts", ""),
            case_facts.get("fact_timeline", ""),
            case_facts.get("facts", ""),
            case_facts.get("timeline", ""),
        )
        if case_summary:
            case_facts["case_summary"] = case_summary
        for legacy_key in ("case_stage", "case_background", "core_facts", "fact_timeline", "facts", "timeline"):
            case_facts.pop(legacy_key, None)

    legal_analysis = fixed.get("legal_analysis")
    if isinstance(legal_analysis, dict):
        legal_analysis.pop("opponent", None)
        allowed_legal_keys = {"legal_frame", "dispute_focus", "frame", "focus"}
        fixed["legal_analysis"] = {
            key: value for key, value in legal_analysis.items() if key in allowed_legal_keys
        }

    root_case_summary = _compose_case_summary_text(
        fixed.get("case_summary", ""),
        fixed.get("case_stage", ""),
        fixed.get("case_background", ""),
        fixed.get("core_facts", ""),
        fixed.get("fact_timeline", ""),
    )
    if root_case_summary:
        fixed["case_summary"] = root_case_summary
    fixed.pop("case_stage", None)
    fixed.pop("core_facts", None)
    fixed.pop("fact_timeline", None)
    fixed.pop("case_background", None)

    return fixed


def has_meaningful_memory(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(has_meaningful_memory(value) for value in payload.values())
    if isinstance(payload, (list, tuple, set)):
        return any(has_meaningful_memory(value) for value in payload)
    return bool(str(payload or "").strip())


def _infer_revision_op(before_text: str, after_text: str) -> str:
    before = str(before_text or "").strip()
    after = str(after_text or "").strip()
    if before == after:
        return "revise"
    if not before and after:
        return "expand"
    if before and after and (after.startswith(before) or before in after):
        return "expand"
    return "revise"


def build_history_entry(
    *,
    memory_owner: str,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
    source_stage: str = "",
    revision_ops_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    before_flat = flatten_memory_payload(before_payload)
    after_flat = flatten_memory_payload(after_payload)
    changed_fields = [
        field
        for field in after_flat
        if before_flat.get(field, "") != after_flat.get(field, "")
    ]
    revision_ops_override = revision_ops_override or {}
    revision_ops = {
        field: revision_ops_override.get(
            field,
            _infer_revision_op(before_flat.get(field, ""), after_flat.get(field, "")),
        )
        for field in changed_fields
    }
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source_stage": str(source_stage or "").strip(),
        "memory_owner": str(memory_owner or "").strip(),
        "changed_fields": changed_fields,
        "revision_ops": revision_ops,
        "before": before_payload,
        "after": after_payload,
    }


def _get_nested_text(payload: dict[str, Any], path: tuple[str, ...]) -> str:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return str(current or "").strip()


def _set_nested_text(payload: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    cursor = payload
    for key in path[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[path[-1]] = str(value or "").strip()


def _allowed_memory_field_paths(memory_owner: str) -> set[str]:
    return set(flatten_memory_payload(get_memory_schema(memory_owner)).keys())


def _merge_append_only_text(before_text: str, after_text: str) -> str:
    before = _dedupe_memory_text_units(str(before_text or "").strip())
    after = _dedupe_memory_text_units(str(after_text or "").strip())
    if not before:
        return after
    if not after:
        return before
    if before in after:
        return after
    if after in before:
        return before
    return f"{before}\n\n{after}"


def apply_memory_operations(
    memory_owner: str,
    before_payload: dict[str, Any],
    operations: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply field-level revise/expand operations to a memory payload."""

    normalized_before = normalize_memory_payload(memory_owner, before_payload)
    if isinstance(operations, dict) and "operations" in operations:
        operations = operations.get("operations")
    if isinstance(operations, str):
        try:
            operations = json.loads(operations)
        except json.JSONDecodeError as exc:
            raise ValueError(f"operations 必须是 JSON 数组或包含 operations 数组的对象: {exc}") from exc
    if not isinstance(operations, list):
        raise ValueError("operations 必须是 JSON 数组。")

    allowed_fields = _allowed_memory_field_paths(memory_owner)
    merged = deepcopy(normalized_before)
    revision_ops: dict[str, str] = {}

    for index, raw_operation in enumerate(operations, start=1):
        if not isinstance(raw_operation, dict):
            raise ValueError(f"operations[{index}] 必须是对象。")
        field = str(
            raw_operation.get("field")
            or raw_operation.get("field_path")
            or raw_operation.get("path")
            or ""
        ).strip()
        operation = str(
            raw_operation.get("operation")
            or raw_operation.get("op")
            or ""
        ).strip().lower()
        raw_content = raw_operation.get("content") if "content" in raw_operation else raw_operation.get("value", "")

        if field not in allowed_fields:
            raise ValueError(
                f"operations[{index}].field 不合法: {field!r}。"
                f"允许字段: {sorted(allowed_fields)}"
            )
        if operation not in MEMORY_OPERATION_TYPES:
            raise ValueError(
                f"operations[{index}].operation 不合法: {operation!r}。"
                "只允许 revise 或 expand。"
            )

        content = _normalize_scalar(raw_content)
        path = tuple(field.split("."))
        if operation == "revise":
            _set_nested_text(merged, path, content)
        else:
            _set_nested_text(
                merged,
                path,
                _merge_append_only_text(_get_nested_text(merged, path), content),
            )
        revision_ops[field] = operation

    return normalize_memory_payload(memory_owner, merged), revision_ops


def merge_memory_update(
    memory_owner: str,
    before_payload: dict[str, Any],
    proposed_payload: dict[str, Any],
    *,
    replace_field_paths: tuple[tuple[str, ...], ...] = (),
) -> dict[str, Any]:
    normalized_before = normalize_memory_payload(memory_owner, before_payload)
    normalized_after = normalize_memory_payload(memory_owner, proposed_payload)
    merged = deepcopy(normalized_after)
    replace_field_keys = {".".join(path) for path in replace_field_paths}

    before_flat = flatten_memory_payload(normalized_before)
    for field_path, before_value in before_flat.items():
        if field_path in replace_field_keys:
            continue
        current_value = str(flatten_memory_payload(merged).get(field_path, "") or "").strip()
        if current_value:
            continue
        _set_nested_text(merged, tuple(field_path.split(".")), before_value)

    for path in APPEND_ONLY_MEMORY_FIELDS.get(str(memory_owner or "").strip().lower(), ()):
        if ".".join(path) in replace_field_keys:
            continue
        _set_nested_text(
            merged,
            path,
            _merge_append_only_text(
                _get_nested_text(normalized_before, path),
                _get_nested_text(normalized_after, path),
            ),
        )

    return normalize_memory_payload(memory_owner, merged)


def resolve_replace_memory_fields(
    memory_owner: str,
    source_stage: str,
) -> tuple[tuple[str, ...], ...]:
    owner = str(memory_owner or "").strip().lower()
    stage = str(source_stage or "").strip().upper()
    if owner == LAWYER_MEMORY_OWNER and stage in LAWYER_DRAFTING_STAGE_CODES:
        return LAWYER_DRAFTING_REPLACE_MEMORY_FIELDS
    return ()


def _sanitize_slot(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "agent"
    normalized = re.sub(r"[^0-9a-zA-Z_-]+", "_", raw)
    normalized = normalized.strip("_")
    return normalized or "agent"


def resolve_memory_slot(agent: Any) -> str:
    explicit = getattr(agent, "memory_owner_slot", None)
    if explicit:
        return _sanitize_slot(explicit)
    return _sanitize_slot(getattr(agent, "agent_id", None) or getattr(agent, "name", None))


def _legacy_json_path_from_explicit(path: Path) -> Path:
    if path.name == "memory.yaml":
        return path.with_name("long_term_memory.json")
    if path.suffix.lower() == ".json":
        return path
    return path.with_name("long_term_memory.json")


def _resolve_explicit_memory_path(agent: Any) -> Path | None:
    for attr_name in ("memory_yaml_path", "long_term_memory_path"):
        raw = str(getattr(agent, attr_name, "") or "").strip()
        if not raw:
            continue
        explicit_path = Path(raw)
        if explicit_path.suffix.lower() == ".json":
            explicit_path = explicit_path.with_name("memory.yaml")
        elif explicit_path.suffix.lower() != ".yaml":
            explicit_path = explicit_path / "memory.yaml"
        return explicit_path.resolve()
    return None


def _resolve_case_id_from_agent(agent: Any) -> str:
    candidate_values: list[str] = []

    for value in (
        getattr(agent, "current_handling_case", None),
        getattr(agent, "case_id", None),
    ):
        text = str(value or "").strip()
        if text:
            candidate_values.append(text)

    scenario_data = getattr(agent, "scenario_data", {}) or {}
    for key in ("case_id", "current_handling_case"):
        text = str(scenario_data.get(key, "") or "").strip()
        if text:
            candidate_values.append(text)

    storage = getattr(agent, "storage", None)
    config_path = getattr(agent, "config_path", None)
    if storage and config_path:
        try:
            config = storage.load_agent_config(config_path)
        except Exception:
            config = {}
        for key in ("current_handling_case", "case_id"):
            text = str(config.get(key, "") or "").strip()
            if text:
                candidate_values.append(text)

    for value in candidate_values:
        if value:
            return value
    raise ValueError("无法从当前 Agent 上下文解析案件 ID。")


def _resolve_case_output_dir(agent: Any) -> Path:
    explicit_memory_path = _resolve_explicit_memory_path(agent)
    if explicit_memory_path is not None:
        return explicit_memory_path.parent.parent.resolve()

    scenario_data = getattr(agent, "scenario_data", {}) or {}
    explicit_case_output_dir = str(scenario_data.get("case_output_dir", "") or "").strip()
    if explicit_case_output_dir:
        return Path(explicit_case_output_dir).resolve()

    storage = getattr(agent, "storage", None)
    if storage is not None:
        case_id = _resolve_case_id_from_agent(agent)
        return (Path(storage.base_dir) / "output" / case_id).resolve()

    raise ValueError("无法解析当前 Agent 对应的案件输出目录。")


def resolve_memory_paths(agent: Any, memory_owner: str) -> ResolvedMemoryPaths:
    explicit_memory_path = _resolve_explicit_memory_path(agent)
    if explicit_memory_path is not None:
        memory_path = explicit_memory_path
        case_output_dir = memory_path.parent.parent.resolve()
        slot = memory_path.parent.name
    else:
        case_output_dir = _resolve_case_output_dir(agent)
        slot = resolve_memory_slot(agent)
        memory_path = case_output_dir / slot / "memory.yaml"

    return ResolvedMemoryPaths(
        memory_path=memory_path.resolve(),
        history_path=memory_path.with_name("memory.history.yaml").resolve(),
        owner=str(memory_owner or "").strip().lower(),
        slot=slot,
        case_output_dir=case_output_dir.resolve(),
    )


def _load_yaml_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_memory_from_legacy(memory_owner: str, legacy_payload: Any) -> dict[str, Any]:
    payload = legacy_payload if isinstance(legacy_payload, dict) else {}
    if memory_owner == CLIENT_MEMORY_OWNER:
        return normalize_memory_payload(
            CLIENT_MEMORY_OWNER,
            {
                "case_knowledge": {
                    "self_narrative": str(
                        payload.get("self_narrative")
                        or payload.get("case_background")
                        or payload.get("case_stage")
                        or ""
                    ).strip(),
                    "case_stage": "",
                },
                "demands": {
                    "core_demands": str(payload.get("core_demands", "") or "").strip(),
                },
            },
        )
    if memory_owner == LAWYER_MEMORY_OWNER:
        return normalize_memory_payload(
            LAWYER_MEMORY_OWNER,
            {
                "case_facts": {
                    "case_summary": _compose_case_summary_text(
                        payload.get("case_summary", ""),
                        payload.get("case_stage", ""),
                        payload.get("case_background", ""),
                        payload.get("core_facts", ""),
                        payload.get("fact_timeline", ""),
                    ),
                    "evidence_ledger": "",
                },
                "legal_analysis": {
                    "legal_frame": str(payload.get("legal_relationship", "") or "").strip(),
                    "dispute_focus": str(payload.get("dispute_focus", "") or "").strip(),
                },
                "client_brief": {
                    "client_profile": "",
                    "client_demand_list": str(
                        payload.get("client_demand_list")
                        or payload.get("claims")
                        or payload.get("appeal_claims")
                        or payload.get("core_demands")
                        or ""
                    ).strip(),
                },
            },
        )
    raise ValueError(f"Unsupported memory owner: {memory_owner}")


def _resolve_initial_client_self_narrative(agent: Any) -> str:
    scenario_data = getattr(agent, "scenario_data", {}) or {}
    explicit = str(scenario_data.get("case_background", "") or "").strip()
    if explicit:
        return explicit

    storage = getattr(agent, "storage", None)
    config_path = getattr(agent, "config_path", None)
    if not storage or not config_path:
        return ""

    try:
        config = storage.load_agent_config(config_path)
    except Exception:
        return ""

    dataset_path = str(config.get("dataset_path", "") or "").strip()
    if not dataset_path:
        return ""

    try:
        from ..data.data_loader import DataLoader

        loader = DataLoader(dataset_path)
        case = loader.resolve_case_for_config(config)
        return str(loader.extract_case_background(case) or "").strip()
    except Exception as exc:
        logger.warning("Failed to resolve initial client self_narrative: %s", exc)
        return ""


def build_default_memory(memory_owner: str, agent: Any | None = None) -> dict[str, Any]:
    payload = get_empty_memory_payload(memory_owner)
    if memory_owner == CLIENT_MEMORY_OWNER and agent is not None:
        payload["case_knowledge"]["self_narrative"] = _resolve_initial_client_self_narrative(agent)
    return payload


def _load_legacy_payload(agent: Any, paths: ResolvedMemoryPaths) -> dict[str, Any] | None:
    candidates: list[Path] = []

    explicit_legacy = getattr(agent, "long_term_memory_path", None)
    if explicit_legacy:
        candidates.append(Path(str(explicit_legacy)).resolve())

    candidates.append(_legacy_json_path_from_explicit(paths.memory_path))

    storage = getattr(agent, "storage", None)
    config_path = getattr(agent, "config_path", None)
    config_memory: Any = None
    if storage and config_path:
        try:
            config = storage.load_agent_config(config_path)
            config_memory = config.get("long_term_memory")
        except Exception:
            config_memory = None

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen or not candidate.exists():
            continue
        seen.add(candidate_str)
        try:
            payload = _read_json_file(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    if isinstance(config_memory, dict):
        return config_memory
    return None


def update_agent_memory_cache(agent: Any, memory_owner: str, payload: dict[str, Any]) -> None:
    normalized = normalize_memory_payload(memory_owner, payload)
    if memory_owner == CLIENT_MEMORY_OWNER:
        setattr(agent, "client_profile", normalized)
        return
    if memory_owner == LAWYER_MEMORY_OWNER:
        setattr(agent, "legal_profile", normalized)
        return
    raise ValueError(f"Unsupported memory owner: {memory_owner}")


def get_agent_memory_cache(agent: Any, memory_owner: str) -> dict[str, Any]:
    if memory_owner == CLIENT_MEMORY_OWNER:
        return normalize_memory_payload(
            memory_owner,
            getattr(agent, "client_profile", get_empty_memory_payload(memory_owner)),
        )
    if memory_owner == LAWYER_MEMORY_OWNER:
        return normalize_memory_payload(
            memory_owner,
            getattr(agent, "legal_profile", get_empty_memory_payload(memory_owner)),
        )
    raise ValueError(f"Unsupported memory owner: {memory_owner}")


def ensure_memory_file(agent: Any, memory_owner: str) -> tuple[dict[str, Any], ResolvedMemoryPaths]:
    paths = resolve_memory_paths(agent, memory_owner)
    setattr(agent, "memory_yaml_path", str(paths.memory_path))
    paths.memory_path.parent.mkdir(parents=True, exist_ok=True)

    if paths.memory_path.exists():
        payload = normalize_memory_payload(memory_owner, _load_yaml_file(paths.memory_path))
        rendered = render_memory_yaml(memory_owner, payload)
        current = paths.memory_path.read_text(encoding="utf-8")
        if current != rendered:
            paths.memory_path.write_text(rendered, encoding="utf-8")
        return payload, paths

    legacy_payload = _load_legacy_payload(agent, paths)
    if isinstance(legacy_payload, dict):
        payload = bootstrap_memory_from_legacy(memory_owner, legacy_payload)
    else:
        payload = build_default_memory(memory_owner, agent=agent)

    paths.memory_path.write_text(render_memory_yaml(memory_owner, payload), encoding="utf-8")
    return payload, paths


def load_memory_for_agent(agent: Any, memory_owner: str) -> tuple[dict[str, Any], ResolvedMemoryPaths]:
    payload, paths = ensure_memory_file(agent, memory_owner)
    update_agent_memory_cache(agent, memory_owner, payload)
    return payload, paths


def _load_history_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = _load_yaml_file(path)
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def save_memory_for_agent(
    agent: Any,
    *,
    memory_owner: str,
    content: str,
) -> dict[str, Any]:
    before_payload, paths = ensure_memory_file(agent, memory_owner)
    source_stage = str(getattr(agent, "_simlaw_stage_code", "") or "").strip().upper()
    after_payload = merge_memory_update(
        memory_owner,
        before_payload,
        parse_memory_yaml(memory_owner, content),
        replace_field_paths=resolve_replace_memory_fields(memory_owner, source_stage),
    )

    paths.memory_path.write_text(
        render_memory_yaml(memory_owner, after_payload),
        encoding="utf-8",
    )
    update_agent_memory_cache(agent, memory_owner, after_payload)

    history_entry = build_history_entry(
        memory_owner=memory_owner,
        before_payload=before_payload,
        after_payload=after_payload,
        source_stage=source_stage,
    )
    history_entries = _load_history_entries(paths.history_path)
    history_entries.append(history_entry)
    _write_yaml_file(paths.history_path, history_entries)

    return {
        "paths": paths,
        "history_entry": history_entry,
        "before": before_payload,
        "after": after_payload,
    }


def load_memory_yaml_text(agent: Any, memory_owner: str) -> str:
    payload, _paths = load_memory_for_agent(agent, memory_owner)
    return render_memory_yaml(memory_owner, payload)


def _render_field_definitions(memory_owner: str) -> str:
    if memory_owner == LAWYER_MEMORY_OWNER:
        return (
            "固定字段及定义：\n"
            "1. case_facts.case_summary：案件阶段 / Case Summary，写剥离对话水分后提炼出的纯粹、客观的已确认事实，以及随着法律流程推进产生的案件事实进展；必须覆盖案件背景、关键事实经过、当前程序进展/最新结论。每次提交都应是压缩后的完整最新案情，不能把旧案情整段重复粘贴。\n"
            "2. case_facts.evidence_ledger：证据台账，写证据内容、来源、证明对象、证明力强弱、对应争点。\n"
            "3. legal_analysis.legal_frame：法律关系定性、请求权基础、适用法条方向、抗辩框架。\n"
            "4. legal_analysis.dispute_focus：当前核心争议点，以及哪些点已清楚、哪些点仍待查明。\n"
            "5. client_brief.client_profile：当事人的沟通风格、配合度、诉求倾向、风险偏好。\n"
            "6. client_brief.client_demand_list：当事人当前明确提出的诉求列表，按项沉淀，便于对标起诉状或上诉状请求。"
        )
    if memory_owner == CLIENT_MEMORY_OWNER:
        return (
            "固定字段及定义：\n"
            "1. case_knowledge.self_narrative：当事人自己眼里事情是怎么发生的，偏案件背景/完整经过/事实叙事，相对稳定，主要写事情本身，不写和律师推进了什么。\n"
            "2. case_knowledge.case_stage：当前案件随着和律师推进到了什么状态，偏过程进展/当前卡点/已完成动作/下一步要做什么，和 self_narrative 必须区分开，而且只能在原有基础上追加，不要覆盖旧进展。\n"
            "3. demands.core_demands：当事人当前核心诉求、底线、优先级，包括是否接受调解、最想达到什么结果、最在意什么风险。"
        )
    raise ValueError(f"Unsupported memory owner: {memory_owner}")


def _render_revision_semantics() -> str:
    return (
        "字段级操作语义：\n"
        "1. revise：新信息与旧判断存在冲突，或对原有信息进行精化、纠正、压缩；工具会用 content 覆盖该字段旧值。\n"
        "2. expand：在原字段基础上追加新条目、新细节、新证据或新进展；工具会把 content 追加到该字段旧值之后并做基础去重。"
    )


def build_load_tool_description(memory_owner: str) -> str:
    role_label = "律师" if memory_owner == LAWYER_MEMORY_OWNER else "当事人"
    return (
        f"读取当前案件下{role_label}的 memory.yaml，并返回完整 canonical YAML 文本。"
        "如果文件不存在，系统会先初始化默认 memory.yaml；如果只存在旧的 long_term_memory.json 或 config.long_term_memory，"
        "系统会先迁移为 memory.yaml 再返回。这个工具会同时更新当前 Agent 的运行时 memory cache。"
        "读取到的内容必须被视为最新版；后续写回时，只提交需要修改的字段级 operations，不要提交整份 YAML。"
    )


def build_save_tool_description(memory_owner: str) -> str:
    role_label = "律师" if memory_owner == LAWYER_MEMORY_OWNER else "当事人"
    drafting_guard = (
        "在 CD/DD/AD/AR 文书起草阶段结束后写回律师记忆时，诉求、答辩意见与证据台账必须严格依据刚起草完成的文书正文，"
        "不得补写文书未列明的诉求、答辩意见或证据。"
        if memory_owner == LAWYER_MEMORY_OWNER
        else ""
    )
    return (
        f"对当前案件下{role_label}的 memory.yaml 执行字段级更新。"
        "这是字段级 JSON operations tool，不接受整份 YAML。"
        "只提交需要修改的字段；未提交字段保持原值。"
        "每个 operation 必须包含 field、operation、content。"
        "operation 只允许 revise 或 expand：revise 覆盖该字段，expand 追加到该字段。"
        "field 必须是固定 schema 的末级字段路径。"
        f"{drafting_guard}"
        "保存成功后，系统会自动根据旧版 memory.yaml 和新版 memory.yaml 的差异生成 memory.history.yaml 留痕；"
        "history 中的 revision_ops 记录本次字段操作类型。"
        "绝对不得编造当前认知中不存在的事实、证据、日期、金额、程序结论。"
        "\n\n"
        f"{_render_field_definitions(memory_owner)}\n\n"
        f"{_render_revision_semantics()}"
    )


def build_save_operations_description(memory_owner: str) -> str:
    role_label = "律师" if memory_owner == LAWYER_MEMORY_OWNER else "当事人"
    drafting_guard = (
        "如果处于 CD/DD/AD/AR 文书起草阶段，诉求、答辩意见和证据内容必须严格依据刚起草完成的文书正文。"
        if memory_owner == LAWYER_MEMORY_OWNER
        else ""
    )
    return (
        f"{role_label} memory.yaml 的字段级更新列表。"
        "格式为 JSON 数组，每个元素只能包含："
        "`field`（末级字段路径）、`operation`（revise 或 expand）、`content`（覆盖或追加的具体文本）。"
        "只写需要修改的字段，不要提交完整 YAML，也不要提交未变化字段。"
        f"{drafting_guard}"
        f"\n\n{_render_field_definitions(memory_owner)}\n\n{_render_revision_semantics()}"
    )


def build_save_content_description(memory_owner: str) -> str:
    return build_save_operations_description(memory_owner)


def build_memory_prompt_block(memory_owner: str, payload: Any) -> str:
    normalized_owner = str(memory_owner or "").strip().lower()
    normalized_payload = normalize_memory_payload(normalized_owner, payload)
    if not has_meaningful_memory(normalized_payload):
        return ""

    if normalized_owner == LAWYER_MEMORY_OWNER:
        section_map = (
            ("案件阶段 / Case Summary", normalized_payload["case_facts"]["case_summary"]),
            ("证据台账", normalized_payload["case_facts"]["evidence_ledger"]),
            ("法律分析", normalized_payload["legal_analysis"]["legal_frame"]),
            ("争议焦点", normalized_payload["legal_analysis"]["dispute_focus"]),
            ("客户画像", normalized_payload["client_brief"]["client_profile"]),
            ("客户诉求", normalized_payload["client_brief"]["client_demand_list"]),
        )
        header = (
            "【长期记忆】\n"
            "若需要更新长期记忆，调用 `load_skill` 加载 `lawyer-memory-writing`，"
            "再基于当前长期记忆内容提交字段级 operations，并调用 `save_lawyer_memory` 写回。"
        )
    elif normalized_owner == CLIENT_MEMORY_OWNER:
        section_map = (
            ("案件自述", normalized_payload["case_knowledge"]["self_narrative"]),
            ("案件推进进展", normalized_payload["case_knowledge"]["case_stage"]),
            ("当前核心诉求", normalized_payload["demands"]["core_demands"]),
        )
        header = (
            "【长期记忆】\n"
            "若需要更新长期记忆，调用 `load_skill` 加载 `client-memory-writing`，"
            "再基于当前长期记忆内容提交字段级 operations，并调用 `save_client_memory` 写回。"
        )
    else:
        raise ValueError(f"Unsupported memory owner: {memory_owner}")

    blocks = [
        f"{label}：\n{str(value or '').strip()}"
        for label, value in section_map
        if str(value or "").strip()
    ]
    if not blocks:
        return ""
    return header + "\n\n" + "\n\n".join(blocks)


def is_mid_flow_stage(stage_code: str) -> bool:
    return str(stage_code or "").strip().upper() in MID_FLOW_STAGE_CODES


def summarize_save_result(result: dict[str, Any]) -> str:
    history_entry = dict(result.get("history_entry") or {})
    changed_fields = history_entry.get("changed_fields") or []
    revision_ops = history_entry.get("revision_ops") or {}
    changed_text = "、".join(changed_fields) if changed_fields else "无字段变化"
    ops_text = (
        "；".join(f"{field}={op}" for field, op in revision_ops.items())
        if isinstance(revision_ops, dict) and revision_ops
        else "无修订语义"
    )
    paths = result.get("paths")
    memory_path = getattr(paths, "memory_path", "")
    history_path = getattr(paths, "history_path", "")
    return (
        "memory 保存成功。\n"
        f"memory_path: {memory_path}\n"
        f"history_path: {history_path}\n"
        f"changed_fields: {changed_text}\n"
        f"revision_ops: {ops_text}"
    )


__all__ = [
    "CLIENT_LOAD_TOOL_NAME",
    "CLIENT_MEMORY_OWNER",
    "CLIENT_MEMORY_SCHEMA",
    "CLIENT_SAVE_TOOL_NAME",
    "LAWYER_LOAD_TOOL_NAME",
    "LAWYER_MEMORY_OWNER",
    "LAWYER_MEMORY_SCHEMA",
    "LAWYER_SAVE_TOOL_NAME",
    "ResolvedMemoryPaths",
    "apply_memory_operations",
    "bootstrap_memory_from_legacy",
    "build_default_memory",
    "build_history_entry",
    "build_load_tool_description",
    "build_memory_prompt_block",
    "build_save_content_description",
    "build_save_operations_description",
    "build_save_tool_description",
    "ensure_memory_file",
    "flatten_memory_payload",
    "get_agent_memory_cache",
    "get_empty_memory_payload",
    "get_memory_schema",
    "has_meaningful_memory",
    "is_mid_flow_stage",
    "load_memory_for_agent",
    "load_memory_yaml_text",
    "normalize_memory_payload",
    "parse_memory_yaml",
    "render_memory_yaml",
    "resolve_memory_paths",
    "resolve_memory_slot",
    "resolve_replace_memory_fields",
    "merge_memory_update",
    "save_memory_for_agent",
    "summarize_save_result",
    "update_agent_memory_cache",
]
