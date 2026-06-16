"""Manifest-driven stage tool resolution and runtime injection."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .stage_tool_registry import (
    create_registered_stage_tool,
    get_registered_stage_tool_ids,
)


logger = logging.getLogger(__name__)

REAL_STAGE_CODES: tuple[str, ...] = (
    "LC",
    "CD",
    "DD",
    "CI",
    "SD",
    "AD",
    "AR",
    "CIA",
)
ALLOWED_AGENT_TYPES: tuple[str, ...] = ("lawyer", "client", "judge", "receptionist")
ALLOWED_STAGE_ROLE_NAMES: tuple[str, ...] = (
    "client",
    "lawyer",
    "plaintiff",
    "defendant",
    "judge",
    "plaintiff_lawyer",
    "defendant_lawyer",
    "appellant",
    "appellee",
    "appellant_lawyer",
    "appellee_lawyer",
)
MANIFEST_PATH = Path(__file__).with_name("stage_tool_manifest.yaml")
OPTIONAL_TOOL_FLAGS: dict[str, str] = {
    "search_laws": "SIMLAW_ENABLE_LAW_RETRIEVAL",
}
TRUE_ENV_VALUES = {"1", "true", "yes", "on", "enabled"}


def _normalize_string_list(values: Iterable[Any], *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        value = str(raw or "").strip()
        if not value:
            raise ValueError(f"{label} contains an empty entry")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _merge_string_lists(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in list(group or []):
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in TRUE_ENV_VALUES


def is_optional_tool_enabled(tool_id: str) -> bool:
    """Return whether an optional heavy tool should be injected at runtime."""
    flag_name = OPTIONAL_TOOL_FLAGS.get(str(tool_id or "").strip())
    if not flag_name:
        return True
    return _env_flag_enabled(flag_name)


def _filter_enabled_tool_ids(tool_ids: Iterable[str]) -> list[str]:
    return [tool_id for tool_id in list(tool_ids or []) if is_optional_tool_enabled(tool_id)]


def validate_stage_tool_manifest(payload: Any) -> dict[str, Any]:
    """Validate and normalize the stage tool manifest payload."""
    if not isinstance(payload, dict):
        raise ValueError("Stage tool manifest must be a mapping")

    registered_tool_ids = set(get_registered_stage_tool_ids())
    version = int(payload.get("version") or 0)
    if version <= 0:
        raise ValueError("Stage tool manifest version must be a positive integer")

    tool_registry_refs = _normalize_string_list(
        payload.get("tool_registry_refs") or [],
        label="tool_registry_refs",
    )
    unknown_registry_refs = sorted(set(tool_registry_refs) - registered_tool_ids)
    if unknown_registry_refs:
        raise ValueError(f"Unknown tool ids in tool_registry_refs: {unknown_registry_refs}")

    raw_defaults = payload.get("agent_type_defaults")
    if not isinstance(raw_defaults, dict):
        raise ValueError("agent_type_defaults must be a mapping")

    agent_type_defaults: dict[str, list[str]] = {}
    missing_agent_types = [agent_type for agent_type in ALLOWED_AGENT_TYPES if agent_type not in raw_defaults]
    if missing_agent_types:
        raise ValueError(f"agent_type_defaults missing agent types: {missing_agent_types}")

    for agent_type, raw_tool_ids in raw_defaults.items():
        normalized_agent_type = str(agent_type or "").strip()
        if normalized_agent_type not in ALLOWED_AGENT_TYPES:
            raise ValueError(f"Unknown agent type in manifest defaults: {agent_type}")
        tool_ids = _normalize_string_list(
            raw_tool_ids or [],
            label=f"agent_type_defaults.{normalized_agent_type}",
        )
        unknown_ids = sorted(set(tool_ids) - registered_tool_ids)
        if unknown_ids:
            raise ValueError(
                f"Unknown tool ids in agent_type_defaults.{normalized_agent_type}: {unknown_ids}"
            )
        undeclared_ids = sorted(set(tool_ids) - set(tool_registry_refs))
        if undeclared_ids:
            raise ValueError(
                f"Undeclared tool ids in agent_type_defaults.{normalized_agent_type}: {undeclared_ids}"
            )
        agent_type_defaults[normalized_agent_type] = tool_ids

    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, dict):
        raise ValueError("stages must be a mapping")

    missing_stage_codes = [stage_code for stage_code in REAL_STAGE_CODES if stage_code not in raw_stages]
    if missing_stage_codes:
        raise ValueError(f"Manifest missing stage definitions: {missing_stage_codes}")

    stages: dict[str, dict[str, Any]] = {}
    for stage_code, raw_stage_config in raw_stages.items():
        normalized_stage_code = str(stage_code or "").strip().upper()
        if normalized_stage_code not in REAL_STAGE_CODES:
            raise ValueError(f"Unknown stage code in manifest: {stage_code}")
        if not isinstance(raw_stage_config, dict):
            raise ValueError(f"Stage config must be a mapping: {normalized_stage_code}")

        shared_tools = _normalize_string_list(
            raw_stage_config.get("shared_tools") or [],
            label=f"stages.{normalized_stage_code}.shared_tools",
        )
        unknown_shared = sorted(set(shared_tools) - registered_tool_ids)
        if unknown_shared:
            raise ValueError(
                f"Unknown tool ids in stages.{normalized_stage_code}.shared_tools: {unknown_shared}"
            )

        raw_role_tools = raw_stage_config.get("role_tools")
        if raw_role_tools is None:
            raw_role_tools = {}
        if not isinstance(raw_role_tools, dict):
            raise ValueError(f"role_tools must be a mapping for stage {normalized_stage_code}")

        role_tools: dict[str, list[str]] = {}
        for role_name, raw_tool_ids in raw_role_tools.items():
            normalized_role_name = str(role_name or "").strip()
            if normalized_role_name not in ALLOWED_STAGE_ROLE_NAMES:
                raise ValueError(
                    f"Unknown role name in stages.{normalized_stage_code}.role_tools: {role_name}"
                )
            tool_ids = _normalize_string_list(
                raw_tool_ids or [],
                label=f"stages.{normalized_stage_code}.role_tools.{normalized_role_name}",
            )
            unknown_ids = sorted(set(tool_ids) - registered_tool_ids)
            if unknown_ids:
                raise ValueError(
                    f"Unknown tool ids in stages.{normalized_stage_code}.role_tools.{normalized_role_name}: "
                    f"{unknown_ids}"
                )
            role_tools[normalized_role_name] = tool_ids

        used_tool_ids = _merge_string_lists(shared_tools, *(role_tools.values()))
        undeclared_ids = sorted(set(used_tool_ids) - set(tool_registry_refs))
        if undeclared_ids:
            raise ValueError(
                f"Undeclared tool ids in stages.{normalized_stage_code}: {undeclared_ids}"
            )

        stages[normalized_stage_code] = {
            "shared_tools": shared_tools,
            "role_tools": role_tools,
        }

    return {
        "version": version,
        "tool_registry_refs": tool_registry_refs,
        "agent_type_defaults": agent_type_defaults,
        "stages": stages,
    }


@lru_cache(maxsize=1)
def load_stage_tool_manifest() -> dict[str, Any]:
    """Load and validate the on-disk stage tool manifest."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return validate_stage_tool_manifest(payload)


def clear_stage_tool_manifest_cache() -> None:
    """Clear the cached stage tool manifest."""
    load_stage_tool_manifest.cache_clear()


def _get_stage_config(stage_code: str) -> tuple[str, dict[str, Any]]:
    normalized_stage_code = str(stage_code or "").strip().upper()
    manifest = load_stage_tool_manifest()
    try:
        return normalized_stage_code, dict(manifest["stages"][normalized_stage_code])
    except KeyError as exc:
        raise ValueError(f"Unknown stage code: {stage_code}") from exc


def get_stage_declared_role_names(stage_code: str) -> list[str]:
    """Return all role names explicitly declared for one stage."""
    _, stage_config = _get_stage_config(stage_code)
    return list(dict(stage_config.get("role_tools") or {}).keys())


def validate_stage_role_name(stage_code: str, role_name: str) -> str:
    """Validate that one role name is both known globally and declared for the stage."""
    normalized_role_name = str(role_name or "").strip()
    if not normalized_role_name:
        raise ValueError(f"Stage role name is required for stage {stage_code}")
    if normalized_role_name not in ALLOWED_STAGE_ROLE_NAMES:
        raise ValueError(f"Unknown stage role name: {role_name}")

    normalized_stage_code, stage_config = _get_stage_config(stage_code)
    declared_roles = list(dict(stage_config.get("role_tools") or {}).keys())
    if normalized_role_name not in declared_roles:
        raise ValueError(
            f"Role '{normalized_role_name}' is not declared for stage {normalized_stage_code}"
        )
    return normalized_role_name


def _merge_tool_instances(*groups: Iterable[Any]) -> list[Any]:
    merged: dict[str, Any] = {}
    for group in groups:
        for tool in list(group or []):
            if tool is None or not hasattr(tool, "get_function_name"):
                continue
            merged[tool.get_function_name()] = tool
    return list(merged.values())


def _extract_tool_names(agent: Any) -> list[str]:
    tool_names: list[str] = []
    seen: set[str] = set()
    for tool in list(getattr(agent, "tools", []) or []):
        if not hasattr(tool, "get_function_name"):
            continue
        name = str(tool.get_function_name() or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        tool_names.append(name)
    return tool_names


def resolve_agent_type(agent: Any) -> str:
    """Resolve a normalized agent type string."""
    agent_type = str(getattr(agent, "agent_type", "") or "").strip().lower()
    if agent_type in ALLOWED_AGENT_TYPES:
        return agent_type

    class_name = type(agent).__name__
    type_map = {
        "ClientAgent": "client",
        "LawyerAgent": "lawyer",
        "JudgeAgent": "judge",
        "ReceptionistAgent": "receptionist",
    }
    resolved = type_map.get(class_name, "")
    if resolved in ALLOWED_AGENT_TYPES:
        return resolved
    raise ValueError(f"Unsupported agent type for stage tool resolution: {class_name}")


def get_agent_type_default_tool_ids(agent_type: str) -> list[str]:
    """Return configured default tool ids for one agent type."""
    normalized_agent_type = str(agent_type or "").strip().lower()
    manifest = load_stage_tool_manifest()
    try:
        return _filter_enabled_tool_ids(manifest["agent_type_defaults"][normalized_agent_type])
    except KeyError as exc:
        raise ValueError(f"Unknown agent type: {agent_type}") from exc


def build_agent_default_tools(
    agent_type: str,
    agent: Any,
    provided_tools: Iterable[Any] | None = None,
) -> list[Any]:
    """Build default tools for one agent type, with provided tools overriding by name."""
    default_tools: list[Any] = []
    for tool_id in get_agent_type_default_tool_ids(agent_type):
        try:
            default_tools.append(create_registered_stage_tool(tool_id, agent))
        except Exception as exc:
            logger.warning(
                "Failed to initialize default tool '%s' for agent type '%s': %s",
                tool_id,
                agent_type,
                exc,
            )
    return _merge_tool_instances(default_tools, provided_tools or [])


def get_stage_shared_tool_ids(stage_code: str) -> list[str]:
    """Return stage-level shared tool ids."""
    _, stage_config = _get_stage_config(stage_code)
    return _filter_enabled_tool_ids(stage_config["shared_tools"])


def get_stage_role_tool_ids(stage_code: str, role_name: str) -> list[str]:
    """Return one role's extra tool ids for a stage."""
    normalized_role_name = validate_stage_role_name(stage_code, role_name)
    _, stage_config = _get_stage_config(stage_code)
    role_tools = stage_config["role_tools"]
    return _filter_enabled_tool_ids(role_tools.get(normalized_role_name, []))


def resolve_configured_tool_names(stage_code: str, role_name: str, agent_type: str) -> list[str]:
    """Resolve configured tool names for one stage-role-agent_type tuple."""
    normalized_role_name = str(role_name or "").strip()
    stage_role_tool_ids: list[str] = []
    if normalized_role_name:
        stage_role_tool_ids = get_stage_role_tool_ids(stage_code, normalized_role_name)
    return _merge_string_lists(
        get_agent_type_default_tool_ids(agent_type),
        get_stage_shared_tool_ids(stage_code),
        stage_role_tool_ids,
    )


def resolve_configured_tool_ids_for_agent(
    stage_code: str,
    role_name: str,
    agent: Any,
) -> list[str]:
    """Resolve the fully configured tool ids for one stage-role-agent tuple."""
    return resolve_configured_tool_names(
        stage_code,
        role_name,
        resolve_agent_type(agent),
    )


def infer_stage_role_name(stage_code: str, agent: Any) -> str:
    """Infer a scenario role name for debug output from stage and agent fields."""
    normalized_stage_code = str(stage_code or "").strip().upper()
    agent_type = resolve_agent_type(agent)
    party_role = str(getattr(agent, "role", "") or "").strip().lower()
    court_role = str(getattr(agent, "court_role", "") or "").strip().lower()

    if normalized_stage_code == "LC":
        return "lawyer" if agent_type == "lawyer" else "client" if agent_type == "client" else ""
    if normalized_stage_code == "CD":
        return "lawyer" if agent_type == "lawyer" else "plaintiff" if agent_type == "client" else ""
    if normalized_stage_code == "DD":
        return "lawyer" if agent_type == "lawyer" else "defendant" if agent_type == "client" else ""
    if normalized_stage_code == "AD":
        return "lawyer" if agent_type == "lawyer" else "appellant" if agent_type == "client" else ""
    if normalized_stage_code == "AR":
        return "lawyer" if agent_type == "lawyer" else "appellee" if agent_type == "client" else ""
    if normalized_stage_code == "CI":
        if agent_type == "judge":
            return "judge"
        if agent_type == "client" and party_role in {"plaintiff", "defendant"}:
            return party_role
        if agent_type == "lawyer" and court_role in {"plaintiff", "defendant"}:
            return f"{court_role}_lawyer"
        return ""
    if normalized_stage_code == "CIA":
        if agent_type == "judge":
            return "judge"
        if agent_type == "client" and party_role in {"appellant", "appellee"}:
            return party_role
        if agent_type == "lawyer" and court_role in {"appellant", "appellee"}:
            return f"{court_role}_lawyer"
        return ""
    return ""


def _update_agent_stage_tool_context(
    agent: Any,
    *,
    stage_code: str,
    role_name: str,
    configured_tool_names: list[str],
) -> list[str]:
    available_tool_names = _extract_tool_names(agent)
    setattr(agent, "_simlaw_stage_code", str(stage_code or "").strip().upper())
    setattr(agent, "_simlaw_stage_role", str(role_name or "").strip())
    setattr(agent, "_simlaw_configured_tool_names", list(configured_tool_names))
    setattr(agent, "_simlaw_available_tool_names", list(available_tool_names))

    chat_agent = getattr(agent, "chat_agent", None)
    owner_meta = dict(getattr(chat_agent, "_simlaw_owner_meta", {}) or {})
    owner_meta["stage_code"] = getattr(agent, "_simlaw_stage_code", "")
    owner_meta["agent_role"] = getattr(agent, "_simlaw_stage_role", "")
    owner_meta["configured_tool_names"] = list(configured_tool_names)
    owner_meta["available_tool_names"] = list(available_tool_names)
    if chat_agent is not None:
        setattr(chat_agent, "_simlaw_owner_meta", owner_meta)
    return available_tool_names


def apply_stage_tool_permissions(
    stage_code: str,
    role_to_agent: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Apply manifest-declared stage tools to already-participating agents."""
    normalized_stage_code = str(stage_code or "").strip().upper()
    if normalized_stage_code not in REAL_STAGE_CODES:
        raise ValueError(f"Unknown stage code: {stage_code}")

    resolved: dict[str, list[str]] = {}
    for role_name, agent in dict(role_to_agent or {}).items():
        if agent is None:
            continue

        normalized_role_name = validate_stage_role_name(normalized_stage_code, role_name)
        configured_tool_ids = resolve_configured_tool_ids_for_agent(
            normalized_stage_code,
            normalized_role_name,
            agent,
        )

        existing_tool_names = set(_extract_tool_names(agent))
        missing_tool_ids = [
            tool_id
            for tool_id in configured_tool_ids
            if tool_id not in existing_tool_names
        ]

        if missing_tool_ids:
            new_tools = [
                create_registered_stage_tool(tool_id, agent)
                for tool_id in missing_tool_ids
            ]
            if hasattr(agent, "add_runtime_tools"):
                agent.add_runtime_tools(new_tools)
            else:
                merged_tools = _merge_tool_instances(
                    getattr(agent, "tools", []) or [],
                    new_tools,
                )
                setattr(agent, "tools", merged_tools)

        resolved[normalized_role_name] = _update_agent_stage_tool_context(
            agent,
            stage_code=normalized_stage_code,
            role_name=normalized_role_name,
            configured_tool_names=configured_tool_ids,
        )

    return resolved


def describe_stage_tool_matrix() -> dict[str, dict[str, list[str]]]:
    """Return stage -> role -> configured tool names for inspection/tests."""
    manifest = load_stage_tool_manifest()
    matrix: dict[str, dict[str, list[str]]] = {}
    for stage_code in REAL_STAGE_CODES:
        stage_roles = manifest["stages"][stage_code]["role_tools"]
        matrix[stage_code] = {}
        for role_name in stage_roles:
            default_agent_type = "lawyer"
            if role_name in {"client", "plaintiff", "defendant", "appellant", "appellee"}:
                default_agent_type = "client"
            elif role_name == "judge":
                default_agent_type = "judge"
            matrix[stage_code][role_name] = resolve_configured_tool_names(
                stage_code,
                role_name,
                default_agent_type,
            )
    return matrix


__all__ = [
    "ALLOWED_AGENT_TYPES",
    "ALLOWED_STAGE_ROLE_NAMES",
    "MANIFEST_PATH",
    "REAL_STAGE_CODES",
    "apply_stage_tool_permissions",
    "build_agent_default_tools",
    "clear_stage_tool_manifest_cache",
    "describe_stage_tool_matrix",
    "get_agent_type_default_tool_ids",
    "get_stage_declared_role_names",
    "get_stage_role_tool_ids",
    "get_stage_shared_tool_ids",
    "infer_stage_role_name",
    "is_optional_tool_enabled",
    "load_stage_tool_manifest",
    "resolve_agent_type",
    "resolve_configured_tool_ids_for_agent",
    "resolve_configured_tool_names",
    "validate_stage_role_name",
    "validate_stage_tool_manifest",
]
