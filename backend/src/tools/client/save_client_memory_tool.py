from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from camel.toolkits import FunctionTool

from ...utils.live_card_memory import (
    CLIENT_MEMORY_OWNER,
    CLIENT_SAVE_TOOL_NAME,
    apply_memory_operations,
    bootstrap_memory_from_legacy,
    build_default_memory,
    build_history_entry,
    build_save_operations_description,
    build_save_tool_description,
    normalize_memory_payload,
    render_memory_yaml,
    resolve_memory_paths,
    summarize_save_result,
    update_agent_memory_cache,
)


SAVE_CLIENT_MEMORY_DESCRIPTION = build_save_tool_description(CLIENT_MEMORY_OWNER)
SAVE_CLIENT_MEMORY_OPERATIONS_DESCRIPTION = build_save_operations_description(CLIENT_MEMORY_OWNER)


def _legacy_json_candidates(agent: Any, memory_path: Path) -> list[Path]:
    candidates: list[Path] = []
    explicit = str(getattr(agent, "long_term_memory_path", "") or "").strip()
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.suffix.lower() != ".json":
            explicit_path = explicit_path / "long_term_memory.json"
        candidates.append(explicit_path.resolve())
    candidates.append(memory_path.with_name("long_term_memory.json"))
    return candidates


def _load_legacy_client_memory(agent: Any, memory_path: Path) -> dict[str, Any] | None:
    seen: set[str] = set()
    for candidate in _legacy_json_candidates(agent, memory_path):
        key = str(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    storage = getattr(agent, "storage", None)
    config_path = getattr(agent, "config_path", None)
    if storage and config_path:
        try:
            config = storage.load_agent_config(config_path)
        except Exception:
            config = {}
        legacy = config.get("long_term_memory")
        if isinstance(legacy, dict):
            return legacy
    return None


def _ensure_client_memory_file(agent: Any) -> tuple[dict[str, Any], Path, Path]:
    paths = resolve_memory_paths(agent, CLIENT_MEMORY_OWNER)
    memory_path = paths.memory_path
    history_path = paths.history_path
    setattr(agent, "memory_yaml_path", str(memory_path))
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    if memory_path.exists():
        payload = yaml.safe_load(memory_path.read_text(encoding="utf-8"))
        normalized = normalize_memory_payload(CLIENT_MEMORY_OWNER, payload)
        rendered = render_memory_yaml(CLIENT_MEMORY_OWNER, normalized)
        memory_path.write_text(rendered, encoding="utf-8")
        update_agent_memory_cache(agent, CLIENT_MEMORY_OWNER, normalized)
        return normalized, memory_path, history_path

    legacy_payload = _load_legacy_client_memory(agent, memory_path)
    if isinstance(legacy_payload, dict):
        normalized = bootstrap_memory_from_legacy(CLIENT_MEMORY_OWNER, legacy_payload)
    else:
        normalized = build_default_memory(CLIENT_MEMORY_OWNER, agent=agent)

    memory_path.write_text(render_memory_yaml(CLIENT_MEMORY_OWNER, normalized), encoding="utf-8")
    update_agent_memory_cache(agent, CLIENT_MEMORY_OWNER, normalized)
    return normalized, memory_path, history_path


class SaveClientMemoryTool:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def save_client_memory(self, operations: list[dict[str, Any]]) -> str:
        before_payload, memory_path, history_path = _ensure_client_memory_file(self.agent)
        after_payload, revision_ops_override = apply_memory_operations(
            CLIENT_MEMORY_OWNER,
            before_payload,
            operations,
        )

        memory_path.write_text(
            render_memory_yaml(CLIENT_MEMORY_OWNER, after_payload),
            encoding="utf-8",
        )
        update_agent_memory_cache(self.agent, CLIENT_MEMORY_OWNER, after_payload)

        history_entries: list[dict[str, Any]] = []
        if history_path.exists():
            try:
                existing = yaml.safe_load(history_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
            if isinstance(existing, list):
                history_entries = [item for item in existing if isinstance(item, dict)]

        history_entry = build_history_entry(
            memory_owner=CLIENT_MEMORY_OWNER,
            before_payload=before_payload,
            after_payload=after_payload,
            source_stage=str(getattr(self.agent, "_simlaw_stage_code", "") or "").strip().upper(),
            revision_ops_override=revision_ops_override,
        )
        history_entries.append(history_entry)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            yaml.safe_dump(
                history_entries,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )

        return summarize_save_result(
            {
                "paths": type("MemoryPaths", (), {"memory_path": memory_path, "history_path": history_path})(),
                "history_entry": history_entry,
                "before": before_payload,
                "after": after_payload,
            }
        )


def normalize_client_memory(payload: Any) -> dict[str, Any]:
    return normalize_memory_payload(CLIENT_MEMORY_OWNER, payload)


def create_save_client_memory_tool(agent: Any) -> FunctionTool:
    impl = SaveClientMemoryTool(agent)
    return FunctionTool(
        impl.save_client_memory,
        openai_tool_schema={
            "type": "function",
            "function": {
                "name": CLIENT_SAVE_TOOL_NAME,
                "description": SAVE_CLIENT_MEMORY_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "description": SAVE_CLIENT_MEMORY_OPERATIONS_DESCRIPTION,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {
                                        "type": "string",
                                        "enum": [
                                            "case_knowledge.self_narrative",
                                            "case_knowledge.case_stage",
                                            "demands.core_demands",
                                        ],
                                        "description": "要操作的当事人 memory 末级字段路径。",
                                    },
                                    "operation": {
                                        "type": "string",
                                        "enum": ["revise", "expand"],
                                        "description": "revise=覆盖该字段；expand=在该字段原内容后追加。",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "用于覆盖或追加到该字段的具体文本。",
                                    },
                                },
                                "required": ["field", "operation", "content"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["operations"],
                    "additionalProperties": False,
                },
            },
        },
    )


__all__ = [
    "SAVE_CLIENT_MEMORY_DESCRIPTION",
    "SAVE_CLIENT_MEMORY_OPERATIONS_DESCRIPTION",
    "SaveClientMemoryTool",
    "create_save_client_memory_tool",
    "normalize_client_memory",
]
