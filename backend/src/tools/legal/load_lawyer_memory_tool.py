from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from camel.toolkits import FunctionTool

from ...utils.live_card_memory import (
    LAWYER_LOAD_TOOL_NAME,
    LAWYER_MEMORY_OWNER,
    bootstrap_memory_from_legacy,
    build_default_memory,
    normalize_memory_payload,
    render_memory_yaml,
    resolve_memory_paths,
    update_agent_memory_cache,
)


LOAD_LAWYER_MEMORY_DESCRIPTION = "读取当前案件下律师的 memory.yaml，返回完整 YAML。"


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


def _load_legacy_lawyer_memory(agent: Any, memory_path: Path) -> dict[str, Any] | None:
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


def _ensure_lawyer_memory_file(agent: Any) -> Path:
    paths = resolve_memory_paths(agent, LAWYER_MEMORY_OWNER)
    memory_path = paths.memory_path
    setattr(agent, "memory_yaml_path", str(memory_path))
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    if memory_path.exists():
        payload = yaml.safe_load(memory_path.read_text(encoding="utf-8"))
        normalized = normalize_memory_payload(LAWYER_MEMORY_OWNER, payload)
        rendered = render_memory_yaml(LAWYER_MEMORY_OWNER, normalized)
        memory_path.write_text(rendered, encoding="utf-8")
        update_agent_memory_cache(agent, LAWYER_MEMORY_OWNER, normalized)
        return memory_path

    legacy_payload = _load_legacy_lawyer_memory(agent, memory_path)
    if isinstance(legacy_payload, dict):
        normalized = bootstrap_memory_from_legacy(LAWYER_MEMORY_OWNER, legacy_payload)
    else:
        normalized = build_default_memory(LAWYER_MEMORY_OWNER, agent=agent)

    memory_path.write_text(render_memory_yaml(LAWYER_MEMORY_OWNER, normalized), encoding="utf-8")
    update_agent_memory_cache(agent, LAWYER_MEMORY_OWNER, normalized)
    return memory_path


class LoadLawyerMemoryTool:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def load_lawyer_memory(
        self,
        content: Any | None = None,
        **_: Any,
    ) -> str:
        # Some models incorrectly echo a meaningless `content` field for zero-arg load tools.
        # Ignore it so memory checkpointing stays robust.
        _ = content
        memory_path = _ensure_lawyer_memory_file(self.agent)
        payload = yaml.safe_load(memory_path.read_text(encoding="utf-8"))
        normalized = normalize_memory_payload(LAWYER_MEMORY_OWNER, payload)
        rendered = render_memory_yaml(LAWYER_MEMORY_OWNER, normalized)
        memory_path.write_text(rendered, encoding="utf-8")
        update_agent_memory_cache(self.agent, LAWYER_MEMORY_OWNER, normalized)
        return rendered


def create_load_lawyer_memory_tool(agent: Any) -> FunctionTool:
    impl = LoadLawyerMemoryTool(agent)
    return FunctionTool(
        impl.load_lawyer_memory,
        openai_tool_schema={
            "type": "function",
            "function": {
                "name": LAWYER_LOAD_TOOL_NAME,
                "description": LOAD_LAWYER_MEMORY_DESCRIPTION,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": ["string", "null"],
                            "description": "Optional redundant field; ignored.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        },
    )


__all__ = [
    "LOAD_LAWYER_MEMORY_DESCRIPTION",
    "LoadLawyerMemoryTool",
    "create_load_lawyer_memory_tool",
]
