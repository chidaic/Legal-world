"""Registry for manifest-addressable runtime tools."""

from __future__ import annotations

from typing import Any, Callable

from camel.toolkits import FunctionTool

from ..tools import (
    create_appeal_drafting_tool,
    create_appeal_response_drafting_tool,
    create_complaint_drafting_tool,
    create_defense_drafting_tool,
    create_first_instance_judgment_drafting_tool,
    create_law_retrieval_tool,
    create_second_instance_judgment_drafting_tool,
    create_save_client_memory_tool,
    create_save_lawyer_memory_tool,
)


ToolFactory = Callable[[Any], FunctionTool]


def _create_search_laws(agent: Any) -> FunctionTool:
    return create_law_retrieval_tool(agent=agent)


REGISTERED_STAGE_TOOL_FACTORIES: dict[str, ToolFactory] = {
    "search_laws": _create_search_laws,
    "save_client_memory": create_save_client_memory_tool,
    "save_lawyer_memory": create_save_lawyer_memory_tool,
    "draft_complaint_document": create_complaint_drafting_tool,
    "draft_defense_document": create_defense_drafting_tool,
    "draft_appeal_document": create_appeal_drafting_tool,
    "draft_appeal_response_document": create_appeal_response_drafting_tool,
    "draft_first_instance_judgment_document": create_first_instance_judgment_drafting_tool,
    "draft_second_instance_judgment_document": create_second_instance_judgment_drafting_tool,
}


def get_registered_stage_tool_ids() -> list[str]:
    """Return all manifest-available tool ids."""
    return list(REGISTERED_STAGE_TOOL_FACTORIES.keys())


def is_registered_stage_tool(tool_id: str) -> bool:
    """Check whether a tool id exists in the registry."""
    return str(tool_id or "").strip() in REGISTERED_STAGE_TOOL_FACTORIES


def create_registered_stage_tool(tool_id: str, agent: Any) -> FunctionTool:
    """Instantiate one configured runtime tool for an agent."""
    normalized = str(tool_id or "").strip()
    try:
        factory = REGISTERED_STAGE_TOOL_FACTORIES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown stage tool id: {tool_id}") from exc
    return factory(agent)


__all__ = [
    "REGISTERED_STAGE_TOOL_FACTORIES",
    "create_registered_stage_tool",
    "get_registered_stage_tool_ids",
    "is_registered_stage_tool",
]
