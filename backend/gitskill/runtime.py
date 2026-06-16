from __future__ import annotations

from typing import Any, Iterable


def _tool_name(tool: Any) -> str:
    return tool.get_function_name()


def apply_reflection_stage(
    agent: Any,
    system_prompt: str,
    skill_dirs: list[str] | None = None,
    extra_tools: Iterable[Any] | None = None,
) -> None:
    """Apply one reflection-stage runtime configuration to an active agent."""
    extra_tools = [tool for tool in list(extra_tools or []) if tool is not None]
    desired_tool_names = {_tool_name(tool) for tool in extra_tools}
    base_tool_names = {_tool_name(tool) for tool in getattr(agent, "base_tools", [])}
    removable_names: list[str] = []

    for tool in list(getattr(agent, "tools", []) or []):
        name = _tool_name(tool)
        if name in base_tool_names or name == "load_skill" or name in desired_tool_names:
            continue
        removable_names.append(name)

    agent.update_runtime_prompt(system_prompt)
    if removable_names:
        agent.remove_runtime_tools(removable_names)
    agent.replace_runtime_skills(skill_dirs or [])

    existing_names = {_tool_name(tool) for tool in list(getattr(agent, "tools", []) or [])}
    additions = [tool for tool in extra_tools if _tool_name(tool) not in existing_names]
    if additions:
        agent.add_runtime_tools(additions)
