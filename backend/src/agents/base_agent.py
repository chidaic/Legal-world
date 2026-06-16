"""Base agent class for SimLawFirm framework.

Supports two modes:
- Shell mode (default): No LLM, just metadata + event handlers. Used in sandbox.
- Active mode: LLM-powered via CAMEL ChatAgent. Activated when entering a scenario.

Pipeline mode uses create_for_pipeline() which creates and immediately activates.
"""

import copy
import json
import logging
import re
import sys
import time
import inspect
from typing import Any, Dict, Optional, TYPE_CHECKING
from abc import ABC
from pathlib import Path
from datetime import datetime

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType
from pydantic import BaseModel

from ..utils.model_config import (
    DEFAULT_RUNTIME_OPENAI_MODEL,
    build_runtime_openai_chat_config,
    resolve_openai_chat_model,
)
from ..utils.chat_agent_runtime_patch import (
    patch_chat_agent_usage_cache,
    patch_chat_agent_usage_serialization,
)
from ..utils.runtime_flags import system_prompt_print_enabled

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager


logger = logging.getLogger(__name__)

patch_chat_agent_usage_serialization()

DEFAULT_STEP_TIMEOUT_SECONDS = 180
DEFAULT_TOOL_CALL_LIMIT_PER_STEP = 6
RUNTIME_TECH_PROGRESS_PHASE = "runtime_tech_used"


def _dedupe_strings(values: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _extract_tool_call_name(record: Any) -> str:
    if isinstance(record, dict):
        for key in ("tool_name", "name", "tool", "function_name"):
            value = str(record.get(key) or "").strip()
            if value:
                return value
        raw_record = record.get("raw_record")
        if raw_record is not None and raw_record is not record:
            return _extract_tool_call_name(raw_record)
        return ""

    for attr in ("tool_name", "name", "tool", "function_name"):
        value = str(getattr(record, attr, "") or "").strip()
        if value:
            return value
    return ""


def _extract_tool_call_arguments(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        for key in ("args", "kwargs", "arguments", "parameters", "input_kwargs"):
            value = record.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
        raw_record = record.get("raw_record")
        if raw_record is not None and raw_record is not record:
            return _extract_tool_call_arguments(raw_record)
        return {}

    for attr in ("args", "kwargs", "arguments", "parameters", "input_kwargs"):
        value = getattr(record, attr, None)
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    return {}


def _extract_skill_names_from_tool_calls(tool_calls: Any) -> list[str]:
    skill_names: list[str] = []
    for record in list(tool_calls or []):
        if _extract_tool_call_name(record) != "load_skill":
            continue
        args = _extract_tool_call_arguments(record)
        requested = args.get("names")
        if isinstance(requested, str):
            skill_names.append(requested)
        elif isinstance(requested, list):
            skill_names.extend(str(item or "").strip() for item in requested)
    return _dedupe_strings(skill_names)


def _extract_skill_names_from_usage_log(skill_usage_log: Any) -> list[str]:
    skill_names: list[str] = []
    for record in list(skill_usage_log or []):
        if not isinstance(record, dict):
            continue
        requested = record.get("requested")
        if isinstance(requested, list):
            skill_names.extend(str(item or "").strip() for item in requested)
        elif isinstance(requested, str):
            skill_names.append(requested)
        for item in list(record.get("resolved") or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("skill_path") or "").strip()
            if name:
                skill_names.append(name)
    return _dedupe_strings(skill_names)


def _record_tool_call_guard(agent: Any, tool_name: str) -> None:
    state = getattr(agent, "_simlaw_step_guard_state", None)
    if not isinstance(state, dict):
        return

    normalized_tool_name = str(tool_name or "").strip()
    if not normalized_tool_name:
        return

    tool_total_counts = state.setdefault("tool_total_counts", {})
    total_count = int(tool_total_counts.get(normalized_tool_name, 0)) + 1
    tool_total_counts[normalized_tool_name] = total_count

    if total_count > DEFAULT_TOOL_CALL_LIMIT_PER_STEP:
        raise RuntimeError(
            f"Tool '{normalized_tool_name}' exceeded the per-step call limit "
            f"({DEFAULT_TOOL_CALL_LIMIT_PER_STEP}); aborting to prevent runaway loops."
        )


def _safe_console_print(text: str) -> None:
    """Print text safely on consoles with limited encodings."""
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return

    encoding = getattr(stream, "encoding", None) or "utf-8"
    payload = (text + "\n").encode(encoding, errors="backslashreplace")
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(payload)
        buffer.flush()
        return

    stream.write(payload.decode(encoding, errors="ignore"))
    stream.flush()


DEFAULT_OPENAI_COMPAT_MODEL = DEFAULT_RUNTIME_OPENAI_MODEL


def _resolve_model_type(model_type: ModelType = None) -> str:
    """Check env override for model, return resolved type.

    Defaults to Qwen 3.5 Flash if no override specified.

    Args:
        model_type: Optional fallback model type (unused, kept for compatibility)

    Returns:
        Model name string (e.g., "qwen3.5-flash")
    """
    resolved_model = resolve_openai_chat_model(
        explicit_model=model_type,
        default_model=DEFAULT_OPENAI_COMPAT_MODEL,
    )
    logger.debug("Using resolved runtime model: %s", resolved_model)
    return resolved_model

class BaseAgent(ABC):
    """Base class for all agents in SimLawFirm.

    Shell mode: agent_id, name, event_bus, storage, config_path — no LLM.
    Active mode: additionally has chat_agent with LLM (call activate()).
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str = "",
        work_memory_path: Optional[str] = None,
        model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
        model_type: Optional[str] = None,  # Changed to Optional[str] for DeepSeek compatibility
        tools: Optional[list] = None,
        skill_dirs: Optional[list[str]] = None,
        event_bus: Optional["EventBus"] = None,
        storage: Optional["FileStorageManager"] = None,
        config_path: Optional[str] = None,
        **kwargs
    ):
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.base_tools = list(tools or [])
        self.tools = list(self.base_tools)
        self.skill_dirs = list(skill_dirs or [])
        self.skill_usage_log: list[dict[str, Any]] = []
        self.config = kwargs

        # Sandbox mode attributes
        self.event_bus = event_bus
        self.storage = storage
        self.config_path = config_path
        self.model_platform = model_platform
        self.model_type = model_type
        self.step_timeout_seconds = int(
            kwargs.pop("step_timeout_seconds", DEFAULT_STEP_TIMEOUT_SECONDS)
            or DEFAULT_STEP_TIMEOUT_SECONDS
        )

        # Shell mode: no LLM by default
        self.chat_agent: Optional[ChatAgent] = None
        self._is_active = False
        self.current_scenario_id: Optional[str] = None  # Track current scenario participation
        self._last_step_info: Dict[str, Any] = {}
        self._last_tool_call_records: list[Any] = []
        self._simlaw_step_guard_state: Dict[str, Any] = {}
        self._simlaw_step_guard_counter: int = 0
        self._simlaw_runtime_tech_callback = kwargs.pop("runtime_tech_callback", None)

        # Work memory
        self.work_memory: Dict[str, Any] = {}
        if work_memory_path:
            self._load_work_memory(work_memory_path)

        # If system_prompt is provided, auto-activate (pipeline compat)
        if system_prompt:
            self.activate(
                system_prompt,
                model_platform,
                model_type,
                tools=self.base_tools,
                skill_dirs=self.skill_dirs,
            )

        logger.info(f"Initialized agent '{self.name}' (ID: {self.agent_id}, active={self._is_active})")

    # ── Activation / Deactivation ──

    def activate(
        self,
        system_prompt: str,
        model_platform: Optional[ModelPlatformType] = None,
        model_type: Optional[str] = None,  # Changed to Optional[str] for DeepSeek compatibility
        tools: Optional[list] = None,
        skill_dirs: Optional[list] = None,
        debug_output_dir: Optional[str] = None,
        scenario_id: Optional[str] = None,
        step_timeout_seconds: Optional[int] = None,
    ) -> None:
        """Create the underlying ChatAgent with LLM. Called when entering a scenario.

        Args:
            skill_dirs: Optional list of directories for loading SKILL files.
                       按优先级从低到高排列，后面的同名 SKILL 覆盖前面的。
            debug_output_dir: Optional directory to save debug info (system prompt)
            scenario_id: Optional scenario ID to track agent participation
        """
        self.system_prompt = system_prompt
        if model_platform:
            self.model_platform = model_platform
        if model_type:
            self.model_type = model_type
        if tools is not None:
            self.base_tools = list(tools)
        if skill_dirs is not None:
            self.skill_dirs = list(skill_dirs)
        if scenario_id:
            self.current_scenario_id = scenario_id
        if step_timeout_seconds is not None:
            self.step_timeout_seconds = int(step_timeout_seconds or DEFAULT_STEP_TIMEOUT_SECONDS)

        active_tools = list(self.base_tools)
        if self.skill_dirs:
            from ..tools.common import load_agent_skills

            skill_tools = load_agent_skills(
                self.skill_dirs,
                usage_recorder=self._record_skill_usage,
            )
            if skill_tools:
                active_tools = self._merge_tools(active_tools, skill_tools)
        self.tools = self._wrap_tools_with_step_guard(active_tools)

        # Save debug info if requested
        if debug_output_dir:
            self._save_debug_info(debug_output_dir)

        resolved_model = _resolve_agent_model_type(self, self.model_type)
        api_base_url = _resolve_agent_api_base_url(self)
        api_key = _resolve_agent_api_key(self)
        model_factory_kwargs: dict[str, Any] = {}
        if api_key:
            model_factory_kwargs["api_key"] = api_key
        if api_base_url:
            model_factory_kwargs["url"] = api_base_url
        model_config = ModelFactory.create(
            model_platform=self.model_platform,
            model_type=resolved_model,
            model_config_dict=build_runtime_openai_chat_config(
                model_name=resolved_model,
                temperature=0.5,
            ),
        )
        self.chat_agent = ChatAgent(
            system_message=self.system_prompt,
            model=model_config,
            tools=self.tools,
            step_timeout=self.step_timeout_seconds,
        )
        patch_chat_agent_usage_cache(self.chat_agent)
        self._is_active = True
        logger.info(f"[{self.name}] Activated with LLM")

        if system_prompt_print_enabled():
            _safe_console_print("")
            _safe_console_print("=" * 80)
            _safe_console_print(f"[DEBUG] Agent activated: {self.name} (ID: {self.agent_id})")
            _safe_console_print("-" * 80)
            _safe_console_print("[System Prompt]")
            _safe_console_print(self.system_prompt)
            _safe_console_print("=" * 80)

    def _require_active_chat_agent(self) -> ChatAgent:
        """Return the active chat agent or raise when unavailable."""
        if not self._is_active or not self.chat_agent:
            raise RuntimeError(f"Agent '{self.name}' is not active. Call activate() first.")
        return self.chat_agent

    def update_runtime_prompt(self, system_prompt: str, reset_memory: bool = True) -> None:
        """Update the active system prompt without recreating the agent instance."""
        chat_agent = self._require_active_chat_agent()
        self.system_prompt = system_prompt
        chat_agent.update_system_message(system_prompt, reset_memory=reset_memory)
        logger.info("[%s] Runtime system prompt updated", self.name)

    def update_runtime_step_timeout(self, timeout_seconds: float) -> None:
        """Update the active step timeout for the underlying chat agent."""
        chat_agent = self._require_active_chat_agent()
        self.step_timeout_seconds = int(
            timeout_seconds or self.step_timeout_seconds or DEFAULT_STEP_TIMEOUT_SECONDS
        )
        chat_agent.step_timeout = self.step_timeout_seconds
        logger.info("[%s] Runtime step timeout updated to %ss", self.name, self.step_timeout_seconds)

    def add_runtime_tools(self, tools: list) -> None:
        """Add tools to the active agent runtime."""
        additions = [tool for tool in list(tools or []) if tool is not None]
        if not additions:
            return

        chat_agent = self._require_active_chat_agent()
        existing_names = {
            tool.get_function_name()
            for tool in list(self.tools or [])
            if tool is not None and hasattr(tool, "get_function_name")
        }
        deduped_additions = [
            tool
            for tool in additions
            if hasattr(tool, "get_function_name")
            and tool.get_function_name() not in existing_names
        ]
        if not deduped_additions:
            return

        wrapped_additions = self._wrap_tools_with_step_guard(deduped_additions)
        chat_agent.add_tools(wrapped_additions)
        self.tools = self._merge_tools(self.tools, wrapped_additions)
        logger.info(
            "[%s] Added runtime tools: %s",
            self.name,
            [tool.get_function_name() for tool in deduped_additions],
        )

    def remove_runtime_tools(self, tool_names: list[str]) -> None:
        """Remove tools from the active agent runtime by name."""
        names = [str(name).strip() for name in list(tool_names or []) if str(name).strip()]
        if not names:
            return

        chat_agent = self._require_active_chat_agent()
        chat_agent.remove_tools(names)
        blocked = set(names)
        self.tools = [tool for tool in self.tools if tool.get_function_name() not in blocked]
        logger.info("[%s] Removed runtime tools: %s", self.name, names)

    def replace_runtime_skills(self, skill_dirs: Optional[list[str]] = None) -> None:
        """Replace the active skill directories and rebuild the load_skill tool."""
        chat_agent = self._require_active_chat_agent()
        del chat_agent

        from ..tools.common import load_agent_skills, normalize_skill_dirs

        normalized_dirs = normalize_skill_dirs(skill_dirs or [])
        self.skill_dirs = normalized_dirs

        self.remove_runtime_tools(["load_skill"])

        if not normalized_dirs:
            logger.info("[%s] Cleared runtime skills", self.name)
            return

        skill_tools = load_agent_skills(
            normalized_dirs,
            usage_recorder=self._record_skill_usage,
        )
        if skill_tools:
            self.add_runtime_tools(skill_tools)
        else:
            logger.info("[%s] No visible skills found for runtime dirs: %s", self.name, normalized_dirs)

    def _save_debug_info(self, output_dir: str) -> None:
        """Save debug information (system prompt and scenario data) to JSON file.

        Args:
            output_dir: Directory to save debug files
        """
        import json
        from pathlib import Path
        from datetime import datetime

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_id}_{timestamp}_debug.json"
        filepath = output_path / filename

        debug_data = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "agent_type": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": self.system_prompt,
            "scenario_data": getattr(self, "scenario_data", {}),
            "tools_count": len(self.tools) if self.tools else 0,
            "skill_usage": self.get_skill_usage_report(),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[{self.name}] Debug info saved to {filepath}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to save debug info: {e}")

    def deactivate(self) -> None:
        """Destroy ChatAgent, return to shell mode."""
        self.chat_agent = None
        self._is_active = False
        self.current_scenario_id = None
        self.system_prompt = ""
        logger.info(f"[{self.name}] Deactivated, back to shell mode")

    def _record_skill_usage(self, payload: dict[str, Any]) -> None:
        resolved = payload.get("resolved") or []
        if not resolved:
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "requested": list(payload.get("requested") or []),
            "missing": list(payload.get("missing") or []),
            "resolved": [
                {
                    "name": item.get("name"),
                    "skill_path": item.get("skill_path"),
                    "path": item.get("path"),
                    "source_root": item.get("source_root"),
                }
                for item in resolved
                if isinstance(item, dict)
            ],
        }
        self.skill_usage_log.append(record)

    def reset_skill_usage_report(self) -> None:
        self.skill_usage_log = []

    def get_skill_usage_report(self) -> Dict[str, Any]:
        summary_by_skill: dict[tuple[str, str, str], dict[str, Any]] = {}
        total_skills_loaded = 0

        for call in self.skill_usage_log:
            for resolved in call.get("resolved", []):
                name = str(resolved.get("name") or "")
                skill_path = str(resolved.get("skill_path") or "")
                source_root = str(resolved.get("source_root") or "")
                key = (name, skill_path, source_root)
                bucket = summary_by_skill.setdefault(
                    key,
                    {
                        "name": name,
                        "skill_path": skill_path,
                        "source_root": source_root,
                        "path": str(resolved.get("path") or ""),
                        "load_count": 0,
                    },
                )
                bucket["load_count"] += 1
                total_skills_loaded += 1

        skills = sorted(
            summary_by_skill.values(),
            key=lambda item: (
                -int(item.get("load_count", 0)),
                str(item.get("name", "")),
                str(item.get("skill_path", "")),
            ),
        )
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "tool_call_count": len(self.skill_usage_log),
            "skill_load_count": total_skills_loaded,
            "skills": skills,
            "tool_calls": list(self.skill_usage_log),
        }

    def recover_from_error(self) -> None:
        """Recover agent from error state without full deactivation.

        Resets the chat agent's memory to clear any stuck state,
        but preserves activation status and scenario participation.
        This allows the agent to continue in subsequent scenarios.
        """
        if self.chat_agent:
            try:
                self.chat_agent.reset()
                logger.info(f"[{self.name}] Recovered from error - memory reset, still active")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to reset chat agent during recovery: {e}")
                # If reset fails, fall back to full deactivation
                self.deactivate()
        else:
            logger.warning(f"[{self.name}] No chat agent to recover, deactivating")
            self.deactivate()

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def agent_type(self) -> str:
        """Return a simplified agent type string based on the class name."""
        cls_name = self.__class__.__name__
        type_map = {
            "ClientAgent": "client",
            "LawyerAgent": "lawyer",
            "JudgeAgent": "judge",
            "ReceptionistAgent": "receptionist",
        }
        return type_map.get(cls_name, "unknown")

    def is_busy(self) -> bool:
        """Check if agent is busy (active and participating in a scenario).

        Returns:
            True if agent is active and has a current scenario, False otherwise
        """
        return self._is_active and self.current_scenario_id is not None

    @classmethod
    def create_for_pipeline(cls, **kwargs) -> "BaseAgent":
        """Factory for pipeline mode: creates agent with system_prompt (auto-activates)."""
        return cls(**kwargs)

    @staticmethod
    def _merge_tools(base_tools: list, extra_tools: list) -> list:
        """Merge tools by function name while preserving the latest definition."""
        merged = {}
        for tool in list(base_tools or []) + list(extra_tools or []):
            if tool is None:
                continue
            merged[tool.get_function_name()] = tool

        prioritized_names = [
            "load_skill",
        ]
        prioritized_tools = []
        for name in prioritized_names:
            tool = merged.pop(name, None)
            if tool is not None:
                prioritized_tools.append(tool)

        return prioritized_tools + list(merged.values())

    def _wrap_tools_with_step_guard(self, tools: list) -> list:
        return [self._wrap_single_tool_with_step_guard(tool) for tool in list(tools or [])]

    def _wrap_single_tool_with_step_guard(self, tool: Any) -> Any:
        if tool is None or getattr(tool, "_simlaw_guard_wrapped", False):
            return tool
        if not isinstance(tool, FunctionTool):
            return tool

        tool_name = str(tool.get_function_name() or "").strip()
        if not tool_name:
            return tool

        original_func = tool.func
        original_schema = copy.deepcopy(tool.get_openai_tool_schema())
        agent = self

        def guarded_tool(*args: Any, **kwargs: Any) -> Any:
            _record_tool_call_guard(agent, tool_name)
            return original_func(*args, **kwargs)

        guarded_tool.__name__ = getattr(original_func, "__name__", tool_name)
        guarded_tool.__qualname__ = getattr(
            original_func,
            "__qualname__",
            guarded_tool.__name__,
        )
        guarded_tool.__doc__ = getattr(original_func, "__doc__", None)

        wrapped_tool = FunctionTool(
            guarded_tool,
            openai_tool_schema=original_schema,
            synthesize_output=getattr(tool, "synthesize_output", False),
            synthesize_output_model=getattr(tool, "synthesize_output_model", None),
            synthesize_output_format=getattr(tool, "synthesize_output_format", None),
        )
        setattr(wrapped_tool, "_simlaw_guard_wrapped", True)
        setattr(wrapped_tool, "_simlaw_original_tool", tool)
        return wrapped_tool

    # ── LLM Interaction ──

    def step(
        self,
        instruction: str,
        response_format: Optional[type[BaseModel]] = None,
        image_list: Optional[list[str]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Execute one step of agent interaction. Requires active mode."""
        if not self._is_active or not self.chat_agent:
            raise RuntimeError(f"Agent '{self.name}' is not active. Call activate() first.")

        logger.debug(f"Agent '{self.name}' processing: {instruction[:100]}...")
        user_message = BaseMessage.make_user_message(role_name="user", content=instruction)
        started_at = time.perf_counter()
        trace_recorder = getattr(self, "_simlaw_trace_recorder", None)
        self._simlaw_step_guard_counter += 1
        self._simlaw_step_guard_state = {
            "step_id": self._simlaw_step_guard_counter,
            "started_at": time.time(),
            "tool_total_counts": {},
            "tool_query_counts": {},
        }
        try:
            if response_format:
                response = self.chat_agent.step(user_message, response_format=response_format)
            else:
                response = self.chat_agent.step(user_message)
        except Exception as exc:
            duration_seconds = time.perf_counter() - started_at
            if trace_recorder is not None and hasattr(trace_recorder, "log_agent_step"):
                try:
                    trace_recorder.log_agent_step(
                        agent=self,
                        instruction=instruction,
                        response_text="",
                        info={},
                        duration_seconds=duration_seconds,
                        error=repr(exc),
                    )
                except Exception as trace_exc:
                    logger.warning("Failed to record failed agent trace for %s: %s", self.name, trace_exc)
            raise

        self._last_step_info = dict(getattr(response, "info", {}) or {})
        self._last_tool_call_records = list(self._last_step_info.get("tool_calls") or [])
        response_content = response.msgs[0].content
        duration_seconds = time.perf_counter() - started_at
        usage = dict(self._last_step_info.get("usage") or {})
        self._simlaw_last_step_response_text = response_content
        self._simlaw_last_step_duration_seconds = duration_seconds
        self._simlaw_last_step_total_tokens = int(usage.get("total_tokens") or 0)
        if trace_recorder is not None and hasattr(trace_recorder, "log_agent_step"):
            try:
                trace_recorder.log_agent_step(
                    agent=self,
                    instruction=instruction,
                    response_text=response_content,
                    info=self._last_step_info,
                    duration_seconds=duration_seconds,
                )
            except Exception as exc:
                logger.warning("Failed to record agent trace for %s: %s", self.name, exc)
        logger.debug(
            "Agent '%s' responded in %.2fs: %s...",
            self.name,
            duration_seconds,
            response_content[:100],
        )
        self._emit_runtime_tech_usage()
        return response_content

    def build_runtime_tech_usage_payload(self, *, case_id: str = "") -> Dict[str, Any]:
        """Build a frontend runtime-progress payload from the last completed step."""
        tool_names = _dedupe_strings(
            _extract_tool_call_name(record)
            for record in list(self._last_tool_call_records or self._last_step_info.get("tool_calls") or [])
        )
        skill_names = _extract_skill_names_from_tool_calls(
            list(self._last_tool_call_records or self._last_step_info.get("tool_calls") or [])
        )
        if "load_skill" in tool_names:
            skill_names = _dedupe_strings(
                [*skill_names, *_extract_skill_names_from_usage_log(getattr(self, "skill_usage_log", []))]
            )
        if not tool_names and not skill_names:
            return {}

        stage_code = (
            str(getattr(self, "_simlaw_trace_stage_code", "") or "").strip().upper()
            or str(getattr(self, "_simlaw_stage_code", "") or "").strip().upper()
            or str(getattr(self, "scenario_type", "") or "").strip().upper()
        )
        payload: Dict[str, Any] = {
            "phase": RUNTIME_TECH_PROGRESS_PHASE,
            "message": "工具/技能已调用",
            "detail": str(getattr(self, "name", "") or getattr(self, "agent_id", "") or "").strip(),
            "blocking": False,
            "agent_id": str(getattr(self, "agent_id", "") or ""),
            "agent_name": str(getattr(self, "name", "") or ""),
            "stage": stage_code,
            "scenario_type": stage_code,
            "case_id": str(case_id or getattr(self, "_simlaw_runtime_case_id", "") or "").strip(),
            "tool_names": tool_names,
            "active_tool_names": tool_names,
            "skill_names": skill_names,
            "active_skill_names": skill_names,
        }
        return payload

    def set_runtime_tech_callback(self, callback: Any, *, case_id: str = "") -> None:
        self._simlaw_runtime_tech_callback = callback
        if case_id:
            self._simlaw_runtime_case_id = str(case_id or "").strip()

    def _emit_runtime_tech_usage(self) -> None:
        callback = getattr(self, "_simlaw_runtime_tech_callback", None)
        if not callable(callback):
            return
        payload = self.build_runtime_tech_usage_payload()
        if not payload:
            return
        try:
            result = callback(payload)
            if inspect.isawaitable(result):
                logger.warning("[%s] Runtime tech callback returned awaitable in sync step", self.name)
        except Exception as exc:
            logger.warning("[%s] Failed to emit runtime tech usage: %s", self.name, exc)

    @staticmethod
    def _extract_json_object_from_text(text: Any) -> Dict[str, Any]:
        if isinstance(text, dict):
            return text

        raw = str(text or "").strip()
        if not raw:
            raise ValueError("Empty response while extracting JSON object.")

        code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()
        else:
            object_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if object_match:
                raw = object_match.group(0).strip()

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Extracted JSON payload must be an object.")
        return payload

    def request_long_term_memory_update(
        self,
        *,
        memory_owner: str,
        existing_memory: Optional[Dict[str, Any]] = None,
        scenario_type: str = "",
        party_role: str = "",
        max_messages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deprecated: long-term memory flow now lives in role-specific agents."""
        del memory_owner, existing_memory, scenario_type, party_role, max_messages
        raise NotImplementedError(
            "Long-term memory update now lives in ClientAgent/LawyerAgent."
        )

    def generate_long_term_memory_via_tool(
        self,
        *,
        memory_owner: str,
        existing_memory: Optional[Dict[str, Any]] = None,
        scenario_type: str = "",
        party_role: str = "",
        max_messages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deprecated: long-term memory flow now lives in role-specific agents."""
        del memory_owner, existing_memory, scenario_type, party_role, max_messages
        raise NotImplementedError(
            "Long-term memory update now lives in ClientAgent/LawyerAgent."
        )

    def reset_memory(self) -> None:
        """Reset the agent's conversation memory."""
        if self.chat_agent:
            self.chat_agent.reset()
            logger.info(f"Agent '{self.name}' memory reset")

    def get_prompt_info(self) -> Dict[str, Any]:
        """Get agent's prompt information for debugging and export."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "agent_class": self.__class__.__name__,
            "system_prompt": self.system_prompt,
        }

    # ── Work Memory ──

    def _load_work_memory(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.work_memory = json.load(f)
            logger.info(f"Agent '{self.name}' loaded work memory from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Work memory file not found: {filepath}")
            self.work_memory = {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse work memory JSON: {e}")
            self.work_memory = {}

    def save_work_memory(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.work_memory, f, ensure_ascii=False, indent=2)
        logger.info(f"Agent '{self.name}' saved work memory to {filepath}")

    # ── History Export / Load ──

    def export_history(self, filepath: str, append: bool = False, include_current_system: bool = True) -> None:
        """Export conversation history to JSONL file."""
        if not self.chat_agent:
            logger.warning(f"Agent '{self.name}' has no chat_agent, skip export")
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        context_records = self.chat_agent.memory.retrieve()
        to_save = []

        for cr in context_records:
            record_dict = cr.memory_record.to_dict()
            record_dict['agent_name'] = self.name
            to_save.append(record_dict)

        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for record in to_save:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')

        logger.info(f"Agent '{self.name}' history exported to {filepath} (append={append})")

    def load_history(self, filepath: str) -> int:
        """Load conversation history from JSONL file. Filters by agent_name."""
        if not self.chat_agent:
            logger.warning(f"Agent '{self.name}' has no chat_agent, skip load")
            return 0
        if not Path(filepath).exists():
            logger.warning(f"History file not found: {filepath}")
            return 0

        try:
            from camel.types import OpenAIBackendRole, RoleType

            loaded_count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_dict = json.loads(line)
                        if record_dict.get('agent_name') != self.name:
                            continue
                        if record_dict.get('role_at_backend') == 'system':
                            continue

                        msg_dict = record_dict.get('message', {})
                        role_type_str = msg_dict.get('role_type', 'user')
                        role_type = RoleType(role_type_str) if isinstance(role_type_str, str) else role_type_str

                        message = BaseMessage(
                            role_name=msg_dict.get('role_name', 'unknown'),
                            role_type=role_type,
                            meta_dict=msg_dict.get('meta_dict'),
                            content=msg_dict.get('content', ''),
                        )
                        backend_role = (
                            OpenAIBackendRole.USER
                            if record_dict.get('role_at_backend') == 'user'
                            else OpenAIBackendRole.ASSISTANT
                        )
                        self.chat_agent.update_memory(message, backend_role)
                        loaded_count += 1
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Failed to load record: {e}")
                        continue

            logger.info(f"Agent '{self.name}' loaded {loaded_count} messages from {filepath}")
            return loaded_count
        except Exception as e:
            logger.error(f"Failed to load history from {filepath}: {e}")
            raise

    # ── Sandbox Memory Management ──

    def update_sandbox_memory(self, stage_name: str, summary: str) -> None:
        """Legacy no-op kept for compatibility after chat-summary removal."""
        del stage_name, summary
        return None

    def clear_case_memory(self) -> None:
        """Clear only in-memory case cache; on-disk memory.yaml remains the source of truth."""
        if hasattr(self, "client_profile"):
            self.client_profile = {
                "case_knowledge": {
                    "self_narrative": "",
                    "case_stage": "",
                },
                "demands": {
                    "core_demands": "",
                },
            }
        if hasattr(self, "legal_profile"):
            self.legal_profile = {
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
        logger.info("Agent '%s' runtime case memory cleared", self.name)
