"""Client (party) agent for legal simulation scenarios."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..pipeline.stage_tool_resolver import build_agent_default_tools
from ..tools.common import normalize_skill_dirs
from ..utils.live_card_memory import (
    CLIENT_MEMORY_OWNER,
    CLIENT_SAVE_TOOL_NAME,
    get_empty_memory_payload,
    is_mid_flow_stage,
    load_memory_for_agent,
    normalize_memory_payload,
)

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager


logger = logging.getLogger(__name__)

DEFAULT_CLIENT_INTERACTION_GUIDELINES = (
    "默认单次发言不要过长，不要一次性铺陈过多事实、问题或情绪。"
)
CLIENT_MEMORY_CHECKPOINT_MARKER = "__CLIENT_MEMORY_CHECKPOINT_REQUEST__"
CLIENT_MEMORY_NO_UPDATE = "CLIENT_MEMORY_NO_UPDATE"
CLIENT_MEMORY_SAVE_DONE = "CLIENT_MEMORY_SAVE_DONE"


def _normalize_memory_path(path: Optional[str]) -> Optional[str]:
    raw = str(path or "").strip()
    if not raw:
        return None
    if raw.lower().endswith(".json"):
        return raw[:-5] + ".yaml"
    if raw.lower().endswith(".yaml"):
        return raw
    return os.path.join(raw, "memory.yaml")


def _build_default_client_tools(
    agent: "ClientAgent",
    provided_tools: Optional[list[Any]] = None,
) -> list[Any]:
    """Build the default toolset for client agents."""
    return build_agent_default_tools("client", agent, provided_tools=provided_tools)


def _resolve_client_config_dir(config_path: Any) -> Optional[Path]:
    raw = str(config_path or "").strip()
    if not raw:
        return None

    path = Path(raw).resolve()
    if path.is_file() or path.name.lower() == "config.yaml":
        return path.parent
    return path


def _resolve_default_client_skill_dirs(*, config_path: Any = None) -> list[str]:
    backend_dir = Path(__file__).resolve().parents[2]
    public_root = backend_dir / "legal-skillhub" / "public"
    skill_dirs: list[str] = [str(public_root)]

    config_dir = _resolve_client_config_dir(config_path)
    if config_dir is not None:
        skill_dirs.append(str(config_dir / "skills" / "private"))

    return normalize_skill_dirs(skill_dirs)


class ClientAgent(BaseAgent):
    """Client (party) agent, shell mode by default and activated per scenario."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        party_type: str = "",
        representative: str = "",
        gender: str = "",
        ethnicity: str = "",
        birth_date: str = "",
        address: str = "",
        personality: str = "",
        speaking_style: str = "",
        interaction_guidelines: str = "",
        role: str = "plaintiff",
        legal_persona_profile: Optional[Dict[str, str]] = None,
        system_prompt: str = "",
        scenario_type: Optional[str] = None,
        scenario_data: Optional[Dict[str, Any]] = None,
        work_memory_path: Optional[str] = None,
        long_term_memory_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        config_path = kwargs.get("config_path")
        storage = kwargs.get("storage")
        provided_tools = kwargs.pop("tools", None)
        provided_skill_dirs = kwargs.pop("skill_dirs", None)
        enable_default_tools = bool(kwargs.pop("enable_default_tools", True))
        self.agent_id = agent_id
        self.name = name
        self.storage = storage
        self.config_path = config_path

        if config_path and storage and os.path.exists(config_path):
            config = storage.load_agent_config(config_path)
            profile_data = config.get("profile", {})
            self.party_type = (
                party_type
                or profile_data.get("party_type", "")
                or profile_data.get("type", "")
            )
            self.representative = representative or profile_data.get("representative", "")
            self.gender = gender or profile_data.get("gender", "")
            self.ethnicity = ethnicity or profile_data.get("ethnicity", "")
            self.birth_date = birth_date or profile_data.get("birth_date", "")
            self.address = address or profile_data.get("address", "")
            self.personality = personality or profile_data.get("personality", "")
            self.speaking_style = speaking_style or profile_data.get(
                "speaking_style",
                "",
            )
            self.interaction_guidelines = (
                interaction_guidelines
                or profile_data.get("interaction_guidelines", "")
                or DEFAULT_CLIENT_INTERACTION_GUIDELINES
            )
            self.legal_persona_profile = (
                legal_persona_profile
                or profile_data.get("legal_persona_profile", {})
                or {}
            )
        else:
            self.party_type = party_type
            self.representative = representative
            self.gender = gender
            self.ethnicity = ethnicity
            self.birth_date = birth_date
            self.address = address
            self.personality = personality
            self.speaking_style = speaking_style
            self.interaction_guidelines = (
                interaction_guidelines or DEFAULT_CLIENT_INTERACTION_GUIDELINES
            )
            self.legal_persona_profile = legal_persona_profile or {}

        self.role = role
        self.scenario_type = scenario_type
        self.scenario_data = scenario_data or {}
        self.long_term_memory_path = long_term_memory_path
        self.memory_yaml_path = _normalize_memory_path(long_term_memory_path)

        self.client_profile: Dict[str, Any] = get_empty_memory_payload(CLIENT_MEMORY_OWNER)

        if scenario_type and not system_prompt:
            system_prompt = self._build_pipeline_prompt()

        runtime_tools = (
            _build_default_client_tools(self, provided_tools)
            if enable_default_tools
            else list(provided_tools or [])
        )
        resolved_skill_dirs = normalize_skill_dirs(
            _resolve_default_client_skill_dirs(config_path=config_path)
            if provided_skill_dirs is None
            else provided_skill_dirs
        )

        super().__init__(
            agent_id=agent_id,
            name=name,
            system_prompt=system_prompt,
            work_memory_path=work_memory_path,
            tools=runtime_tools,
            skill_dirs=resolved_skill_dirs,
            **kwargs,
        )

    def _build_pipeline_prompt(self) -> str:
        """Build system prompt for pipeline mode using PromptAssembler."""
        from ..prompts.prompt_assembler import PromptAssembler

        try:
            memory_payload, _paths = load_memory_for_agent(self, CLIENT_MEMORY_OWNER)
            self.client_profile = memory_payload
        except Exception as exc:
            logger.warning("[%s] Failed to load client memory for prompt injection: %s", self.name, exc)
            memory_payload = normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)
        court_role = self.role if self.scenario_type in {"CI", "CIA"} else None
        scenario_body = PromptAssembler.build_scenario_prompt(
            "client",
            self.scenario_type,
            self.scenario_data,
            court_role,
        )

        return PromptAssembler.build(
            profile={
                "name": self.name,
                "party_type": self.party_type,
                "representative": self.representative,
                "gender": self.gender,
                "ethnicity": self.ethnicity,
                "birth_date": self.birth_date,
                "address": self.address,
                "personality": self.personality,
                "speaking_style": self.speaking_style,
                "interaction_guidelines": self.interaction_guidelines,
                "legal_persona_profile": self.legal_persona_profile,
            },
            long_term_memory=memory_payload,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=scenario_body,
        )

    def load_long_term_memory(self, filepath: str) -> None:
        self.memory_yaml_path = _normalize_memory_path(filepath)
        payload, _paths = load_memory_for_agent(self, CLIENT_MEMORY_OWNER)
        self.client_profile = payload

    def _memory_checkpoint_enabled(self) -> bool:
        stage_code = (
            str(getattr(self, "_simlaw_stage_code", "") or "").strip().upper()
            or str(self.scenario_type or "").strip().upper()
        )
        return is_mid_flow_stage(stage_code)

    def _require_memory_tools_loaded(self) -> None:
        existing_tool_names = {
            tool.get_function_name()
            for tool in list(getattr(self, "tools", []) or [])
            if tool is not None and hasattr(tool, "get_function_name")
        }
        required = {CLIENT_SAVE_TOOL_NAME}
        missing = sorted(required - existing_tool_names)
        if missing:
            raise RuntimeError(
                f"Agent '{self.name}' 缺少 memory tool: {missing}"
            )

    def _last_memory_tool_failure(self) -> Optional[str]:
        for record in reversed(list(self._last_tool_call_records or [])):
            if isinstance(record, dict):
                record_tool_name = str(
                    record.get("tool_name")
                    or record.get("name")
                    or record.get("tool")
                    or ""
                ).strip()
                record_result = record.get("result")
            else:
                record_tool_name = str(getattr(record, "tool_name", "") or "").strip()
                record_result = getattr(record, "result", None)

            if record_tool_name != CLIENT_SAVE_TOOL_NAME:
                continue
            if isinstance(record_result, str) and record_result.startswith("Tool execution failed:"):
                return record_result
        return None

    def _did_save_memory(self) -> bool:
        for record in reversed(list(self._last_tool_call_records or [])):
            if isinstance(record, dict):
                record_tool_name = str(
                    record.get("tool_name")
                    or record.get("name")
                    or record.get("tool")
                    or ""
                ).strip()
            else:
                record_tool_name = str(getattr(record, "tool_name", "") or "").strip()
            if record_tool_name == CLIENT_SAVE_TOOL_NAME:
                return True
        return False

    def _request_client_memory_checkpoint(self) -> Dict[str, Any]:
        self._require_active_chat_agent()
        self._require_memory_tools_loaded()
        prompt = (
            f"{CLIENT_MEMORY_CHECKPOINT_MARKER}\n"
            "本阶段已经结束。"
            "如果你形成了新的稳定认知、修正了旧判断、补充了案件进展或诉求，先调用 `load_skill` 加载 `client-memory-writing`，"
            "再基于当前系统提示词中的长期记忆块，只为需要修改的字段构造 JSON operations，并调用 `save_client_memory(operations=[...])` 写回。"
            "不要提交整份 YAML；每个 operation 只能包含 field、operation、content。"
            "operation 只允许 revise 或 expand：revise 覆盖该字段，expand 追加到该字段。"
            f"如果完成保存，只回复 `{CLIENT_MEMORY_SAVE_DONE}`。"
            f"如果没有需要更新的稳定认知，只回复 `{CLIENT_MEMORY_NO_UPDATE}`。"
        )
        response = self.step(prompt)
        failure = self._last_memory_tool_failure()
        if failure:
            raise RuntimeError(failure)
        if self._did_save_memory():
            return normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)
        if CLIENT_MEMORY_NO_UPDATE in str(response or ""):
            return normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)
        logger.info("[%s] Client memory checkpoint produced no save; keep current cache.", self.name)
        return normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)

    def extract_and_save_long_term_memory(
        self,
        filepath: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Checkpoint helper: let the active agent decide whether to update memory.yaml."""
        try:
            if filepath:
                self.memory_yaml_path = _normalize_memory_path(filepath)
            if not self._memory_checkpoint_enabled():
                return normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)
            if not self.is_active:
                return normalize_memory_payload(CLIENT_MEMORY_OWNER, self.client_profile)
            return self._request_client_memory_checkpoint()
        except Exception as exc:
            logger.error("Failed to checkpoint client memory: %s", exc)
            if raise_on_error:
                raise
            return None

    @property
    def case_state(self) -> str:
        if not self.storage or not self.config_path:
            return "空闲"
        config = self.storage.load_agent_config(self.config_path)
        return config.get("case_state", "空闲")

    @property
    def dataset_path(self) -> str:
        if not self.storage or not self.config_path:
            return ""
        config = self.storage.load_agent_config(self.config_path)
        return config.get("dataset_path", "")

    @property
    def case_id(self) -> str:
        if not self.storage or not self.config_path:
            return ""
        config = self.storage.load_agent_config(self.config_path)
        return config.get("case_id", "")

    @property
    def party_role(self) -> str:
        if not self.storage or not self.config_path:
            return self.role
        config = self.storage.load_agent_config(self.config_path)
        return config.get("party_role", "plaintiff")

    def register_sandbox_events(self) -> None:
        if not self.event_bus:
            return
        from ..core.event_bus import EventType

        self.event_bus.subscribe(EventType.CASE_ASSIGNED, self._on_case_assigned)

    async def _on_case_assigned(self, payload: dict) -> None:
        """Log assignment completion after receptionist dispatch."""
        if payload.get("client_id") != self.agent_id:
            return
        party_role = payload.get("party_role", "plaintiff")
        logger.info("[当事人 %s] 收到分单通知，角色=%s", self.name, party_role)
