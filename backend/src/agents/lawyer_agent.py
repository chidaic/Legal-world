"""Lawyer agent for legal simulation scenarios."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_agent import BaseAgent
from ..pipeline.stage_tool_resolver import build_agent_default_tools
from ..tools.common import normalize_skill_dirs
from ..utils.live_card_memory import (
    LAWYER_MEMORY_OWNER,
    LAWYER_SAVE_TOOL_NAME,
    get_empty_memory_payload,
    is_mid_flow_stage,
    load_memory_for_agent,
    normalize_memory_payload,
)

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.file_storage_manager import FileStorageManager


logger = logging.getLogger(__name__)

LAWYER_MEMORY_CHECKPOINT_MARKER = "__LAWYER_MEMORY_CHECKPOINT_REQUEST__"
LAWYER_MEMORY_NO_UPDATE = "LAWYER_MEMORY_NO_UPDATE"
LAWYER_MEMORY_SAVE_DONE = "LAWYER_MEMORY_SAVE_DONE"


def _normalize_memory_path(path: Optional[str]) -> Optional[str]:
    raw = str(path or "").strip()
    if not raw:
        return None
    if raw.lower().endswith(".json"):
        return raw[:-5] + ".yaml"
    if raw.lower().endswith(".yaml"):
        return raw
    return os.path.join(raw, "memory.yaml")


def _resolve_lawyer_config_dir(config_path: Any) -> Optional[Path]:
    raw = str(config_path or "").strip()
    if not raw:
        return None

    path = Path(raw).resolve()
    if path.is_file() or path.name.lower() == "config.yaml":
        return path.parent
    return path


def _extra_lawyer_skill_dirs_from_env() -> list[str]:
    raw = str(os.environ.get("SIMLAW_LAWYER_EXTRA_SKILL_DIRS") or "").strip()
    if not raw:
        return []

    candidates: list[str] = []
    for chunk in raw.replace("\n", os.pathsep).replace(",", os.pathsep).split(os.pathsep):
        item = chunk.strip().strip('"').strip("'")
        if item:
            candidates.append(item)
    return candidates


def _sanitize_case_cause_component(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", raw)
    cleaned = cleaned.rstrip(". ").strip()
    return cleaned


def _case_cause_from_scenario_data(scenario_data: Optional[Dict[str, Any]]) -> str:
    if not isinstance(scenario_data, dict):
        return ""
    for key in ("case_cause", "cause", "case_type"):
        value = scenario_data.get(key)
        if value:
            return _sanitize_case_cause_component(value)
    return ""


def _scope_extra_skill_dirs_by_case_cause(
    extra_dirs: list[str],
    *,
    scenario_data: Optional[Dict[str, Any]] = None,
) -> list[str]:
    case_cause_dir = _case_cause_from_scenario_data(scenario_data)
    if not case_cause_dir:
        return list(extra_dirs)

    scoped_dirs: list[str] = []
    for item in extra_dirs:
        root = Path(item).expanduser()
        case_cause_path = root / case_cause_dir
        if case_cause_path.is_dir():
            scoped_dirs.append(str(case_cause_path))
        else:
            scoped_dirs.append(item)
    return scoped_dirs


def _experiment_gitskill_disabled() -> bool:
    return str(os.environ.get("SIMLAW_CSD_GITSKILL_DISABLED") or "").strip() == "1"


def _resolve_default_lawyer_skill_dirs(
    *,
    config_path: Any = None,
    scenario_data: Optional[Dict[str, Any]] = None,
) -> list[str]:
    backend_dir = Path(__file__).resolve().parents[2]
    public_root = backend_dir / "legal-skillhub" / "public"
    skill_dirs: list[str] = [str(public_root)]

    config_dir = _resolve_lawyer_config_dir(config_path)
    if config_dir is not None:
        lowered_parts = [part.lower() for part in config_dir.parts]
        if "lawyers" in lowered_parts:
            skill_dirs.append(str(config_dir / "skills" / "private"))

    skill_dirs.extend(
        _scope_extra_skill_dirs_by_case_cause(
            _extra_lawyer_skill_dirs_from_env(),
            scenario_data=scenario_data,
        )
    )
    return normalize_skill_dirs(skill_dirs)


def _build_default_lawyer_tools(
    agent: "LawyerAgent",
    provided_tools: Optional[List[Any]] = None,
) -> List[Any]:
    """Build the default toolset for lawyer agents."""
    return build_agent_default_tools("lawyer", agent, provided_tools=provided_tools)


class LawyerAgent(BaseAgent):
    """Lawyer agent, shell mode by default and activated per scenario."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        specialty_areas: Optional[List[str]] = None,
        law_firm: str = "",
        firm_id: Optional[str] = None,
        system_prompt: str = "",
        scenario_type: Optional[str] = None,
        scenario_data: Optional[Dict[str, Any]] = None,
        court_role: Optional[str] = None,
        prompt_template_key: Optional[str] = None,
        work_memory_path: Optional[str] = None,
        long_term_memory_path: Optional[str] = None,
        enable_long_term_memory: bool = True,
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

        self.specialty_areas = specialty_areas or []
        self.law_firm = law_firm
        self.firm_id = firm_id

        if config_path and storage and os.path.exists(config_path):
            match = re.search(r"law_firms[/\\]([^/\\]+)", str(config_path))
            detected_firm_id = match.group(1) if match else None
            if not self.firm_id and detected_firm_id:
                self.firm_id = detected_firm_id
            if detected_firm_id:
                roster_path = (
                    storage.base_dir
                    / "law_firms"
                    / detected_firm_id
                    / "lawyer_roster.yaml"
                )
                if roster_path.exists():
                    roster = storage.load_yaml(roster_path)
                    self.law_firm = self.law_firm or roster.get("firm_name", "")
                    for lawyer in roster.get("lawyers", []):
                        if lawyer["id"] == agent_id:
                            self.specialty_areas = (
                                self.specialty_areas
                                or lawyer.get("specialty", [])
                            )
                            break

        self.scenario_type = scenario_type
        self.scenario_data = scenario_data or {}
        self.court_role = court_role
        self.prompt_template_key = str(prompt_template_key or "").strip() or None
        self.long_term_memory_path = long_term_memory_path
        self.memory_yaml_path = _normalize_memory_path(long_term_memory_path)
        self.enable_long_term_memory = enable_long_term_memory

        self.legal_profile: Dict[str, Any] = get_empty_memory_payload(LAWYER_MEMORY_OWNER)

        if scenario_type and not system_prompt:
            system_prompt = self._build_pipeline_prompt()

        runtime_tools = (
            _build_default_lawyer_tools(self, provided_tools)
            if enable_default_tools
            else list(provided_tools or [])
        )
        resolved_skill_dirs = normalize_skill_dirs(
            _resolve_default_lawyer_skill_dirs(config_path=config_path, scenario_data=self.scenario_data)
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

        memory_payload: Optional[Dict[str, Any]] = None
        if self.enable_long_term_memory:
            try:
                memory_payload, _paths = load_memory_for_agent(self, LAWYER_MEMORY_OWNER)
                self.legal_profile = memory_payload
            except Exception as exc:
                logger.warning("[%s] Failed to load lawyer memory for prompt injection: %s", self.name, exc)
                memory_payload = normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)
        scenario_body = PromptAssembler.build_scenario_prompt(
            "lawyer",
            self.scenario_type,
            self.scenario_data,
            self.court_role,
            template_key=self.prompt_template_key,
        )
        gitskill_guideline = (
            ""
            if _experiment_gitskill_disabled()
            else "\n6. 如当前可见共享 GitSkill，先根据案由、阶段、已知事实线索和争点假设判断是否有相关 Skill；加载明显相关、能帮助当前任务的 Skill，不要为了加载而加载。"
        )

        return PromptAssembler.build(
            profile={
                "name": self.name,
                "seniority": "从业十余年的执业律师",
                "personality": "沉稳干练，兼顾专业判断与沟通同理心",
                "speaking_style": "表达温和直接，引用法条时尽量解释其现实意义",
                "law_firm": self.law_firm,
                "specialty": self.specialty_areas,
                "interaction_guidelines": (
                    "[核心互动准则]\n"
                    "1. 在法律咨询、文书沟通等非庭审场景，可以先用 1-2 句承接当事人处境。\n"
                    "2. 在 CI/CIA 庭审场景中，应直接围绕审判长指令、案件争点和证据发言。\n"
                    "3. 不要把回答写成 1.2.3 的提纲，也不要输出 Markdown 标题、表格、列表或代码块。\n"
                    "4. 单次回复尽量控制在 200 字以内，避免一次性倾倒过多信息。\n"
                    "5. 仅输出发言文本，不要写括号动作、语气旁白或舞台提示。"
                    f"{gitskill_guideline}"
                ),
            },
            long_term_memory=memory_payload,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=scenario_body,
        )

    def load_long_term_memory(self, filepath: str) -> None:
        self.memory_yaml_path = _normalize_memory_path(filepath)
        payload, _paths = load_memory_for_agent(self, LAWYER_MEMORY_OWNER)
        self.legal_profile = payload

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
        required = {LAWYER_SAVE_TOOL_NAME}
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

            if record_tool_name != LAWYER_SAVE_TOOL_NAME:
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
            if record_tool_name == LAWYER_SAVE_TOOL_NAME:
                return True
        return False

    def _request_lawyer_memory_checkpoint(self) -> Dict[str, Any]:
        self._require_active_chat_agent()
        self._require_memory_tools_loaded()
        stage_code = (
            str(getattr(self, "_simlaw_stage_code", "") or "").strip().upper()
            or str(self.scenario_type or "").strip().upper()
        )
        drafting_memory_guard = ""
        if stage_code in {"CD", "DD", "AD", "AR"}:
            drafting_memory_guard = (
                "本阶段是文书起草阶段。写回长期记忆时，"
                "`client_brief.client_demand_list` 中的诉求、上诉请求、答辩意见，"
                "以及 `case_facts.evidence_ledger` 中的证据内容，必须严格依据刚才起草完成的文书正文整理。"
                "如果长期记忆、LC 咨询对话或客户临场说法与文书正文冲突，以本阶段文书正文为准；"
                "不得把文书未列明的诉求、答辩意见或证据补写进长期记忆。"
                "在 AD/AR 阶段，`case_facts.evidence_ledger` 只写二审新证据；如果文书没有列明二审新证据，写“无二审新证据”，不要保留或复制一审证据列表。"
                "系统会对上述两个字段执行覆盖式保存，留空也不会自动回填旧值。"
            )
        prompt = (
            f"{LAWYER_MEMORY_CHECKPOINT_MARKER}\n"
            "本阶段已经结束。"
            "如果你形成了新的稳定认知、修正了旧判断、补充了事实/证据/法律分析或客户信息，先调用 `load_skill` 加载 `lawyer-memory-writing`，"
            "再基于当前系统提示词中的长期记忆块，只为需要修改的字段构造 JSON operations，并调用 `save_lawyer_memory(operations=[...])` 写回。"
            "不要提交整份 YAML；每个 operation 只能包含 field、operation、content。"
            "operation 只允许 revise 或 expand：revise 覆盖该字段，expand 追加到该字段。"
            "`case_facts.case_summary` 如需更新，应对该字段使用 revise 写入压缩后的完整最新案情，不要用 expand 追加整段旧案情。"
            f"{drafting_memory_guard}"
            f"如果完成保存，只回复 `{LAWYER_MEMORY_SAVE_DONE}`。"
            f"如果没有需要更新的稳定认知，只回复 `{LAWYER_MEMORY_NO_UPDATE}`。"
        )
        response = self.step(prompt)
        failure = self._last_memory_tool_failure()
        if failure:
            raise RuntimeError(failure)
        if self._did_save_memory():
            return normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)
        if LAWYER_MEMORY_NO_UPDATE in str(response or ""):
            return normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)
        logger.info("[%s] Lawyer memory checkpoint produced no save; keep current cache.", self.name)
        return normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)

    def extract_and_save_long_term_memory(
        self,
        filepath: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Checkpoint helper: let the active lawyer decide whether to update memory.yaml."""
        try:
            if filepath:
                self.memory_yaml_path = _normalize_memory_path(filepath)
            if not self._memory_checkpoint_enabled():
                return normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)
            if not self.is_active:
                return normalize_memory_payload(LAWYER_MEMORY_OWNER, self.legal_profile)
            return self._request_lawyer_memory_checkpoint()
        except Exception as exc:
            logger.error("Failed to checkpoint lawyer memory: %s", exc)
            if raise_on_error:
                raise
            return None

    @property
    def current_handling_case(self) -> Optional[str]:
        if not self.storage or not self.config_path:
            return None
        config = self.storage.load_agent_config(self.config_path)
        return config.get("current_handling_case")

    @property
    def case_queue(self) -> List[str]:
        if not self.storage or not self.config_path:
            return []
        config = self.storage.load_agent_config(self.config_path)
        return config.get("case_queue", [])

    def register_sandbox_events(self) -> None:
        if not self.event_bus:
            return
        from ..core.event_bus import EventType

        self.event_bus.subscribe(EventType.CASE_ASSIGNED, self._on_case_assigned)

    async def _on_case_assigned(self, payload: dict) -> None:
        if payload.get("lawyer_id") != self.agent_id:
            return
        if payload.get("firm_id") != self.firm_id and self.firm_id:
            return

        case_id = payload.get("case_id", "")
        logger.info("[律师 %s] 收到分单: %s", self.name, case_id)

        if self.storage and self.config_path:
            self.storage.append_to_queue(self.config_path, "case_queue", case_id)
            logger.debug("[律师 %s] 案件 %s 已加入队列", self.name, case_id)

    async def start_handling_case(self, case_id: str) -> None:
        """Begin handling one case under orchestrator control."""
        if self.storage and self.config_path:
            self.storage.update_agent_field(
                self.config_path,
                "current_handling_case",
                case_id,
            )
            logger.info("[律师 %s] 开始办理案件 %s", self.name, case_id)

    async def on_case_closed(self, case_id: str) -> None:
        logger.info("[律师 %s] 案件 %s 结案", self.name, case_id)
        if self.storage and self.config_path:
            self.storage.update_agent_field(
                self.config_path,
                "current_handling_case",
                None,
            )
        self.clear_case_memory()
        if self._is_active:
            self.deactivate()
