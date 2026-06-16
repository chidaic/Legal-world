"""场景编排器 (ScenarioOrchestrator)。

监听 FSM 触发的场景进入事件，从 Registry 查找参与 Agent，
加载案件数据，构建 Prompt，激活 Agent，执行场景，保存输出。

替代 sandbox_main.py 中硬编码的闭包编排逻辑。
"""

import asyncio
import concurrent.futures
import json
import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..prompts.prompt_assembler import PromptAssembler
from ..data.data_loader import DataLoader
from ..player_lawyer.responsibility_marker import build_player_responsibility_marker
from ..pipeline.stage_tool_resolver import apply_stage_tool_permissions
from ..runtime_tech_strategy import RuntimeTechStrategy
from ..utils.drafted_document_sections import (
    extract_appeal_prompt_fields,
    extract_complaint_prompt_fields,
    resolve_stage_document_text,
)
from ..utils.appeal_witnesses import extract_second_instance_witness_entries
from ..utils.prompt_profile import resolve_prompt_profile_max_turns
from ..utils.runtime_flags import (
    player_lawyer_ai_surrogate_enabled,
    player_lawyer_mode_for_frontend,
    scenario_verbose_enabled,
)
from ..utils.live_card_memory import (
    CLIENT_LOAD_TOOL_NAME,
    CLIENT_MEMORY_OWNER,
    CLIENT_SAVE_TOOL_NAME,
    LAWYER_LOAD_TOOL_NAME,
    LAWYER_MEMORY_OWNER,
    LAWYER_SAVE_TOOL_NAME,
    flatten_memory_payload,
    get_empty_memory_payload,
    has_meaningful_memory,
    load_memory_for_agent,
)
from ..utils.agent_trace import CaseAgentTraceRecorder, bind_agent_trace_context
from ..core.event_bus import EventType

if TYPE_CHECKING:
    from .agent_registry import AgentRegistry
    from ..core.event_bus import EventBus
    from .case_fsm import CaseStateMachine
    from ..core.file_storage_manager import FileStorageManager
    from ..simulation.map_engine import TownAvatarInterface

logger = logging.getLogger(__name__)
SCENARIO_VERBOSE = scenario_verbose_enabled()

DEFAULT_CLIENT_INTERACTION_GUIDELINES = (
    "请像真实当事人一样自然说话，默认单次发言不要过长，尽量控制在一段内说清当前最相关的内容；"
    "通常用2到4句口语化短句表达即可，不要一次性铺陈太多事实、问题或情绪，其余内容留到律师追问后再继续补充。"
)

CHARACTER_POOL = [
    "Adam",
    "Alex",
    "Amelia",
    "Ash",
    "Bob",
    "Bruce",
    "Conference_man",
    "Conference_woman",
    "Dan",
    "Edward",
    "Lucy",
    "Molly",
    "Pier",
    "Rob",
    "Roki",
    "Samuel",
]


class ScenarioOrchestrator:
    """Bridges FSM state transitions to scenario execution."""

    STAGE_DISPLAY_NAMES = {
        "PLC": "原告咨询",
        "LC": "法律咨询",
        "DLC": "被告咨询",
        "CD": "起诉状起草",
        "DD": "答辩状起草",
        "CI": "一审庭审",
        "AD": "上诉状起草",
        "AR": "上诉答辩状起草",
        "CIA": "二审庭审",
        "FINAL_VERDICT": "终审判决",
    }

    def __init__(
        self,
        registry: "AgentRegistry",
        event_bus: "EventBus",
        fsm: "CaseStateMachine",
        storage: "FileStorageManager",
        output_dir: Path,
        map_engine: "TownAvatarInterface" = None,
    ):
        self.registry = registry
        self.event_bus = event_bus
        self.fsm = fsm
        self.storage = storage
        self.output_dir = output_dir
        self.map_engine = map_engine

        # Import Path for type checking
        from pathlib import Path as PathType
        self.Path = PathType

    def __init__(
        self,
        registry: "AgentRegistry",
        event_bus: "EventBus",
        fsm: "CaseStateMachine",
        storage: "FileStorageManager",
        sandbox_data_dir: Path,
        map_engine: "TownAvatarInterface | None" = None,
        checkpoint_manager: "Any | None" = None,
    ):
        self.registry = registry
        self.event_bus = event_bus
        self.fsm = fsm
        self.storage = storage
        self.map_engine = map_engine
        self.checkpoint_manager = checkpoint_manager
        self.sandbox_data_dir = Path(sandbox_data_dir)
        self.output_dir = self.sandbox_data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 位置占用追踪
        self._occupied_locations: dict[str, str] = {}  # {loc_id: agent_id}

        # 等候队列 (每个律所一个队列)
        self._waiting_queues: dict[str, list[dict]] = {}  # {firm_id: [{client_id, case_id, sofa_id}]}
        self._resource_lock = asyncio.Lock()
        self._trial_queues: dict[str, deque[dict[str, Any]]] = {
            "courtA": deque(),
            "courtB": deque(),
        }
        self._court_reservations: dict[str, str] = {}
        self._judge_reservations: dict[str, str] = {}
        self.runtime_issue_reporter: Callable[..., Any] | None = None
        self._case_trace_recorders: dict[str, CaseAgentTraceRecorder] = {}
        self._runtime_tech_loop: asyncio.AbstractEventLoop | None = None
        self._player_document_followup_sessions: dict[str, dict[str, Any]] = {}

        self._register_hooks()

    async def _report_runtime_issue(
        self,
        *,
        case_id: str,
        scenario_type: str,
        exc: Exception,
        stage_label: str = "",
    ) -> bool:
        reporter = getattr(self, "runtime_issue_reporter", None)
        if not callable(reporter):
            return False
        try:
            return bool(
                await reporter(
                    case_id=case_id,
                    scenario_type=scenario_type,
                    exc=exc,
                    stage_label=stage_label,
                )
            )
        except Exception as report_exc:
            logger.warning(
                "[Orchestrator] failed to report runtime issue for %s/%s: %s",
                case_id,
                scenario_type,
                report_exc,
            )
            return False

    def _register_hooks(self) -> None:
        """Subscribe to scenario-entry events."""
        from ..core.event_bus import EventType
        self.event_bus.subscribe(EventType.ENTER_PLAINTIFF_CONSULTATION, self._run_consultation)
        self.event_bus.subscribe(EventType.ENTER_DEFENDANT_CONSULTATION, self._run_consultation)
        self.event_bus.subscribe(EventType.PLAINTIFF_CONSULTATION_COMPLETED, self._auto_close_case)
        self.event_bus.subscribe(EventType.DEFENDANT_CONSULTATION_COMPLETED, self._auto_close_case)
        self.event_bus.subscribe(EventType.CASE_ASSIGNED, self._choreograph_case_assigned)
        self.event_bus.subscribe(EventType.CLIENT_CALLED, self._choreograph_client_called)
        self.event_bus.subscribe(EventType.ENTER_COMPLAINT_DRAFTING, self._run_complaint_drafting)
        self.event_bus.subscribe(EventType.COMPLAINT_DRAFTING_COMPLETED, self._on_complaint_filed)
        self.event_bus.subscribe(EventType.LAWSUIT_FILED, self._activate_defendant)
        self.event_bus.subscribe(EventType.DEFENDANT_ARRIVED, self._on_defendant_arrived)
        self.event_bus.subscribe(EventType.ENTER_DEFENSE_DRAFTING, self._run_defense_drafting)
        self.event_bus.subscribe(EventType.DEFENSE_DRAFTING_COMPLETED, self._on_defense_filed)
        self.event_bus.subscribe(EventType.DEFENSE_FILED, self._check_trial_ready)
        self.event_bus.subscribe(EventType.ENTER_TRIAL_FIRST_INSTANCE, self._choreograph_first_trial)
        self.event_bus.subscribe(EventType.TRIAL_FIRST_INSTANCE_READY, self._run_first_instance_trial)
        self.event_bus.subscribe(EventType.TRIAL_FIRST_INSTANCE_COMPLETED, self._on_first_instance_verdict)
        self.event_bus.subscribe(EventType.FIRST_INSTANCE_VERDICT_ISSUED, self._start_appeal_decision)
        self.event_bus.subscribe(EventType.APPEAL_DECISION_MADE, self._handle_appeal_decision)
        self.event_bus.subscribe(EventType.APPEAL_DRAFTING_COMPLETED, self._on_appeal_filed)
        self.event_bus.subscribe(EventType.APPEAL_FILED, self._activate_appeal_response)
        self.event_bus.subscribe(EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED, self._on_appeal_response_filed)
        self.event_bus.subscribe(EventType.APPEAL_RESPONSE_FILED, self._check_appeal_trial_ready)
        self.event_bus.subscribe(EventType.ENTER_TRIAL_SECOND_INSTANCE, self._choreograph_second_trial)
        self.event_bus.subscribe(EventType.TRIAL_SECOND_INSTANCE_READY, self._run_second_instance_trial)
        self.event_bus.subscribe(EventType.TRIAL_SECOND_INSTANCE_COMPLETED, self._on_final_verdict)
        self.event_bus.subscribe(EventType.CASE_CLOSED, self._choreograph_case_closed)

    @staticmethod
    def _configure_stage_tools(stage_code: str, role_to_agent: dict[str, Any]) -> dict[str, list[str]]:
        """Apply manifest-declared tool permissions for active scenario participants."""
        return apply_stage_tool_permissions(stage_code, role_to_agent)

    # ── Helper: load case data from client config ──

    def _load_case_data(self, client_config_path: str) -> tuple:
        """Load DataLoader and case dict from a client's config.yaml.

        Returns:
            (data_loader, case_dict, client_config)
        """
        config = self.storage.load_agent_config(client_config_path)
        dataset_path = config.get("dataset_path", "")

        data_loader = DataLoader(dataset_path)
        case = data_loader.resolve_case_for_config(
            config,
            fallback_dataset_paths=self._build_dataset_fallback_candidates(dataset_path),
        )
        return data_loader, case, config

    @staticmethod
    def _build_dataset_fallback_candidates(dataset_path: str) -> list[str]:
        current_path = str(dataset_path or "").strip()
        project_root = Path(__file__).resolve().parents[3]
        data_root = project_root / "data"
        if not data_root.exists():
            return []

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(candidate: Path) -> None:
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

        if current_path:
            candidate_name = Path(current_path.replace("\\", "/")).name
            if candidate_name:
                same_name_candidate = data_root / candidate_name
                if same_name_candidate.exists():
                    _add(same_name_candidate)

        for item in sorted(data_root.glob("*.json")):
            if item.is_file():
                _add(item)

        return candidates

    @staticmethod
    def _build_client_prompt_profile(agent: Any, extracted_profile: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = extracted_profile or {}
        return {
            "name": str(getattr(agent, "name", "") or profile.get("name", "") or "").strip(),
            "party_type": str(
                getattr(agent, "party_type", "")
                or profile.get("party_type", "")
                or profile.get("type", "")
                or ""
            ).strip(),
            "representative": str(
                getattr(agent, "representative", "")
                or profile.get("representative", "")
                or ""
            ).strip(),
            "gender": str(getattr(agent, "gender", "") or profile.get("gender", "") or "").strip(),
            "birth_date": str(getattr(agent, "birth_date", "") or profile.get("birth_date", "") or "").strip(),
            "ethnicity": str(getattr(agent, "ethnicity", "") or profile.get("ethnicity", "") or "").strip(),
            "address": str(getattr(agent, "address", "") or profile.get("address", "") or "").strip(),
            "personality": str(
                getattr(agent, "personality", "") or profile.get("personality", "") or ""
            ).strip(),
            "speaking_style": str(
                getattr(agent, "speaking_style", "") or profile.get("speaking_style", "") or ""
            ).strip(),
            "interaction_guidelines": str(
                getattr(agent, "interaction_guidelines", "")
                or profile.get("interaction_guidelines", "")
                or DEFAULT_CLIENT_INTERACTION_GUIDELINES
            ).strip(),
            "legal_persona_profile": (
                getattr(agent, "legal_persona_profile", None)
                or profile.get("legal_persona_profile", {})
                or {}
            ),
        }

    @staticmethod
    def _build_lawyer_profile(lawyer: Any) -> dict[str, Any]:
        """Build the full lawyer profile dict for PromptAssembler.

        Mirrors the profile used in LawyerAgent._build_pipeline_prompt so that
        sandbox-mode scenarios and pipeline-mode scenarios produce identical
        system prompts (including interaction_guidelines).
        """
        return {
            "name": getattr(lawyer, "name", ""),
            "seniority": "从业十余年的执业律师",
            "personality": "沉稳干练，具备极强的同理心；不仅提供专业建议，更是客户的情绪稳定剂",
            "speaking_style": "坚定温和，口语化表达；引用法条时用白话解释实际影响",
            "law_firm": getattr(lawyer, "law_firm", ""),
            "specialty": getattr(lawyer, "specialty_areas", []),
            "interaction_guidelines": (
                "[核心交互准则]\n"
                "1. 情绪安抚：在法律咨询、文书沟通等非庭审场景，回答核心诉求前可先用1-2句话安抚或认可对方情境；但在庭审场景（CI/CIA）中，不得以安抚当事人情绪作为开头，应直接围绕审判长指令、案件争点和证据发言。\n"
                "2. 拒绝机械宣讲：绝对不要分点1.2.3.4回答、不要列小标题、不要像提纲或模板答案；禁止使用 Markdown 标题、加粗星号样式、星号列表、表格、代码块。\n"
                "3. 信息切块：单次回复控制在200字内，不要一次性输出太多信息。\n"
                "4. 纯文本表达：不要输出括号中的动作、表情、语气描写，如“（起立）”“（沉默）”“（声音越来越小）”。"
            ),
        }

    @staticmethod
    def _normalize_case_id(case_id: str) -> str:
        case_key = str(case_id or "")
        if case_key.startswith("case_"):
            return case_key[5:]
        return case_key

    def _get_case_output_dir(self, case_id: str) -> Path:
        case_output_dir = self.output_dir / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        return case_output_dir

    def _get_case_trace_recorder(self, case_id: str) -> CaseAgentTraceRecorder:
        recorder = self._case_trace_recorders.get(case_id)
        case_output_dir = self._get_case_output_dir(case_id)
        if recorder is None or recorder.case_output_dir != case_output_dir.resolve():
            recorder = CaseAgentTraceRecorder(case_output_dir)
            self._case_trace_recorders[case_id] = recorder
        return recorder

    def _bind_case_stage_trace_agents(
        self,
        case_id: str,
        stage_code: str,
        stage_key: str,
        agents: list[Any],
    ) -> CaseAgentTraceRecorder:
        recorder = self._get_case_trace_recorder(case_id)
        case_output_dir = self._get_case_output_dir(case_id)
        callback = self._build_runtime_tech_callback(case_id)
        for agent in list(agents or []):
            if agent is None:
                continue
            agent_id = str(getattr(agent, "agent_id", "") or "agent").strip() or "agent"
            bind_agent_trace_context(
                agent,
                recorder=recorder,
                output_dir=case_output_dir / "_debug" / "agent_traces" / agent_id,
                stage_code=stage_code,
                stage_key=stage_key,
            )
            if hasattr(agent, "set_runtime_tech_callback"):
                agent.set_runtime_tech_callback(callback, case_id=case_id)
        return recorder

    def _build_runtime_tech_callback(self, case_id: str):
        try:
            self._runtime_tech_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._runtime_tech_loop = None

        def _callback(payload: dict[str, Any]) -> None:
            if not payload or not self.map_engine or not hasattr(self.map_engine, "broadcast_runtime_progress"):
                return
            effective_case_id = str(payload.get("case_id") or case_id or "").strip()
            if not effective_case_id:
                return
            metadata = {
                key: value
                for key, value in dict(payload).items()
                if key not in {"case_id", "phase", "message", "detail", "blocking"}
            }

            async def _broadcast() -> None:
                await self.map_engine.broadcast_runtime_progress(
                    effective_case_id,
                    phase=str(payload.get("phase") or "runtime_tech_used"),
                    message=str(payload.get("message") or "工具/技能已调用"),
                    detail=str(payload.get("detail") or ""),
                    blocking=bool(payload.get("blocking", False)),
                    metadata=metadata,
                )

            loop = self._runtime_tech_loop
            if loop is not None and loop.is_running():
                try:
                    future = asyncio.run_coroutine_threadsafe(_broadcast(), loop)
                    future.add_done_callback(self._log_runtime_tech_broadcast_result)
                    return
                except RuntimeError:
                    logger.warning("[Orchestrator] Runtime tech event loop unavailable")

        return _callback

    @staticmethod
    def _log_runtime_tech_broadcast_result(future: concurrent.futures.Future) -> None:
        try:
            future.result()
        except Exception as exc:
            logger.warning("[Orchestrator] Runtime tech broadcast failed: %s", exc)

    def _runtime_tech_strategy(self, trace_recorder: Any | None = None) -> RuntimeTechStrategy:
        return RuntimeTechStrategy(
            map_engine=self.map_engine,
            trace_recorder=trace_recorder,
        )

    async def _emit_runtime_stage_start(
        self,
        *,
        case_id: str,
        stage_code: str,
        trace_recorder: Any | None = None,
    ) -> None:
        await self._runtime_tech_strategy(trace_recorder).emit_stage_start(
            case_id=case_id,
            stage_code=stage_code,
        )

    async def _emit_runtime_stage_research(
        self,
        *,
        case_id: str,
        stage_code: str,
        case_cause: str = "",
        case_background: str = "",
        trace_recorder: Any | None = None,
    ) -> None:
        await self._runtime_tech_strategy(trace_recorder).emit_stage_research(
            case_id=case_id,
            stage_code=stage_code,
            case_cause=case_cause,
            case_background=case_background,
        )

    async def _emit_runtime_document_complete(
        self,
        *,
        case_id: str,
        stage_code: str,
        document_text: str = "",
        compare_left: str = "",
        compare_right: str = "",
        compare_labels: tuple[str, str] = ("document_a", "document_b"),
        trace_recorder: Any | None = None,
    ) -> None:
        await self._runtime_tech_strategy(trace_recorder).emit_document_complete(
            case_id=case_id,
            stage_code=stage_code,
            document_text=document_text,
            compare_left=compare_left,
            compare_right=compare_right,
            compare_labels=compare_labels,
        )

    @staticmethod
    def _resolve_stage_max_turns(stage_code: str, prod_default: int) -> int:
        return resolve_prompt_profile_max_turns(stage_code, prod_default)

    @staticmethod
    def _resolve_lc_max_turns(question_count: int, *, player_lawyer_enabled: bool = False) -> int:
        from ..scenarios.legal_consultation import LegalConsultationScenario

        base_max_turns = ScenarioOrchestrator._resolve_stage_max_turns(
            "LC",
            LegalConsultationScenario.DEFAULT_MAX_TURNS,
        )
        if not player_lawyer_enabled or question_count <= 0:
            return base_max_turns
        return max(1, min(base_max_turns, question_count, 2))

    @staticmethod
    def _consultation_display_stage_code(party_role: str) -> str:
        return "DLC" if str(party_role or "").strip().lower() == "defendant" else "PLC"

    def _get_case_role_bundle(self, case_id: str, party_role: str) -> dict[str, Any]:
        client, client_path = self._find_client_for_case(case_id, party_role=party_role)
        config = self.storage.load_agent_config(client_path) if client_path else {}
        lawyer_id = config.get("assigned_lawyer_id", "")
        lawyer = self.registry.get_agent(lawyer_id) if lawyer_id else None
        return {
            "client": client,
            "client_path": client_path,
            "config": config,
            "lawyer_id": lawyer_id,
            "lawyer": lawyer,
        }

    def _get_case_parties(self, case_id: str) -> dict[str, dict[str, Any]]:
        return {
            "plaintiff": self._get_case_role_bundle(case_id, "plaintiff"),
            "defendant": self._get_case_role_bundle(case_id, "defendant"),
        }

    def _set_shared_case_state(self, case_id: str, state: str) -> None:
        runtime: dict[str, Any] = {}
        try:
            runtime = self.storage.load_case_runtime(case_id)
        except Exception:
            runtime = {}

        for party_role in ("plaintiff", "defendant"):
            agent_path = self.storage.get_case_agent_path(case_id, party_role)
            if (agent_path / "config.yaml").exists():
                try:
                    self.storage.update_agent_field(agent_path, "case_state", state)
                except Exception as exc:
                    logger.warning(
                        "[Orchestrator] 更新%s案件状态失败: case=%s state=%s error=%s",
                        party_role,
                        case_id,
                        state,
                        exc,
                    )

        runtime.update(
            {
                "case_id": self._normalize_case_id(case_id),
                "overall_state": state,
                "plaintiff_state": state,
                "defendant_state": state,
                "active_party_role": "shared",
            }
        )
        try:
            self.storage.save_case_runtime(case_id, runtime)
        except Exception as exc:
            logger.warning("[Orchestrator] 保存共享案件运行态失败: case=%s state=%s error=%s", case_id, state, exc)

    def _select_silent_opponent_lawyer(
        self,
        case_id: str,
        plaintiff_lawyer_id: str,
        preferred_firm: str = "",
    ) -> Any | None:
        lawyers = self.registry.get_agents_by_type("lawyer") if self.registry else []
        candidates = [
            lawyer
            for lawyer in lawyers
            if str(getattr(lawyer, "agent_id", "") or "").strip()
            and str(getattr(lawyer, "agent_id", "") or "").strip() != plaintiff_lawyer_id
        ]
        if not candidates:
            logger.error("[Orchestrator] 无可用对手律师: case=%s plaintiff_lawyer=%s", case_id, plaintiff_lawyer_id)
            return None
        normalized_preferred = self._normalize_firm_id(preferred_firm)
        candidates.sort(
            key=lambda lawyer: (
                0 if normalized_preferred and self._normalize_firm_id(str(getattr(lawyer, "firm_id", "") or "")) == normalized_preferred else 1,
                0 if str(getattr(lawyer, "firm_id", "") or "").lower().endswith("b") else 1,
                str(getattr(lawyer, "agent_id", "") or ""),
            )
        )
        return candidates[0]

    def _ensure_player_trial_opponent_bundle(self, payload: dict) -> dict[str, Any]:
        """Prepare the opponent for trial only; no opponent consultation/document events are emitted."""
        from ..agents.client_agent import ClientAgent

        case_id = payload.get("case_id", "")
        plaintiff_path = payload.get("client_path", "")
        plaintiff_config = self.storage.load_agent_config(plaintiff_path) if plaintiff_path else {}
        plaintiff_lawyer_id = str(
            payload.get("plaintiff_lawyer_id")
            or payload.get("lawyer_id")
            or plaintiff_config.get("assigned_lawyer_id", "")
            or ""
        ).strip()

        data_loader, case, _ = self._load_case_data(plaintiff_path)
        party_info = case.get("extracted_info", {}).get("party_info", {})
        defendant_data = party_info.get("defendant", {})
        if isinstance(defendant_data, list):
            defendant_data = defendant_data[0] if defendant_data else {}
        defendant_info = DataLoader.normalize_party_profile(defendant_data if isinstance(defendant_data, dict) else {})

        defendant, defendant_path = self._find_client_for_case(case_id, party_role="defendant")
        if not defendant_path:
            defendant_config_path = self.storage.get_case_agent_path(case_id, "defendant")
            defendant_config = {
                "case_id": case_id,
                "party_role": "defendant",
                "profile": {
                    "name": defendant_info.get("name", "被告"),
                    "type": defendant_info.get("type", "") or defendant_info.get("party_type", ""),
                    "party_type": defendant_info.get("party_type", "") or defendant_info.get("type", ""),
                    "gender": defendant_info.get("gender", ""),
                    "ethnicity": defendant_info.get("ethnicity", ""),
                    "birth_date": defendant_info.get("birth_date", ""),
                    "address": defendant_info.get("address", ""),
                    "representative": defendant_info.get("representative", ""),
                    "legal_persona_profile": defendant_info.get("legal_persona_profile", {}) or {},
                },
                "case_state": "等待一审开庭",
                "dataset_path": plaintiff_config.get("dataset_path", ""),
            }
            self.storage.save_agent_config(defendant_config_path, defendant_config)
            defendant_path = str(defendant_config_path)

        if not defendant:
            defendant_id = f"defendant_{case_id}"
            defendant = ClientAgent(
                agent_id=defendant_id,
                name=defendant_info.get("name", "被告"),
                gender=defendant_info.get("gender", ""),
                role="defendant",
                event_bus=self.event_bus,
                storage=self.storage,
                config_path=defendant_path,
            )
            self.registry._agents[defendant_id] = defendant

        defendant_config = self.storage.load_agent_config(defendant_path)
        defendant_lawyer_id = str(defendant_config.get("assigned_lawyer_id", "") or "").strip()
        if not defendant_lawyer_id or defendant_lawyer_id == plaintiff_lawyer_id:
            lawyer = self._select_silent_opponent_lawyer(
                case_id,
                plaintiff_lawyer_id,
                preferred_firm=str(defendant_config.get("assigned_firm", "") or ""),
            )
            defendant_lawyer_id = str(getattr(lawyer, "agent_id", "") or "").strip() if lawyer else ""
        lawyer = self.registry.get_agent(defendant_lawyer_id) if defendant_lawyer_id else None
        if defendant_lawyer_id:
            defendant_config["assigned_lawyer_id"] = defendant_lawyer_id
            defendant_config["current_handling_case"] = case_id
            self.storage.save_agent_config(defendant_path, defendant_config)

        return {
            "defendant": defendant,
            "defendant_id": getattr(defendant, "agent_id", ""),
            "defendant_path": defendant_path,
            "defendant_lawyer": lawyer,
            "defendant_lawyer_id": defendant_lawyer_id,
        }

    def _build_ci_opponent_lawyer_scenario_data(
        self,
        data_loader: DataLoader,
        case: dict[str, Any],
        case_output_dir: Path,
    ) -> dict[str, Any]:
        """Dataset-backed opponent prompt: 你没有经历本轮前序咨询或答辩状起草，系统直接依据真实数据集回填。"""
        return {
            "court_name": data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case),
            "case_cause": data_loader.extract_case_cause(case),
            "case_background": data_loader.extract_case_background(case),
            "claims": data_loader.extract_claims(case),
            "facts_and_reasons": data_loader.extract_facts_and_reasons(case),
            "my_position": data_loader.extract_defendant_defense(case),
            "my_evidence": data_loader.extract_defendant_evidence(case),
            "case_output_dir": str(case_output_dir.resolve()),
        }

    def _build_cia_opponent_lawyer_scenario_data(
        self,
        data_loader: DataLoader,
        case: dict[str, Any],
        *,
        court_role: str,
        first_instance_verdict: str,
        case_output_dir: Path,
    ) -> dict[str, Any]:
        """Dataset-backed opponent prompt: 你没有经历本轮前序上诉文书起草，系统直接依据真实数据集回填。"""
        appellant_info = data_loader.extract_appellant_appeal(case)
        appeal_claims = appellant_info.get("claim", [])
        appeal_requests = (
            "\n".join(f"{index + 1}. {claim}" for index, claim in enumerate(appeal_claims))
            if isinstance(appeal_claims, list)
            else str(appeal_claims or "")
        )
        return {
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "case_cause": data_loader.extract_case_cause(case),
            "case_background": data_loader.extract_case_background(case),
            "first_instance_judgment": first_instance_verdict,
            "appeal_requests": appeal_requests,
            "appeal_reasons": appellant_info.get("reasons", ""),
            "my_position": (
                data_loader.extract_second_instance_appellee_defense(case)
                if court_role == "appellee"
                else appellant_info.get("reasons", "")
            ),
            "my_new_evidence": data_loader.extract_second_instance_evidence(
                case,
                side="appellee" if court_role == "appellee" else "appellant",
            ) or "（暂无新证据）",
            "case_output_dir": str(case_output_dir.resolve()),
        }

    def _get_agent_memory_payload(self, agent: Any | None, memory_owner: str) -> dict[str, Any]:
        if agent is None:
            return get_empty_memory_payload(memory_owner)
        try:
            payload, _paths = load_memory_for_agent(agent, memory_owner)
            return payload
        except Exception as exc:
            logger.warning(
                "[Orchestrator] failed to load %s memory for %s: %s",
                memory_owner,
                getattr(agent, "agent_id", agent),
                exc,
            )
            return get_empty_memory_payload(memory_owner)

    def _get_lawyer_prompt_memory(
        self,
        lawyer: Any | None,
        case_id: str,
    ) -> dict[str, Any]:
        _ = case_id
        return self._get_agent_memory_payload(lawyer, LAWYER_MEMORY_OWNER)

    def _get_client_prompt_memory(
        self,
        client: Any | None,
        case_id: str,
    ) -> dict[str, Any]:
        _ = case_id
        return self._get_agent_memory_payload(client, CLIENT_MEMORY_OWNER)

    @staticmethod
    def _extract_config_profile(config: dict[str, Any] | None) -> dict[str, Any]:
        profile = (config or {}).get("profile", {}) or {}
        return profile if isinstance(profile, dict) else {}

    @staticmethod
    def _extract_memory_text(memory_payload: dict[str, Any] | None, key: str) -> str:
        current: Any = memory_payload if isinstance(memory_payload, dict) else {}
        for part in str(key or "").split("."):
            part = part.strip()
            if not part or not isinstance(current, dict):
                return ""
            current = current.get(part, "")
        return str(current or "").strip()

    @staticmethod
    def _has_meaningful_long_term_memory(memory_payload: Any) -> bool:
        return has_meaningful_memory(memory_payload)

    def _resolve_lawyer_case_background(
        self,
        default_background: Any,
        *,
        long_term_memory: dict[str, Any] | None,
    ) -> str:
        if self._has_meaningful_long_term_memory(long_term_memory or {}):
            return ""
        return self._stringify_prompt_value(default_background, fallback="")

    def _build_case_party_context(
        self,
        case_id: str,
        *,
        party_role: str,
        case: dict[str, Any] | None = None,
        default_case_background: Any = "",
        default_claims: Any = "",
        default_evidence: Any = "",
    ) -> dict[str, str]:
        normalized_case_id = self._normalize_case_id(case_id)
        storage = getattr(self, "storage", None)
        if storage is None:
            extracted_info = (case or {}).get("extracted_info", {}) or {}
            return {
                "plaintiff_name": "",
                "plaintiff_gender": "",
                "plaintiff_birth_date": "",
                "plaintiff_ethnicity": "",
                "plaintiff_address": "",
                "plaintiff_representative": "",
                "defendant_name": "",
                "defendant_gender": "",
                "defendant_birth_date": "",
                "defendant_ethnicity": "",
                "defendant_address": "",
                "defendant_representative": "",
                "case_background": self._stringify_prompt_value(
                    default_case_background or extracted_info.get("case_background", ""),
                    fallback="",
                ),
                "claims": self._stringify_prompt_value(default_claims, fallback=""),
                "evidence": self._stringify_prompt_value(default_evidence, fallback=""),
            }

        plaintiff_path = storage.get_case_agent_path(normalized_case_id, "plaintiff")
        defendant_path = storage.get_case_agent_path(normalized_case_id, "defendant")

        try:
            plaintiff_config = storage.load_agent_config(plaintiff_path)
        except Exception:
            plaintiff_config = {}
        try:
            defendant_config = storage.load_agent_config(defendant_path)
        except Exception:
            defendant_config = {}

        extracted_info = (case or {}).get("extracted_info", {}) or {}
        party_info = extracted_info.get("party_info", {}) or {}
        extracted_plaintiff = DataLoader.normalize_party_profile(party_info.get("plaintiff", {}) or {})
        extracted_defendant = party_info.get("defendant", {}) or {}
        if isinstance(extracted_defendant, list):
            extracted_defendant = extracted_defendant[0] if extracted_defendant else {}
        extracted_defendant = DataLoader.normalize_party_profile(extracted_defendant)

        plaintiff_profile = self._extract_config_profile(plaintiff_config)
        defendant_profile = self._extract_config_profile(defendant_config)
        plaintiff_agent, _ = self._find_client_for_case(normalized_case_id, party_role="plaintiff")
        defendant_agent, _ = self._find_client_for_case(normalized_case_id, party_role="defendant")
        plaintiff_memory = self._get_agent_memory_payload(plaintiff_agent, CLIENT_MEMORY_OWNER)
        defendant_memory = self._get_agent_memory_payload(defendant_agent, CLIENT_MEMORY_OWNER)
        current_memory = plaintiff_memory if party_role == "plaintiff" else defendant_memory

        plaintiff_name = str(
            plaintiff_profile.get("name")
            or extracted_plaintiff.get("name")
            or ""
        ).strip()
        defendant_name = str(
            defendant_profile.get("name")
            or extracted_defendant.get("name")
            or ""
        ).strip()

        case_background = (
            self._extract_memory_text(current_memory, "case_knowledge.self_narrative")
            or self._extract_memory_text(plaintiff_memory, "case_knowledge.self_narrative")
            or self._extract_memory_text(defendant_memory, "case_knowledge.self_narrative")
            or self._stringify_prompt_value(default_case_background, fallback="")
            or str(extracted_info.get("case_background", "") or "").strip()
        )
        claims = (
            self._extract_memory_text(current_memory, "demands.core_demands")
            or self._extract_memory_text(plaintiff_memory, "demands.core_demands")
            or self._extract_memory_text(defendant_memory, "demands.core_demands")
            or self._stringify_prompt_value(default_claims, fallback="")
        )
        evidence = self._stringify_prompt_value(default_evidence, fallback="")

        return {
            "plaintiff_name": plaintiff_name,
            "plaintiff_gender": str(plaintiff_profile.get("gender", "") or extracted_plaintiff.get("gender", "") or "").strip(),
            "plaintiff_birth_date": str(plaintiff_profile.get("birth_date", "") or extracted_plaintiff.get("birth_date", "") or "").strip(),
            "plaintiff_ethnicity": str(plaintiff_profile.get("ethnicity", "") or extracted_plaintiff.get("ethnicity", "") or "").strip(),
            "plaintiff_address": str(plaintiff_profile.get("address", "") or extracted_plaintiff.get("address", "") or "").strip(),
            "plaintiff_representative": str(plaintiff_profile.get("representative", "") or extracted_plaintiff.get("representative", "") or "").strip(),
            "defendant_name": defendant_name,
            "defendant_gender": str(defendant_profile.get("gender", "") or extracted_defendant.get("gender", "") or "").strip(),
            "defendant_birth_date": str(defendant_profile.get("birth_date", "") or extracted_defendant.get("birth_date", "") or "").strip(),
            "defendant_ethnicity": str(defendant_profile.get("ethnicity", "") or extracted_defendant.get("ethnicity", "") or "").strip(),
            "defendant_address": str(defendant_profile.get("address", "") or extracted_defendant.get("address", "") or "").strip(),
            "defendant_representative": str(defendant_profile.get("representative", "") or extracted_defendant.get("representative", "") or "").strip(),
            "case_background": case_background,
            "claims": claims,
            "evidence": evidence,
        }

    def _normalize_first_instance_state(
        self,
        case_id: str,
        plaintiff_path: str,
        defendant_path: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        """在一审共享阶段恢复时对齐双方状态，避免事件并发产生竞态。"""
        plaintiff_config = self.storage.load_agent_config(plaintiff_path)
        defendant_config = self.storage.load_agent_config(defendant_path)

        plaintiff_state = plaintiff_config.get("case_state", "")
        defendant_state = defendant_config.get("case_state", "")
        observed_states = {plaintiff_state, defendant_state}
        target_state = None

        if "一审庭审中" in observed_states:
            target_state = "一审庭审中"
        elif "等待一审开庭" in observed_states:
            target_state = "等待一审开庭"

        if not target_state:
            return plaintiff_config, defendant_config, None

        for path, config in (
            (plaintiff_path, plaintiff_config),
            (defendant_path, defendant_config),
        ):
            if config.get("case_state") != target_state:
                self.storage.update_agent_field(path, "case_state", target_state)
                config["case_state"] = target_state

        return plaintiff_config, defendant_config, target_state

    def _resolve_map_prefix_from_lawyer(self, lawyer: Any) -> str:
        firm_id = str(getattr(lawyer, "firm_id", "") or "").lower()
        if firm_id in {"law_firm_b", "lawfirmb"}:
            return "lawfirmB"
        return "lawfirmA"

    def _build_player_lawyer_adapter(self, lawyer: Any, *, case_id: str, stage: str) -> Any:
        from ..player_lawyer.agent import PlayerPlaintiffLawyerAgent

        gateway = getattr(self, "_player_gateway", None)
        if gateway is None:
            return lawyer
        adapter = PlayerPlaintiffLawyerAgent(
            agent_id=getattr(lawyer, "agent_id", ""),
            name=getattr(lawyer, "name", "原告律师"),
            law_firm=getattr(lawyer, "law_firm", ""),
            firm_id=getattr(lawyer, "firm_id", ""),
            gateway=gateway,
            case_id=case_id,
            sandbox_id=getattr(self, "_sandbox_id", 0),
            broadcast_fn=getattr(self, "_player_broadcast_fn", None),
        )
        adapter.config_path = getattr(lawyer, "config_path", None)
        adapter.storage = getattr(lawyer, "storage", None)
        adapter.set_stage(stage)
        return adapter

    def _player_lawyer_mode(self) -> str:
        map_engine = getattr(self, "map_engine", None)
        frontend_mode = getattr(map_engine, "_frontend_mode", None)
        supports_player_v2 = False
        supports_fn = getattr(map_engine, "supports_player_v2_runtime", None)
        if callable(supports_fn):
            supports_player_v2 = bool(supports_fn())
        return player_lawyer_mode_for_frontend(
            frontend_mode=frontend_mode,
            has_player_v2_client=supports_player_v2,
        )

    def _player_plaintiff_lawyer_enabled(self) -> bool:
        return self._player_lawyer_mode() == "plaintiff"

    def _player_ai_surrogate_enabled(self) -> bool:
        return player_lawyer_ai_surrogate_enabled()

    @staticmethod
    def _resolve_birth_location_for_map_prefix(map_prefix: str) -> str:
        return "birth_locationB" if str(map_prefix).lower().endswith("b") else "birth_locationA"

    def _get_birth_location_for_agent(self, agent_id: str) -> str:
        if self.map_engine:
            state = getattr(self.map_engine, "_agent_states", {}).get(agent_id, {})
            birth_loc_id = state.get("birth_loc_id")
            if birth_loc_id:
                return birth_loc_id

        agent = self.registry.get_agent(agent_id)
        if not agent:
            return "birth_locationA"

        agent_type = getattr(agent, "agent_type", "")
        if agent_type == "judge":
            return "birth_locationB"
        if agent_type == "lawyer":
            return self._resolve_birth_location_for_map_prefix(
                self._resolve_map_prefix_from_lawyer(agent)
            )
        if getattr(agent, "config_path", None):
            try:
                config = self.storage.load_agent_config(agent.config_path)
                if config.get("party_role") == "defendant":
                    return "birth_locationB"
            except Exception:
                pass
        return "birth_locationA"

    @staticmethod
    def _match_party_name(target_name: str, candidate_names: list[str]) -> bool:
        normalized_target = str(target_name or "").strip()
        if not normalized_target:
            return False
        return any(
            candidate and (candidate in normalized_target or normalized_target in candidate)
            for candidate in candidate_names
        )

    def _resolve_appeal_roles(self, case: dict) -> dict[str, str]:
        extracted_info = case.get("extracted_info", {})
        appellant_raw = str(extracted_info.get("appellant", "") or "").strip()
        party_info = extracted_info.get("party_info", {})
        plaintiff_profile = party_info.get("plaintiff", {}) if isinstance(party_info.get("plaintiff", {}), dict) else {}
        defendant_raw = party_info.get("defendant", {})
        if isinstance(defendant_raw, list):
            defendant_profile = defendant_raw[0] if defendant_raw else {}
        else:
            defendant_profile = defendant_raw if isinstance(defendant_raw, dict) else {}
        defendant_names = [
            str(defendant_profile.get("name", "") or "").strip(),
        ]
        if isinstance(defendant_raw, list):
            defendant_names.extend(
                [
                    str(defendant.get("name", "") or "").strip()
                    for defendant in defendant_raw
                    if isinstance(defendant, dict) and str(defendant.get("name", "") or "").strip()
                ]
            )

        appellant_role = ""
        lowered = appellant_raw.lower()
        if "原告" in appellant_raw or lowered == "plaintiff":
            appellant_role = "plaintiff"
        elif "被告" in appellant_raw or lowered == "defendant":
            appellant_role = "defendant"
        elif self._match_party_name(appellant_raw, [str(plaintiff_profile.get("name", "") or "").strip()]):
            appellant_role = "plaintiff"
        elif self._match_party_name(appellant_raw, defendant_names):
            appellant_role = "defendant"

        if not appellant_role:
            logger.warning("[Orchestrator] 无法根据数据集识别上诉方，默认按不上诉处理: appellant=%s", appellant_raw)
            return {
                "appellant_role": "",
                "appellee_role": "",
                "appellant_name": appellant_raw,
                "appellee_name": "",
            }

        appellee_role = "defendant" if appellant_role == "plaintiff" else "plaintiff"
        appellant_name = (
            plaintiff_profile.get("name", "")
            if appellant_role == "plaintiff"
            else defendant_profile.get("name", "")
        )
        appellee_name = (
            defendant_profile.get("name", "")
            if appellant_role == "plaintiff"
            else plaintiff_profile.get("name", "")
        )
        return {
            "appellant_role": appellant_role,
            "appellee_role": appellee_role,
            "appellant_name": str(appellant_name or ""),
            "appellee_name": str(appellee_name or ""),
        }

    def _collect_case_participant_ids(self, case_id: str, extra_ids: list[str] | None = None) -> list[str]:
        participant_ids: list[str] = []
        for role_bundle in self._get_case_parties(case_id).values():
            client = role_bundle.get("client")
            lawyer_id = role_bundle.get("lawyer_id", "")
            if client:
                participant_ids.append(client.agent_id)
            if lawyer_id:
                participant_ids.append(lawyer_id)
        if extra_ids:
            participant_ids.extend([agent_id for agent_id in extra_ids if agent_id])

        deduped: list[str] = []
        seen: set[str] = set()
        for agent_id in participant_ids:
            if agent_id and agent_id not in seen:
                seen.add(agent_id)
                deduped.append(agent_id)
        return deduped

    def _get_or_assign_character_name(self, agent: Any) -> str:
        configured = str(getattr(agent, "character_name", "") or "").strip()
        if configured:
            return configured

        config_path = getattr(agent, "config_path", None)
        if config_path:
            try:
                config = self.storage.load_agent_config(config_path)
                configured = str(config.get("character_name", "") or "").strip()
                if configured:
                    setattr(agent, "character_name", configured)
                    return configured
            except Exception:
                pass

        configured = random.choice(CHARACTER_POOL)
        setattr(agent, "character_name", configured)
        if config_path:
            try:
                self.storage.update_agent_field(config_path, "character_name", configured)
            except Exception as exc:
                logger.debug("[Orchestrator] Failed to persist character name for %s: %s", getattr(agent, "agent_id", ""), exc)
        return configured

    def _get_character_name_for_client(self, client: Any, party_role: str) -> str:
        del party_role
        return self._get_or_assign_character_name(client)

    def _get_character_name_for_lawyer(self, lawyer: Any) -> str:
        return self._get_or_assign_character_name(lawyer)

    def _find_client_for_case(self, case_id: str, party_role: str = "plaintiff") -> tuple:
        """Find the client agent for a given case_id and party_role.

        Returns:
            (client_agent, client_config_path) or (None, None)
        """
        from pathlib import Path as PathType

        # Priority 1: Try new case-based structure path
        case_agent_path = self.storage.get_case_agent_path(case_id, party_role)
        if (case_agent_path / "config.yaml").exists():
            # Find agent by path
            for client in self.registry.get_agents_by_type("client"):
                if not client.config_path:
                    continue
                client_path = PathType(client.config_path)
                client_dir = client_path.parent if client_path.name == "config.yaml" else client_path
                if client_dir == case_agent_path:
                    return client, client.config_path

        # Priority 2: Search through all registered clients (legacy support)
        normalized_case_id = self._normalize_case_id(case_id)
        for client in self.registry.get_agents_by_type("client"):
            if not client.config_path:
                continue
            config = self.storage.load_agent_config(client.config_path)
            if config.get("party_role") != party_role:
                continue
            config_case_id = self._normalize_case_id(config.get("case_id", ""))
            if config_case_id and config_case_id == normalized_case_id:
                return client, client.config_path
        return None, None

    @staticmethod
    def _stringify_prompt_value(value: Any, fallback: str = "（暂无）") -> str:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if not cleaned:
                return fallback
            return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(cleaned))

        if isinstance(value, dict):
            cleaned = {
                key: item for key, item in value.items()
                if item not in (None, "", [], {})
            }
            if not cleaned:
                return fallback
            return json.dumps(cleaned, ensure_ascii=False, indent=2)

        text = str(value or "").strip()
        return text or fallback

    def _save_result(self, case_id: str, stage: str, result: dict) -> None:
        """Save scenario result to sandbox_data/output/{case_id}/."""
        case_output = self.output_dir / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        filepath = case_output / f"{stage}_result.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"[Orchestrator] Saved {stage} result to {filepath}")

    def _load_consultation_history(self, case_id: str, stage: str = "PLC") -> list[dict[str, Any]]:
        case_output_dir = self._get_case_output_dir(case_id)
        stage_code = str(stage or "PLC").strip().upper()
        candidate_stages = [stage_code]
        if stage_code == "PLC":
            candidate_stages.append("LC")

        for candidate_stage in candidate_stages:
            result_path = case_output_dir / f"{candidate_stage}_result.json"
            if not result_path.exists():
                continue
            try:
                with result_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                dialog_history = payload.get("dialog_history", [])
                if isinstance(dialog_history, list):
                    return dialog_history
            except Exception as exc:
                logger.warning("[Orchestrator] 读取咨询记录失败: case=%s stage=%s error=%s", case_id, candidate_stage, exc)
        return []

    def _collect_stage_prompts(
        self,
        case_id: str,
        stage: str,
        *agents: Any,
        reset: bool = False,
    ) -> None:
        """Persist stage-scoped system prompts into one output JSON file."""
        case_output_dir = self._get_case_output_dir(case_id)
        filepath = case_output_dir / "system_prompt.json"
        export_data: dict[str, Any] = {"case_id": case_id, "stages": {}}

        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    export_data.update(loaded)
            except Exception as e:
                logger.warning(f"[Orchestrator] 读取 system_prompt.json 失败，将重建文件: {e}")

        stages = export_data.setdefault("stages", {})
        if reset or stage not in stages or not isinstance(stages.get(stage), dict):
            stages[stage] = {
                "stage_code": stage,
                "stage_name": self.STAGE_DISPLAY_NAMES.get(stage, stage),
                "agents": [],
            }

        stage_entry = stages[stage]
        existing_agents = {
            agent_info.get("agent_id"): agent_info
            for agent_info in stage_entry.get("agents", [])
            if isinstance(agent_info, dict) and agent_info.get("agent_id")
        }

        for agent in agents:
            if not agent or not hasattr(agent, "get_prompt_info"):
                continue
            prompt_info = agent.get_prompt_info()
            if not prompt_info.get("system_prompt"):
                continue
            existing_agents[prompt_info["agent_id"]] = prompt_info

        now = datetime.now().isoformat()
        stage_entry["agents"] = list(existing_agents.values())
        stage_entry["updated_at"] = now
        export_data["updated_at"] = now

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Orchestrator] 已更新阶段 system prompt 汇总: {filepath}")

    async def _checkpoint_stage_memories(
        self,
        *,
        case_id: str,
        stage_code: str,
        stage_label: str,
        agents: list[Any],
    ) -> None:
        """Persist stage memories without leaving the frontend in a silent gap."""
        checkpoints = [
            self._build_memory_checkpoint_event(agent)
            for agent in list(agents or [])
            if agent is not None and hasattr(agent, "extract_and_save_long_term_memory")
        ]
        checkpoints = [item for item in checkpoints if item]
        if not checkpoints:
            return

        tool_names = self._memory_checkpoint_tool_names(checkpoints)
        skill_names = self._memory_checkpoint_skill_names(checkpoints)

        if self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
            try:
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase="memory_checkpoint",
                    message=f"{stage_label or stage_code}已结束，正在整理阶段材料",
                    detail="整理完成后会自动进入下一阶段",
                    blocking=False,
                    metadata={
                        "stage": stage_code,
                        "scenario_type": stage_code,
                        "agent_count": len(checkpoints),
                        "memory_events": self._public_memory_checkpoint_events(checkpoints),
                        "tool_names": tool_names,
                        "skill_names": skill_names,
                        "active_tool_names": tool_names,
                        "active_skill_names": skill_names,
                    },
                )
            except Exception as exc:
                logger.warning("[Orchestrator] 广播阶段记忆整理进度失败: %s", exc)

        results = await asyncio.gather(
            *[
                asyncio.to_thread(item["agent"].extract_and_save_long_term_memory)
                for item in checkpoints
            ],
            return_exceptions=True,
        )
        for item, result in zip(checkpoints, results):
            if isinstance(result, Exception):
                item["status"] = "failed"
                item["error"] = str(result)
                logger.error(
                    "[Orchestrator] %s 阶段记忆整理失败: agent=%s error=%s",
                    stage_code,
                    item.get("agent_id") or item.get("agent_name"),
                    result,
                )
                continue
            item["status"] = "completed" if isinstance(result, dict) else "skipped"
            after_payload = result if isinstance(result, dict) else {}
            item["changed_fields"] = self._memory_changed_fields(item.get("before_payload"), after_payload)
            item["changed_count"] = len(item["changed_fields"])

        memory_events = self._public_memory_checkpoint_events(checkpoints)
        changed_total = sum(int(item.get("changed_count") or 0) for item in memory_events)
        checked_count = sum(1 for item in memory_events if item.get("status") == "completed")

        if self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
            try:
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase="memory_checkpoint_complete",
                    message=f"{stage_label or stage_code}阶段材料整理完成",
                    detail=f"已检查 {checked_count} 个长期记忆槽，更新 {changed_total} 个字段",
                    blocking=False,
                    metadata={
                        "stage": stage_code,
                        "scenario_type": stage_code,
                        "agent_count": len(memory_events),
                        "checked_count": checked_count,
                        "changed_count": changed_total,
                        "memory_events": memory_events,
                        "tool_names": tool_names,
                        "skill_names": skill_names,
                        "active_tool_names": tool_names,
                        "active_skill_names": skill_names,
                    },
                )
            except Exception as exc:
                logger.warning("[Orchestrator] 广播阶段记忆整理完成失败: %s", exc)

    def _build_memory_checkpoint_event(self, agent: Any) -> dict[str, Any]:
        owner = self._memory_owner_for_agent(agent)
        before_payload: dict[str, Any] = {}
        if owner:
            try:
                before_payload, _paths = load_memory_for_agent(agent, owner)
            except Exception as exc:
                logger.warning(
                    "[Orchestrator] 读取记忆 checkpoint 前置状态失败: agent=%s owner=%s error=%s",
                    getattr(agent, "agent_id", getattr(agent, "name", "")),
                    owner,
                    exc,
                )
        return {
            "agent": agent,
            "agent_id": str(getattr(agent, "agent_id", "") or ""),
            "agent_name": str(getattr(agent, "name", "") or getattr(agent, "agent_id", "") or "未知 Agent"),
            "owner": owner,
            "owner_label": self._memory_owner_label(owner),
            "status": "pending",
            "changed_fields": [],
            "changed_count": 0,
            "before_payload": before_payload,
        }

    @staticmethod
    def _public_memory_checkpoint_events(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        public_keys = {
            "agent_id",
            "agent_name",
            "owner",
            "owner_label",
            "status",
            "changed_fields",
            "changed_count",
            "error",
        }
        return [
            {key: value for key, value in item.items() if key in public_keys and value not in (None, "")}
            for item in checkpoints
        ]

    @staticmethod
    def _memory_owner_for_agent(agent: Any) -> str:
        if hasattr(agent, "legal_profile"):
            return LAWYER_MEMORY_OWNER
        if hasattr(agent, "client_profile"):
            return CLIENT_MEMORY_OWNER
        return ""

    @staticmethod
    def _memory_owner_label(owner: str) -> str:
        if owner == LAWYER_MEMORY_OWNER:
            return "律师长期记忆"
        if owner == CLIENT_MEMORY_OWNER:
            return "当事人长期记忆"
        return "运行代理记忆"

    @staticmethod
    def _memory_changed_fields(before_payload: Any, after_payload: Any) -> list[str]:
        before_flat = flatten_memory_payload(before_payload or {})
        after_flat = flatten_memory_payload(after_payload or {})
        fields = sorted(set(before_flat) | set(after_flat))
        return [field for field in fields if before_flat.get(field, "") != after_flat.get(field, "")]

    @staticmethod
    def _memory_checkpoint_tool_names(checkpoints: list[dict[str, Any]]) -> list[str]:
        tools: set[str] = set()
        owners = {str(item.get("owner") or "") for item in checkpoints}
        if LAWYER_MEMORY_OWNER in owners:
            tools.update({LAWYER_LOAD_TOOL_NAME, LAWYER_SAVE_TOOL_NAME})
        if CLIENT_MEMORY_OWNER in owners:
            tools.update({CLIENT_LOAD_TOOL_NAME, CLIENT_SAVE_TOOL_NAME})
        return sorted(tools)

    @staticmethod
    def _memory_checkpoint_skill_names(checkpoints: list[dict[str, Any]]) -> list[str]:
        skills: set[str] = set()
        owners = {str(item.get("owner") or "") for item in checkpoints}
        if LAWYER_MEMORY_OWNER in owners:
            skills.add("lawyer-memory-writing")
        if CLIENT_MEMORY_OWNER in owners:
            skills.add("client-memory-writing")
        return sorted(skills)

    def _mark_case_stage_active(
        self,
        case_id: str,
        scenario_type: str,
        participant_ids: list[str],
        display_stage_code: str = "",
    ) -> None:
        participant_ids = [pid for pid in participant_ids if pid]
        if not participant_ids:
            return

        self.event_bus.register_active_scenario(case_id, scenario_type, participant_ids)
        if self.checkpoint_manager:
            self.checkpoint_manager.sync_active_scenarios_from_event_bus()
        if self.map_engine and hasattr(self.map_engine, "broadcast_scenario_start"):
            frontend_stage_code = str(display_stage_code or scenario_type or "").strip().upper()
            asyncio.get_running_loop().create_task(
                self.map_engine.broadcast_scenario_start(case_id, frontend_stage_code, participant_ids)
            )

    def _clear_case_stage_active(self, case_id: str) -> None:
        active_snapshot = self.event_bus.get_active_scenarios_snapshot().get(case_id, {})
        scenario_type = str(active_snapshot.get("scenario_type", "") or "")
        self.event_bus.unregister_active_scenario(case_id)
        if self.checkpoint_manager:
            self.checkpoint_manager.sync_active_scenarios_from_event_bus()
        if scenario_type and self.map_engine and hasattr(self.map_engine, "broadcast_scenario_end"):
            asyncio.get_running_loop().create_task(
                self.map_engine.broadcast_scenario_end(case_id, scenario_type)
            )

    @staticmethod
    def _normalize_firm_id(firm_id: str) -> str:
        key = str(firm_id or "").strip().lower()
        if key in {"lawfirma", "law_firm_a"}:
            return "law_firm_A"
        if key in {"lawfirmb", "law_firm_b"}:
            return "law_firm_B"
        return firm_id

    def _get_available_firm_ids(self) -> list[str]:
        firm_ids: list[str] = []
        for firm_id in self.registry._firms.keys():
            normalized = self._normalize_firm_id(str(firm_id))
            if normalized and normalized not in firm_ids:
                firm_ids.append(normalized)
        return firm_ids or ["law_firm_A", "law_firm_B"]

    def _resolve_map_prefix_from_firm(self, firm_id: str) -> str:
        normalized = self._normalize_firm_id(firm_id)
        if normalized == "law_firm_B":
            return "lawfirmB"
        return "lawfirmA"

    def _choose_case_firm(
        self,
        *,
        config_path: str | None = None,
        preferred_firm: str = "",
        force_random: bool = False,
    ) -> tuple[str, str]:
        firms = self._get_available_firm_ids()
        normalized_preferred = self._normalize_firm_id(preferred_firm)
        if force_random or normalized_preferred not in firms:
            target_firm = random.choice(firms)
        else:
            target_firm = normalized_preferred

        if config_path and target_firm:
            try:
                self.storage.update_agent_field(config_path, "assigned_firm", target_firm)
            except Exception as exc:
                logger.warning("[Orchestrator] 更新 assigned_firm 失败: %s", exc)

        return target_firm, self._resolve_map_prefix_from_firm(target_firm)

    def _infer_case_firm_for_defendant(self, payload: dict, plaintiff_config: dict) -> str:
        candidates: list[str] = [
            str(payload.get("firm_id", "") or ""),
            str(payload.get("target_firm", "") or ""),
            str(plaintiff_config.get("assigned_firm", "") or ""),
        ]

        plaintiff_lawyer_id = str(
            plaintiff_config.get("assigned_lawyer_id", "") or payload.get("lawyer_id", "") or ""
        ).strip()
        if plaintiff_lawyer_id:
            lawyer = self.registry.get_agent(plaintiff_lawyer_id) if self.registry else None
            candidates.append(str(getattr(lawyer, "firm_id", "") or ""))

        available_firms = set(self._get_available_firm_ids())
        for candidate in candidates:
            normalized = self._normalize_firm_id(candidate)
            if normalized in available_firms:
                return normalized
        return ""

    def _select_available_judge(self, court_level: str, case_id: str, preferred_judge_id: str = "") -> Any | None:
        judges = [
            judge
            for judge in self.registry.get_agents_by_type("judge")
            if getattr(judge, "court_level", "") == court_level
        ]
        if not judges:
            return None

        if preferred_judge_id:
            preferred = next((judge for judge in judges if judge.agent_id == preferred_judge_id), None)
            if preferred and self._is_judge_available(preferred.agent_id, case_id):
                return preferred

        for judge in judges:
            if self._is_judge_available(judge.agent_id, case_id):
                return judge
        return None

    def _is_judge_available(self, judge_id: str, case_id: str) -> bool:
        reservation_case_id = self._judge_reservations.get(judge_id)
        if reservation_case_id and reservation_case_id != case_id:
            return False
        if self.event_bus.is_agent_busy(judge_id):
            return False

        judge = self.registry.get_agent(judge_id)
        current_case_id = getattr(judge, "current_handling_case", None) if judge else None
        if current_case_id and current_case_id != case_id:
            return False
        return True

    def _reserve_trial_resources(self, court: str, case_id: str, judge_id: str) -> None:
        self._court_reservations[court] = case_id
        self._judge_reservations[judge_id] = case_id
        judge = self.registry.get_agent(judge_id)
        if judge and getattr(judge, "config_path", None):
            try:
                self.storage.update_agent_field(judge.config_path, "current_handling_case", case_id)
            except Exception as exc:
                logger.warning("[Orchestrator] failed to reserve judge %s for %s: %s", judge_id, case_id, exc)

    async def _schedule_trial_entry(
        self,
        *,
        case_id: str,
        payload: dict[str, Any],
        court: str,
        court_level: str,
        event_type: EventType,
    ) -> None:
        publish_payload: dict[str, Any] | None = None
        async with self._resource_lock:
            if self._court_reservations.get(court) == case_id:
                return
            if any(item["case_id"] == case_id for item in self._trial_queues[court]):
                return

            preferred_judge_id = str(payload.get("judge_id", "") or "")
            judge = None
            if court not in self._court_reservations:
                judge = self._select_available_judge(court_level, case_id, preferred_judge_id)

            if judge:
                self._reserve_trial_resources(court, case_id, judge.agent_id)
                publish_payload = {**payload, "judge_id": judge.agent_id}
            else:
                self._trial_queues[court].append({
                    "case_id": case_id,
                    "court_level": court_level,
                    "event_type": event_type,
                    "payload": dict(payload),
                })
                logger.info("[Scheduler] queued %s for %s", case_id, court)

        if publish_payload:
            logger.info("[Scheduler] dispatching %s to %s", case_id, court)
            await self.event_bus.publish(event_type, publish_payload)

    async def _release_trial_slot(self, court: str, case_id: str) -> None:
        next_dispatch: tuple[EventType, dict[str, Any]] | None = None

        async with self._resource_lock:
            if self._court_reservations.get(court) == case_id:
                self._court_reservations.pop(court, None)

            released_judge_ids = [
                judge_id
                for judge_id, reserved_case_id in list(self._judge_reservations.items())
                if reserved_case_id == case_id
            ]
            for judge_id in released_judge_ids:
                self._judge_reservations.pop(judge_id, None)
                judge = self.registry.get_agent(judge_id)
                if judge and getattr(judge, "config_path", None):
                    try:
                        self.storage.update_agent_field(judge.config_path, "current_handling_case", None)
                    except Exception as exc:
                        logger.warning("[Scheduler] failed to release judge %s: %s", judge_id, exc)

            queue = self._trial_queues[court]
            while queue:
                next_item = queue.popleft()
                judge = self._select_available_judge(
                    next_item["court_level"],
                    next_item["case_id"],
                    str(next_item["payload"].get("judge_id", "") or ""),
                )
                if not judge:
                    queue.appendleft(next_item)
                    break

                self._reserve_trial_resources(court, next_item["case_id"], judge.agent_id)
                next_dispatch = (
                    next_item["event_type"],
                    {**next_item["payload"], "judge_id": judge.agent_id},
                )
                break

        if next_dispatch:
            logger.info("[Scheduler] releasing %s triggered queued case %s", court, next_dispatch[1].get("case_id", ""))
            await self.event_bus.publish(next_dispatch[0], next_dispatch[1])

    @staticmethod
    def _sanitize_bubble_text(content: Any, max_length: int = 80) -> str:
        text = str(content or "").strip()
        if not text:
            return ""

        text = text.replace("【起草结束】", "").replace("【提取结束】", "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        text = " ".join(text.split())
        if len(text) > max_length:
            text = text[: max_length - 1].rstrip() + "…"
        return text

    async def _play_dialog_bubbles(
        self,
        dialog_history: list[dict[str, Any]],
        role_to_agent_id: dict[str, str],
        role_to_location_id: dict[str, str] | None = None,
        role_to_direction: dict[str, str] | None = None,
        gap: float = 0.9,
    ) -> None:
        if not self.map_engine or not dialog_history:
            return

        await self._prepare_dialogue_agents(
            role_to_agent_id,
            role_to_location_id=role_to_location_id,
            role_to_direction=role_to_direction,
        )

        for entry in dialog_history:
            if not await self._show_dialog_entry_bubble(entry, role_to_agent_id):
                return
            await asyncio.sleep(gap)

    async def _show_dialog_entry_bubble(
        self,
        entry: dict[str, Any],
        role_to_agent_id: dict[str, str],
    ) -> bool:
        if not self.map_engine:
            return False

        agent_id = role_to_agent_id.get(str(entry.get("role", "") or ""))
        if not agent_id:
            return True

        bubble_text = self._sanitize_bubble_text(entry.get("content", ""))
        if not bubble_text:
            return True

        duration = min(2.6, max(1.4, len(bubble_text) * 0.04))
        try:
            await self.map_engine.show_bubble(agent_id, bubble_text, duration)
        except Exception as exc:
            logger.warning("[Orchestrator] 显示气泡失败: agent=%s, error=%s", agent_id, exc)
            return False
        return True

    async def _broadcast_dialog_entry(
        self,
        case_id: str,
        entry: dict[str, Any],
        role_to_agent_id: dict[str, str],
        turn: int,
        scenario_type: str = "",
    ) -> None:
        if not self.map_engine or not hasattr(self.map_engine, "broadcast_dialogue"):
            return

        role = str(entry.get("role", "") or "")
        content = str(entry.get("content", "") or "").strip()
        agent_id = role_to_agent_id.get(role, "")
        if not case_id or not agent_id or not content:
            return

        speaker_name = role
        agent = self.registry.get_agent(agent_id) if self.registry else None
        if agent and getattr(agent, "name", ""):
            speaker_name = str(agent.name)

        marker = build_player_responsibility_marker(
            role=role,
            stage=scenario_type or entry.get("scenario_type", ""),
            player_lawyer_enabled=self._player_plaintiff_lawyer_enabled(),
            ai_surrogate_enabled=self._player_ai_surrogate_enabled(),
            content=content,
        )

        await self.map_engine.broadcast_dialogue(
            case_id,
            agent_id,
            speaker_name,
            content,
            turn,
            scenario_type=scenario_type,
            generation_duration_seconds=entry.get("generation_duration_seconds"),
            generation_total_tokens=entry.get("generation_total_tokens"),
            **(marker or {}),
        )

    def _register_player_document_followup_session(
        self,
        *,
        request_id: str,
        case_id: str,
        stage: str,
        client: Any,
        client_role: str,
        lawyer: Any,
        dialog_history: list[dict[str, Any]],
        role_to_agent_id: dict[str, str],
    ) -> None:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return
        self._player_document_followup_sessions[normalized_request_id] = {
            "case_id": str(case_id or "").strip(),
            "stage": str(stage or "").strip().upper(),
            "client": client,
            "client_role": str(client_role or "plaintiff").strip() or "plaintiff",
            "lawyer": lawyer,
            "dialog_history": dialog_history,
            "role_to_agent_id": dict(role_to_agent_id),
        }

    def _unregister_player_document_followup_session(self, request_id: str) -> None:
        self._player_document_followup_sessions.pop(str(request_id or "").strip(), None)

    def _restore_player_document_followup_session(self, *, request: Any) -> dict[str, Any] | None:
        request_id = str(getattr(request, "request_id", "") or "").strip()
        case_id = str(getattr(request, "case_id", "") or "").strip()
        stage = str(getattr(request, "stage", "") or "").strip().upper()
        if not request_id or not case_id or stage not in {"CD", "DD", "AD", "AR"}:
            return None

        client_role = {
            "AD": "appellant",
            "AR": "appellee",
            "DD": "defendant",
        }.get(stage, "plaintiff")
        party_role = "defendant" if client_role == "defendant" else "plaintiff"
        client, client_path = self._find_client_for_case(case_id, party_role=party_role)
        if client is None or not client_path:
            return None

        client_config = self.storage.load_agent_config(client_path)
        lawyer_id = str(client_config.get("assigned_lawyer_id") or "").strip()
        lawyer = self.registry.get_agent(lawyer_id) if lawyer_id else None
        if lawyer is None:
            return None

        self._ensure_player_document_client_active(
            case_id=case_id,
            client_path=str(client_path),
            client=client,
            client_role=client_role,
            stage=stage,
        )
        dialog_history: list[dict[str, Any]] = []
        self._register_player_document_followup_session(
            request_id=request_id,
            case_id=case_id,
            stage=stage,
            client=client,
            client_role=client_role,
            lawyer=lawyer,
            dialog_history=dialog_history,
            role_to_agent_id={
                client_role: client.agent_id,
                "plaintiff": client.agent_id,
                "lawyer": lawyer.agent_id,
            },
        )
        logger.info("[Orchestrator] Restored player document follow-up session: request=%s stage=%s", request_id, stage)
        return self._player_document_followup_sessions.get(request_id)

    async def handle_player_document_followup(self, *, request_id: str, message: str, request: Any = None) -> dict[str, str]:
        normalized_request_id = str(request_id or "").strip()
        question = str(message or "").strip()
        if not question:
            raise ValueError("message is required.")
        session = self._player_document_followup_sessions.get(normalized_request_id)
        if not session and request is not None:
            session = self._restore_player_document_followup_session(request=request)
        if not session:
            raise RuntimeError("当前文书任务没有可追问的当事人会话。")

        case_id = str(session["case_id"])
        stage = str(session["stage"])
        role_to_agent_id = dict(session["role_to_agent_id"])
        dialog_history = session["dialog_history"]
        lawyer_turn = len(dialog_history) + 1
        lawyer_entry = {
            "turn": lawyer_turn,
            "role": "lawyer",
            "content": question,
            "timestamp": datetime.now().isoformat(),
        }
        dialog_history.append(lawyer_entry)
        await self._broadcast_dialog_entry(
            case_id,
            {"role": "lawyer", "content": question},
            role_to_agent_id,
            lawyer_turn,
            scenario_type=stage,
        )

        answer = str(await asyncio.to_thread(session["client"].step, question) or "").strip()
        client_turn = len(dialog_history) + 1
        client_entry = {
            "turn": client_turn,
            "role": session["client_role"],
            "content": answer,
            "timestamp": datetime.now().isoformat(),
        }
        dialog_history.append(client_entry)
        await self._broadcast_dialog_entry(
            case_id,
            {"role": session["client_role"], "content": answer},
            role_to_agent_id,
            client_turn,
            scenario_type=stage,
        )
        return {"question": question, "answer": answer}

    async def _ensure_agent_visualized(
        self,
        agent_id: str,
        role_hint: str = "",
        location_id: str = "",
        direction: str = "down",
    ) -> None:
        if not self.map_engine or not agent_id:
            return

        agent = self.registry.get_agent(agent_id)
        if not agent:
            return

        if agent_id not in getattr(self.map_engine, "_agent_states", {}):
            if role_hint in {"plaintiff", "defendant"}:
                character_name = self._get_character_name_for_client(agent, role_hint)
            else:
                character_name = self._get_character_name_for_lawyer(agent)
            await self.map_engine.spawn_agent(
                agent_id=agent_id,
                name=agent.name,
                character_name=character_name,
                birth_loc_id=self._get_birth_location_for_agent(agent_id),
                role=role_hint,
            )

        if location_id:
            await self.map_engine.move_to_location(agent_id, location_id)
            await self._stand_agent_on_location(
                agent_id,
                location_id,
                direction=direction,
            )

    async def _prepare_dialogue_agents(
        self,
        role_to_agent_id: dict[str, str],
        role_to_location_id: dict[str, str] | None = None,
        role_to_direction: dict[str, str] | None = None,
    ) -> None:
        if not self.map_engine:
            return

        role_to_location_id = role_to_location_id or {}
        role_to_direction = role_to_direction or {}

        for role, agent_id in role_to_agent_id.items():
            await self._ensure_agent_visualized(
                agent_id,
                role_hint=role,
                location_id=role_to_location_id.get(role, ""),
                direction=role_to_direction.get(role, "down"),
            )

    async def _run_sync_scenario_with_live_bubbles(
        self,
        case_id: str,
        scenario_factory: Callable[[Callable[[str, str], None] | None], Any],
        role_to_agent_id: dict[str, str],
        role_to_location_id: dict[str, str] | None = None,
        role_to_direction: dict[str, str] | None = None,
        gap: float = 0.9,
        trace_recorder: CaseAgentTraceRecorder | None = None,
        trace_stage_code: str = "",
        trace_stage_key: str = "",
        trace_agents: list[Any] | None = None,
        trace_result_path: str | Path | None = None,
    ) -> dict[str, Any]:
        def _attach_trace(scenario: Any) -> None:
            if scenario is None or trace_recorder is None:
                return
            setattr(scenario, "trace_recorder", trace_recorder)
            setattr(scenario, "trace_stage_code", str(trace_stage_code or "").strip().upper())
            setattr(
                scenario,
                "trace_stage_key",
                str(trace_stage_key or trace_stage_code or "").strip().upper(),
            )

        if not self.map_engine:
            scenario = scenario_factory(None)
            _attach_trace(scenario)
            try:
                result = await asyncio.to_thread(scenario.execute)
            except Exception as exc:
                if trace_recorder is not None and trace_stage_code:
                    trace_recorder.export_stage(
                        stage_code=trace_stage_code,
                        stage_key=trace_stage_key or trace_stage_code,
                        agents=list(trace_agents or []),
                        stage_result=None,
                        stage_result_path=trace_result_path,
                        status="failed",
                        error=repr(exc),
                    )
                raise
            if trace_recorder is not None and trace_stage_code:
                trace_recorder.export_stage(
                    stage_code=trace_stage_code,
                    stage_key=trace_stage_key or trace_stage_code,
                    agents=list(trace_agents or []),
                    stage_result=result,
                    stage_result_path=trace_result_path,
                    status="completed",
                )
            return result

        await self._prepare_dialogue_agents(
            role_to_agent_id,
            role_to_location_id=role_to_location_id,
            role_to_direction=role_to_direction,
        )

        bubble_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        async def consume_bubbles() -> None:
            turn = 0
            while True:
                entry = await bubble_queue.get()
                try:
                    if entry is None:
                        return
                    turn += 1
                    await self._broadcast_dialog_entry(
                        case_id,
                        entry,
                        role_to_agent_id,
                        turn,
                        scenario_type=trace_stage_key or trace_stage_code,
                    )
                    await self._show_dialog_entry_bubble(entry, role_to_agent_id)
                    await asyncio.sleep(gap)
                finally:
                    bubble_queue.task_done()

        consumer_task = asyncio.create_task(consume_bubbles())

        def bubble_publisher(role: str, content: str, entry: dict[str, Any] | None = None) -> None:
            loop.call_soon_threadsafe(
                bubble_queue.put_nowait,
                dict(entry or {"role": role, "content": content}),
            )

        try:
            scenario = scenario_factory(bubble_publisher)
            _attach_trace(scenario)
            try:
                result = await asyncio.to_thread(scenario.execute)
            except Exception as exc:
                if trace_recorder is not None and trace_stage_code:
                    trace_recorder.export_stage(
                        stage_code=trace_stage_code,
                        stage_key=trace_stage_key or trace_stage_code,
                        agents=list(trace_agents or []),
                        stage_result=None,
                        stage_result_path=trace_result_path,
                        status="failed",
                        error=repr(exc),
                    )
                raise
            if trace_recorder is not None and trace_stage_code:
                trace_recorder.export_stage(
                    stage_code=trace_stage_code,
                    stage_key=trace_stage_key or trace_stage_code,
                    agents=list(trace_agents or []),
                    stage_result=result,
                    stage_result_path=trace_result_path,
                    status="completed",
                )
            return result
        finally:
            await bubble_queue.join()
            await bubble_queue.put(None)
            await consumer_task

    async def _stand_agent_on_location(
        self,
        agent_id: str,
        loc_id: str,
        direction: str = "down",
        y_offset: float = 0.0,
    ) -> None:
        if not self.map_engine:
            return

        loc = None
        if getattr(self.map_engine, "registry", None):
            loc = self.map_engine.registry.get(loc_id)

        if loc:
            try:
                await self.map_engine.stand_agent(
                    agent_id,
                    direction_override=direction,
                    x=loc.x,
                    y=loc.y + y_offset,
                )
            except TypeError:
                await self.map_engine.stand_agent(agent_id, direction_override=direction)
            return

        await self.map_engine.stand_agent(agent_id, direction_override=direction)


    # ══════════════════════════════════════════════════════════
    #  位置占用管理
    # ══════════════════════════════════════════════════════════

    def _occupy_location(self, loc_id: str, agent_id: str) -> bool:
        """占用位置，返回是否成功。"""
        if loc_id in self._occupied_locations:
            logger.warning(f"[Location] {loc_id} 已被 {self._occupied_locations[loc_id]} 占用")
            return False
        self._occupied_locations[loc_id] = agent_id
        logger.debug(f"[Location] {agent_id} 占用 {loc_id}")
        return True

    def _release_location(self, loc_id: str) -> None:
        """释放位置。"""
        if loc_id in self._occupied_locations:
            agent_id = self._occupied_locations.pop(loc_id)
            logger.debug(f"[Location] {agent_id} 释放 {loc_id}")

    def _get_live_occupied_location_ids(self) -> set[str]:
        """Infer occupied chairs/sofas from current frontend agent states."""
        if not self.map_engine or not self.map_engine.registry:
            return set()

        tracked_locations = {
            **self.map_engine.registry.lawfirm_chairs,
            **self.map_engine.registry.lawfirm_sofas,
            **self.map_engine.registry.lawfirm_waiting_spots,
        }
        occupied: set[str] = set()
        for state in getattr(self.map_engine, "_agent_states", {}).values():
            sitting = state.get("sitting") or {}
            x = sitting.get("x")
            y = sitting.get("y")
            if x is None or y is None:
                continue
            for loc_id, loc in tracked_locations.items():
                if abs(loc.x - x) < 0.5 and abs(loc.y - y) < 0.5:
                    occupied.add(loc_id)
        return occupied

    def _get_reception_reserved_sofa_ids(self) -> set[str]:
        reserved: set[str] = set()
        for agent in self.registry.get_all_agents():
            queued_client_sofas = getattr(agent, "_queued_client_sofas", None)
            if isinstance(queued_client_sofas, dict):
                reserved.update(str(sofa_id) for sofa_id in queued_client_sofas.values() if sofa_id)
        return reserved

    def _get_reception_reserved_waiting_spot_ids(self) -> set[str]:
        reserved: set[str] = set()
        for agent in self.registry.get_all_agents():
            queued_client_wait_spots = getattr(agent, "_queued_client_wait_spots", None)
            if isinstance(queued_client_wait_spots, dict):
                reserved.update(str(wait_id) for wait_id in queued_client_wait_spots.values() if wait_id)
        return reserved

    def _get_available_sofa(self, firm_id: str) -> str | None:
        """获取律所中第一个空闲沙发。"""
        if not self.map_engine or not self.map_engine.registry:
            return None
        occupied = (
            set(self._occupied_locations.keys())
            | self._get_live_occupied_location_ids()
            | self._get_reception_reserved_sofa_ids()
        )
        return self.map_engine.registry.get_available_sofa(firm_id, occupied)

    def _get_available_chair_pair(self, firm_id: str) -> tuple[str | None, str | None]:
        """获取一对空闲会议椅（客户侧 left + 律师侧 right）。"""
        if not self.map_engine or not self.map_engine.registry:
            return None, None
        occupied = set(self._occupied_locations.keys()) | self._get_live_occupied_location_ids()
        return self.map_engine.registry.get_meeting_chair_pair(firm_id, occupied)

    def _get_available_waiting_spot(self, firm_id: str) -> tuple[str | None, Any]:
        """获取律所内可用的站立候位点。"""
        if not self.map_engine or not self.map_engine.registry:
            return None, None
        occupied = (
            set(self._occupied_locations.keys())
            | self._get_live_occupied_location_ids()
            | self._get_reception_reserved_waiting_spot_ids()
        )
        return self.map_engine.registry.get_available_waiting_spot(firm_id, occupied)

    def _reserve_lawyer_workspace(self, lawyer: Any) -> tuple[str, str, str]:
        """为律师分配一个临时工作位，避免占用正式法庭席位。"""
        map_prefix = self._resolve_map_prefix_from_lawyer(lawyer)

        _client_chair, lawyer_chair = self._get_available_chair_pair(map_prefix)
        if lawyer_chair and self._occupy_location(lawyer_chair, lawyer.agent_id):
            loc = self.map_engine.registry.get(lawyer_chair) if self.map_engine and self.map_engine.registry else None
            return lawyer_chair, getattr(loc, "direction", "") or "right", lawyer_chair

        wait_spot_id, wait_spot = self._get_available_waiting_spot(map_prefix)
        if wait_spot_id and self._occupy_location(wait_spot_id, lawyer.agent_id):
            return wait_spot_id, getattr(wait_spot, "direction", "") or "down", wait_spot_id

        fallback_loc_id = self._resolve_birth_location_for_map_prefix(map_prefix)
        logger.warning(
            "[Choreography] 律师 %s 无可用工作位，回退到出生点 %s",
            getattr(lawyer, "name", getattr(lawyer, "agent_id", "")),
            fallback_loc_id,
        )
        return fallback_loc_id, "down", ""

    async def _cleanup_visualized_agent(self, agent_id: str, reserved_loc_id: str = "") -> None:
        """将临时可视化的 Agent 离场，并释放占用位置。"""
        try:
            await self._return_agents_to_birth_and_despawn([agent_id])
        finally:
            if reserved_loc_id:
                self._release_location(reserved_loc_id)

    # ══════════════════════════════════════════════════════════
    #  等候队列管理
    # ══════════════════════════════════════════════════════════

    async def _add_to_waiting_queue(
        self,
        firm_id: str,
        client_id: str,
        case_id: str,
        wait_loc_id: str,
        party_role: str = "plaintiff",
        wait_mode: str = "sofa",
    ) -> None:
        """将当事人加入等候队列。"""
        if firm_id not in self._waiting_queues:
            self._waiting_queues[firm_id] = []

        self._waiting_queues[firm_id].append({
            "client_id": client_id,
            "case_id": case_id,
            "wait_loc_id": wait_loc_id,
            "wait_mode": wait_mode,
            "party_role": party_role,  # 保存 party_role
        })

        from ..core.event_bus import EventType
        await self.event_bus.publish(EventType.CLIENT_WAITING, {
            "firm_id": firm_id,
            "client_id": client_id,
            "case_id": case_id,
            "queue_position": len(self._waiting_queues[firm_id]),
        })
        logger.info(
            f"[WaitingQueue] {client_id} 加入 {firm_id} 等候队列 "
            f"(位置: {len(self._waiting_queues[firm_id])})"
        )

    async def _notify_next_waiting_client(self, lawyer_id: str, firm_id: str) -> None:
        """律师空闲后通知下一个等候的当事人。"""
        queue = self._waiting_queues.get(firm_id, [])
        if not queue:
            logger.info(f"[WaitingQueue] {firm_id} 等候队列为空，律师 {lawyer_id} 待命")
            return

        next_client_info = queue.pop(0)
        client_id = next_client_info["client_id"]
        case_id = next_client_info["case_id"]
        wait_loc_id = next_client_info.get("wait_loc_id") or next_client_info.get("sofa_id", "")
        wait_mode = next_client_info.get("wait_mode", "sofa")
        party_role = next_client_info.get("party_role", "plaintiff")  # 获取 party_role

        logger.info(f"[WaitingQueue] 通知 {client_id} 从{wait_mode}前往咨询区")

        # 释放等待位置
        if wait_loc_id:
            self._release_location(wait_loc_id)

        # 发布通知事件，重新触发分配流程
        from ..core.event_bus import EventType
        await self.event_bus.publish(EventType.CLIENT_CALLED, {
            "client_id": client_id,
            "case_id": case_id,
            "lawyer_id": lawyer_id,
            "firm_id": firm_id,
            "party_role": party_role,  # 传递 party_role
        })

    def _is_agent_available(self, agent_id: str) -> bool:
        """检查 Agent 是否空闲（不再读取文件，改为查询 EventBus）。

        Args:
            agent_id: Agent ID

        Returns:
            True 如果 Agent 空闲，False 如果正在参与活跃场景
        """
        is_busy = self.event_bus.is_agent_busy(agent_id)
        agent = self.registry.get_agent(agent_id)
        agent_name = agent.name if agent else agent_id
        current_case_id = getattr(agent, "current_handling_case", None) if agent else None
        is_available = not is_busy and not current_case_id

        logger.debug(
            f"[Orchestrator] Agent {agent_name} 状态检查: "
            f"busy={is_busy}, current_case={current_case_id}, available={is_available}"
        )
        return is_available

    # ══════════════════════════════════════════════════════════
    #  Scenario Runners
    # ══════════════════════════════════════════════════════════

    async def _run_consultation(self, payload: dict) -> None:
        """Run Legal Consultation (LC) scenario."""
        from ..scenarios.legal_consultation import LegalConsultationScenario
        from ..core.event_bus import EventType
        from .case_fsm import CaseState

        case_id = payload.get("case_id", "")
        party_role = payload.get("party_role", "plaintiff")
        lawyer_id = payload.get("lawyer_id", "")

        lawyer = self.registry.get_agent(lawyer_id)
        if not lawyer:
            logger.error(f"[Orchestrator] Lawyer {lawyer_id} not found")
            return

        # ── Player-lawyer adapter (feature-gated) ──
        _player_adapter = None
        if (
            party_role == "plaintiff"
            and self._player_plaintiff_lawyer_enabled()
            and not self._player_ai_surrogate_enabled()
        ):
            from ..player_lawyer.agent import PlayerPlaintiffLawyerAgent
            _player_gateway = getattr(self, "_player_gateway", None)
            if _player_gateway is not None:
                _player_adapter = PlayerPlaintiffLawyerAgent(
                    agent_id=lawyer.agent_id,
                    name=lawyer.name,
                    law_firm=getattr(lawyer, "law_firm", ""),
                    firm_id=getattr(lawyer, "firm_id", ""),
                    gateway=_player_gateway,
                    case_id=case_id,
                    sandbox_id=getattr(self, "_sandbox_id", 0),
                    broadcast_fn=getattr(self, "_player_broadcast_fn", None),
                )
                _player_adapter.config_path = lawyer.config_path
                _player_adapter.storage = lawyer.storage
                _player_adapter.set_stage("LC")
                lawyer = _player_adapter
                logger.info("[Orchestrator] Using player plaintiff-lawyer adapter for LC: %s", lawyer.name)

        # Find the correct client for this role
        client, client_path = self._find_client_for_case(case_id, party_role=party_role)
        if not client:
            # Fallback: search all clients and check role + case_id directly
            clients = self.registry.get_agents_by_type("client")
            for c in clients:
                if c.config_path:
                    cfg = self.storage.load_agent_config(c.config_path)
                    if cfg.get("party_role") == party_role:
                        # Use loose matching for case_id
                        c_case_id = str(cfg.get("case_id", ""))
                        if c_case_id == case_id or f"case_{c_case_id}" == case_id:
                            client = c
                            client_path = c.config_path
                            break

        if not client or not client_path:
            logger.error(f"[Orchestrator] No client found for case {case_id} with role {party_role}")
            return

        logger.info(f"[Orchestrator] Starting LC: lawyer={lawyer.name}, client={client.name}, role={party_role}")

        scenario_id = f"LC_{case_id}_{party_role}"
        case_output_dir = self._get_case_output_dir(case_id)
        display_stage_code = self._consultation_display_stage_code(party_role)
        output_path = str(case_output_dir / f"{display_stage_code}_result.json")
        trace_stage_key = f"{display_stage_code}_{party_role}".upper()
        trace_recorder: CaseAgentTraceRecorder | None = None
        scenario_succeeded = False

        # 注册活跃场景到 EventBus（在激活 Agent 之前）
        self._mark_case_stage_active(case_id, "LC", [client.agent_id, lawyer.agent_id], display_stage_code=display_stage_code)

        # 注册场景到 CheckpointManager
        if self.checkpoint_manager:
            self.checkpoint_manager.register_scenario(
                scenario_id=scenario_id,
                case_id=case_id,
                scenario_type="LC",
                party_role=party_role,
                client_id=client.agent_id,
                lawyer_id=lawyer.agent_id,
            )
            # 同步活跃场景到检查点
            self.checkpoint_manager.sync_active_scenarios_from_event_bus()

        # 发送前台交互消息：当事人移动到前台并显示对话
        if self.map_engine:
            # 获取律所信息
            lawfirm = payload.get("map_prefix", "lawfirmA")

            # 发送当事人移动到前台的消息
            await self.map_engine.send_goto_front_desk(
                agent_id=client.agent_id,
                lawfirm=lawfirm,
                dialogue_text="您好，我想咨询一下法律问题，请问律师在吗？"
            )

        # 注意：不在这里更新状态！状态已经在 CASE_ASSIGNED 事件中更新为 "原告咨询中"
        # 这里只负责执行咨询场景

        try:
            # Load case data
            data_loader, case, client_config = self._load_case_data(client_path)
            del client_config

            default_claims = data_loader.extract_claims(case)
            default_evidence = (
                data_loader.extract_plaintiff_evidence(case)
                if party_role == "plaintiff"
                else data_loader.extract_defendant_evidence(case)
            )
            default_position = (
                default_claims
                if party_role == "plaintiff"
                else data_loader.extract_defendant_defense(case)
            )
            case_context = self._build_case_party_context(
                case_id,
                party_role=party_role,
                case=case,
                default_case_background=data_loader.extract_case_background(case),
                default_claims=default_claims,
                default_evidence=default_evidence,
            )

            # Extract profile and questions based on role
            if party_role == "plaintiff":
                profile_data = data_loader.extract_plaintiff_profile(case)
            else:
                profile_data = data_loader.extract_defendant_profile(case)

            scenario_data = {
                "case_background": case_context.get("case_background", ""),
                "questions": profile_data.get("questions", []),
                "claims": case_context.get("claims", ""),
                "my_position": self._stringify_prompt_value(default_position, fallback=""),
                "evidence": case_context.get("evidence", ""),
                "case_cause": data_loader.extract_case_cause(case),
                "current_lawyer_name": lawyer.name,
                "current_lawyer_firm": lawyer.law_firm,
                "case_output_dir": str(case_output_dir.resolve()),
            }

            # Build prompts via PromptAssembler
            lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "LC", scenario_data)
            lawyer_config = self.storage.load_agent_config(lawyer.config_path) if lawyer.config_path else {}
            del lawyer_config
            lawyer_memory = self._get_lawyer_prompt_memory(lawyer, case_id)
            lawyer_prompt = PromptAssembler.build(
                profile=self._build_lawyer_profile(lawyer),
                long_term_memory=lawyer_memory,
                memory_owner=LAWYER_MEMORY_OWNER,
                scenario_prompt=lawyer_scenario,
            )

            client_scenario = PromptAssembler.build_scenario_prompt("client", "LC", scenario_data)
            client_memory = self._get_client_prompt_memory(client, case_id)
            client_prompt = PromptAssembler.build(
                profile=self._build_client_prompt_profile(client, profile_data),
                long_term_memory=client_memory,
                memory_owner=CLIENT_MEMORY_OWNER,
                scenario_prompt=client_scenario,
            )

            # Activate agents
            lawyer.activate(lawyer_prompt)

            # Set scenario_data for client before activation
            client.scenario_data = scenario_data
            client.activate(client_prompt)
            self._configure_stage_tools(
                "LC",
                {
                    "client": client,
                    "lawyer": lawyer,
                },
            )
            trace_recorder = self._bind_case_stage_trace_agents(
                case_id,
                display_stage_code,
                trace_stage_key,
                [client, lawyer],
            )
            self._collect_stage_prompts(case_id, "LC", client, lawyer, reset=True)
            await self._emit_runtime_stage_start(
                case_id=case_id,
                stage_code=display_stage_code,
                trace_recorder=trace_recorder,
            )
            await self._emit_runtime_stage_research(
                case_id=case_id,
                stage_code=display_stage_code,
                case_cause=scenario_data.get("case_cause", ""),
                case_background=scenario_data.get("case_background", ""),
                trace_recorder=trace_recorder,
            )

            # 确保律师已经在地图上生成（防止场景卡死等待律师加入）
            if self.map_engine and lawyer.agent_id not in self.map_engine._agent_states:
                logger.info(f"[Orchestrator] 律师 {lawyer.name} 尚未生成，先生成律师精灵")
                await self.map_engine.spawn_agent(
                    agent_id=lawyer.agent_id,
                    name=lawyer.name,
                    character_name=self._get_character_name_for_lawyer(lawyer),
                    birth_loc_id=self._get_birth_location_for_agent(lawyer.agent_id),
                    role="lawyer",
                )

            scenario = LegalConsultationScenario(
                client_agent=client,
                lawyer_agent=lawyer,
                max_turns=self._resolve_lc_max_turns(
                    len(profile_data.get("questions") or []),
                    player_lawyer_enabled=_player_adapter is not None,
                ),
                output_path=output_path,
                verbose=SCENARIO_VERBOSE,
                map_engine=self.map_engine,
                checkpoint_manager=self.checkpoint_manager,
                scenario_id=scenario_id,
                trace_recorder=trace_recorder,
                trace_stage_code=display_stage_code,
                trace_stage_key=trace_stage_key,
            )
            result = await scenario.execute()
            self._save_result(case_id, display_stage_code, result or {})
            if display_stage_code == "PLC":
                self._save_result(case_id, "LC", result or {})
            if trace_recorder is not None:
                trace_recorder.export_stage(
                    stage_code=display_stage_code,
                    stage_key=trace_stage_key,
                    agents=[client, lawyer],
                    stage_result=result or {},
                    stage_result_path=case_output_dir / f"{display_stage_code}_result.json",
                    status="completed",
                )

            # Persist memory back into agent config before advancing to the drafting stage.
            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code=display_stage_code,
                stage_label=self.STAGE_DISPLAY_NAMES.get(display_stage_code, display_stage_code),
                agents=[lawyer, client],
            )

            # Mark scenario as completed in checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_scenario_completed(scenario_id)
            scenario_succeeded = True

        except Exception as e:
            logger.exception("[Orchestrator] LC scenario failed")
            if trace_recorder is not None:
                trace_recorder.export_stage(
                    stage_code=display_stage_code,
                    stage_key=trace_stage_key,
                    agents=[client, lawyer],
                    stage_result=None,
                    stage_result_path=case_output_dir / f"{display_stage_code}_result.json",
                    status="failed",
                    error=repr(e),
                )
            if lawyer.is_active:
                lawyer.recover_from_error()
            if client.is_active:
                client.recover_from_error()
            reported = await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="LC",
                exc=e,
                stage_label="法律咨询",
            )
            if reported:
                return
            logger.error(
                "[Orchestrator] LC runtime issue was not escalated to sandbox error state: case=%s reporter=%s",
                case_id,
                callable(getattr(self, "runtime_issue_reporter", None)),
            )
        finally:
            self._clear_case_stage_active(case_id)

            # Deactivate agents only if scenario completed successfully
            if lawyer.is_active:
                lawyer.deactivate()
            if client.is_active:
                client.deactivate()

        if not scenario_succeeded:
            return

        # Advance FSM
        completion_event = (
            EventType.PLAINTIFF_CONSULTATION_COMPLETED
            if party_role == "plaintiff"
            else EventType.DEFENDANT_CONSULTATION_COMPLETED
        )
        
        await self.event_bus.publish(completion_event, {
            "case_id": case_id,
            "client_path": client_path,
            "client_id": client.agent_id,
            "lawyer_id": lawyer_id,
            "party_role": party_role,
            "firm_id": getattr(lawyer, "firm_id", "law_firm_A"),
            "client_chair": payload.get("client_chair", ""),
            "lawyer_chair": payload.get("lawyer_chair", ""),
        })

    # ══════════════════════════════════════════════════════════
    #  Movement Choreography (map_engine driven)
    # ══════════════════════════════════════════════════════════

    async def _choreograph_case_assigned(self, payload: dict) -> None:
        """CASE_ASSIGNED: 根据律师状态决定流程（空闲→直接咨询，繁忙→沙发等待）。"""
        client_id = payload.get("client_id", "")
        lawyer_id = payload.get("lawyer_id", "")
        case_id = payload.get("case_id", "")
        target_firm = payload.get("map_prefix") or payload.get("target_firm", "lawfirmA")
        party_role = payload.get("party_role", "plaintiff")

        lawyer = self.registry.get_agent(lawyer_id)
        client = self.registry.get_agent(client_id)
        if not lawyer or not client:
            return

        logger.info("[Choreography] CASE_ASSIGNED: %s + %s @ %s", client.name, lawyer.name, target_firm)

        # 判断律师是否空闲（通过 EventBus 查询）
        if self._is_agent_available(lawyer_id):
            logger.info(f"[Choreography] 律师 {lawyer.name} 空闲，立即开始咨询")
            await self._start_consultation_immediately(
                client, lawyer, case_id, target_firm, party_role, payload
            )
        else:
            logger.info(f"[Choreography] 律师 {lawyer.name} 繁忙，当事人 {client.name} 移动到沙发等待")
            await self._move_client_to_waiting_area(
                client, case_id, target_firm, party_role
            )

    async def _start_consultation_immediately(
        self, client, lawyer, case_id, firm_id, party_role, payload
    ) -> None:
        """律师空闲：当事人和律师就座，开始咨询。"""

        # 关键修复：将律师 ID 写入当事人配置
        if client.config_path:
            try:
                self.storage.update_agent_field(client.config_path, "assigned_lawyer_id", lawyer.agent_id)
                logger.info(f"[Orchestrator] 已将律师 {lawyer.agent_id} 分配给当事人 {client.name}")
            except Exception as e:
                logger.error(f"[Orchestrator] 更新 assigned_lawyer_id 失败: {e}")

        # 获取空闲会议椅对
        client_chair, lawyer_chair = self._get_available_chair_pair(firm_id)
        if not client_chair or not lawyer_chair:
            logger.error(f"[Choreography] 无可用会议椅: {firm_id}，将当事人加入等候队列")
            await self._move_client_to_waiting_area(client, case_id, firm_id, party_role)
            return

        # 占用椅子
        client_reserved = self._occupy_location(client_chair, client.agent_id)
        lawyer_reserved = self._occupy_location(lawyer_chair, lawyer.agent_id)
        if not client_reserved or not lawyer_reserved:
            if client_reserved:
                self._release_location(client_chair)
            if lawyer_reserved:
                self._release_location(lawyer_chair)
            logger.error(f"[Choreography] 会议椅占用失败: {client_chair} / {lawyer_chair}")
            await self._move_client_to_waiting_area(client, case_id, firm_id, party_role)
            return

        await lawyer.start_handling_case(case_id)

        if self.map_engine:
            if lawyer.agent_id not in self.map_engine._agent_states:
                logger.info(f"[Choreography] {lawyer.name} 出生并准备前往椅子 {lawyer_chair}")
                await self.map_engine.spawn_agent(
                    agent_id=lawyer.agent_id,
                    name=lawyer.name,
                    character_name=self._get_character_name_for_lawyer(lawyer),
                    birth_loc_id=self._get_birth_location_for_agent(lawyer.agent_id),
                    role="lawyer",
                )
            else:
                logger.info(f"[Choreography] {lawyer.name} 已存在，准备前往椅子 {lawyer_chair}")

            await asyncio.gather(
                self.map_engine.stand_agent(client.agent_id),
                self.map_engine.stand_agent(lawyer.agent_id),
            )
            logger.info(
                f"[Choreography] {client.name} 与 {lawyer.name} 并行前往会议椅 "
                f"{client_chair} / {lawyer_chair}"
            )
            await asyncio.gather(
                self.map_engine.move_to_location(client.agent_id, client_chair),
                self.map_engine.move_to_location(lawyer.agent_id, lawyer_chair),
            )
            await asyncio.gather(
                self.map_engine.sit_agent(client.agent_id, client_chair),
                self.map_engine.sit_agent(lawyer.agent_id, lawyer_chair),
            )
            logger.info(f"[Choreography] ✓ 双方已就座，准备开始咨询")

        # 发布咨询事件
        from ..core.event_bus import EventType
        consultation_event = (
            EventType.ENTER_PLAINTIFF_CONSULTATION
            if party_role == "plaintiff"
            else EventType.ENTER_DEFENDANT_CONSULTATION
        )

        await self.event_bus.publish(consultation_event, {
            **payload,
            "client_chair": client_chair,
            "lawyer_chair": lawyer_chair,
        })

    async def _move_client_to_waiting_area(
        self, client, case_id, firm_id, party_role: str = "plaintiff"
    ) -> None:
        """律师繁忙：当事人移动到沙发等待。"""

        # 获取空闲沙发
        sofa_id = self._get_available_sofa(firm_id)
        if sofa_id:
            # 占用沙发
            if not self._occupy_location(sofa_id, client.agent_id):
                logger.error(f"[Choreography] 沙发占用失败: {sofa_id}，当事人 {client.name} 继续等待")
                return

            if self.map_engine:
                logger.info(f"[Choreography] {client.name} 移动到沙发 {sofa_id}")
                await self.map_engine.stand_agent(client.agent_id)
                await self.map_engine.move_to_location(client.agent_id, sofa_id)
                sofa_direction = "left" if str(firm_id).lower().endswith("b") else None
                await self.map_engine.sit_agent(client.agent_id, sofa_id, direction_override=sofa_direction)

            await self._add_to_waiting_queue(
                firm_id,
                client.agent_id,
                case_id,
                sofa_id,
                party_role,
                wait_mode="sofa",
            )
            return

        wait_spot_id, wait_spot = self._get_available_waiting_spot(firm_id)
        if not wait_spot_id or not wait_spot:
            logger.error(f"[Choreography] 无可用沙发或站立候位点: {firm_id}，当事人 {client.name} 原地等待")
            return

        if not self._occupy_location(wait_spot_id, client.agent_id):
            logger.error(f"[Choreography] 候位点占用失败: {wait_spot_id}，当事人 {client.name} 继续等待")
            return

        if self.map_engine:
            logger.info(f"[Choreography] {client.name} 移动到站立候位点 {wait_spot_id}")
            await self.map_engine.stand_agent(client.agent_id)
            await self.map_engine.move_to_location(client.agent_id, wait_spot_id)
            await self.map_engine.stand_agent(
                client.agent_id,
                direction_override=getattr(wait_spot, "direction", "") or "down",
            )

        await self._add_to_waiting_queue(
            firm_id,
            client.agent_id,
            case_id,
            wait_spot_id,
            party_role,
            wait_mode="standing_queue",
        )

    async def _ensure_agent_spawned(
        self,
        agent_id: str,
        name: str,
        character_name: str,
        birth_loc_id: str,
        role: str,
    ) -> None:
        if not self.map_engine:
            return
        if agent_id in self.map_engine._agent_states:
            return
        await self.map_engine.spawn_agent(
            agent_id=agent_id,
            name=name,
            character_name=character_name,
            birth_loc_id=birth_loc_id,
            role=role,
        )

    async def _seat_client_and_lawyer_for_drafting(
        self,
        client: Any,
        lawyer: Any,
        map_prefix: str,
        client_role: str,
    ) -> tuple[str, str]:
        if not self.map_engine:
            return "", ""

        client_chair, lawyer_chair = self._get_available_chair_pair(map_prefix)
        if not client_chair or not lawyer_chair:
            logger.error("[Choreography] %s 无可用会议椅，无法进入文书起草子场景", map_prefix)
            return "", ""

        client_reserved = self._occupy_location(client_chair, client.agent_id)
        lawyer_reserved = self._occupy_location(lawyer_chair, lawyer.agent_id)
        if not client_reserved or not lawyer_reserved:
            if client_reserved:
                self._release_location(client_chair)
            if lawyer_reserved:
                self._release_location(lawyer_chair)
            logger.error("[Choreography] %s 会议椅占用失败，无法进入文书起草子场景", map_prefix)
            return "", ""

        birth_loc_id = self._resolve_birth_location_for_map_prefix(map_prefix)
        await self._ensure_agent_spawned(
            agent_id=client.agent_id,
            name=client.name,
            character_name=self._get_character_name_for_client(client, client_role),
            birth_loc_id=birth_loc_id,
            role=client_role,
        )
        await self._ensure_agent_spawned(
            agent_id=lawyer.agent_id,
            name=lawyer.name,
            character_name=self._get_character_name_for_lawyer(lawyer),
            birth_loc_id=birth_loc_id,
            role="lawyer",
        )

        logger.info(
            "[Choreography] %s 与 %s 并行前往会议椅 %s / %s",
            client.name,
            lawyer.name,
            client_chair,
            lawyer_chair,
        )
        await asyncio.gather(
            self.map_engine.stand_agent(client.agent_id),
            self.map_engine.stand_agent(lawyer.agent_id),
        )
        await asyncio.gather(
            self.map_engine.move_to_location(client.agent_id, client_chair),
            self.map_engine.move_to_location(lawyer.agent_id, lawyer_chair),
        )
        await asyncio.gather(
            self.map_engine.sit_agent(client.agent_id, client_chair),
            self.map_engine.sit_agent(lawyer.agent_id, lawyer_chair),
        )
        return client_chair, lawyer_chair

    async def _return_agents_to_birth_and_despawn(self, agent_ids: list[str]) -> None:
        if not self.map_engine:
            return

        live_agent_ids: list[str] = []
        seen: set[str] = set()
        for agent_id in agent_ids:
            if not agent_id or agent_id in seen:
                continue
            if agent_id not in self.map_engine._agent_states:
                continue
            seen.add(agent_id)
            live_agent_ids.append(agent_id)

        if not live_agent_ids:
            return

        for agent_id in live_agent_ids:
            await self.map_engine.stand_agent(agent_id)

        move_tasks = [
            self.map_engine.move_to_location(agent_id, self._get_birth_location_for_agent(agent_id))
            for agent_id in live_agent_ids
        ]
        if move_tasks:
            await asyncio.gather(*move_tasks)

        for agent_id in live_agent_ids:
            await self.map_engine.despawn_agent(agent_id)

    async def _file_document_and_despawn(
        self,
        lawyer: Any,
        court_entrance: str,
        birth_loc_id: str,
        document_name: str,
    ) -> None:
        if not self.map_engine or not lawyer:
            return

        await self._ensure_agent_spawned(
            agent_id=lawyer.agent_id,
            name=lawyer.name,
            character_name=self._get_character_name_for_lawyer(lawyer),
            birth_loc_id=birth_loc_id,
            role="lawyer",
        )

        court_loc = self.map_engine.registry.get(court_entrance)
        if not court_loc:
            logger.warning("[Choreography] 法院入口 %s 不存在，跳过%s递交动作", court_entrance, document_name)
        else:
            await self.map_engine.move_to_location(lawyer.agent_id, court_entrance)
            logger.info("[Choreography] %s 正在递交%s", lawyer.name, document_name)
            await self.map_engine.play_animation(lawyer.agent_id, "typing", 3.0)

        await self.map_engine.move_to_location(lawyer.agent_id, birth_loc_id)
        await self.map_engine.despawn_agent(lawyer.agent_id)

    async def _choreograph_first_trial(self, payload: dict) -> None:
        """ENTER_TRIAL_FIRST_INSTANCE: all participants move to courtA."""
        try:
            await self._choreograph_trial(payload, "courtA")

            # 移动完成后，发布 READY 事件以触发实际庭审场景
            await self.event_bus.publish(EventType.TRIAL_FIRST_INSTANCE_READY, payload)
        except Exception:
            await self._release_trial_slot("courtA", payload.get("case_id", ""))
            raise

    async def _choreograph_second_trial(self, payload: dict) -> None:
        """ENTER_TRIAL_SECOND_INSTANCE: all participants move to courtB."""
        try:
            await self._choreograph_trial(payload, "courtB")

            # 移动完成后，发布 READY 事件以触发实际庭审场景
            await self.event_bus.publish(EventType.TRIAL_SECOND_INSTANCE_READY, payload)
        except Exception:
            await self._release_trial_slot("courtB", payload.get("case_id", ""))
            raise

    async def _choreograph_trial(self, payload: dict, court: str) -> None:
        """Move all trial participants to court positions."""
        if not self.map_engine:
            return

        case_id = payload.get("case_id", "")
        logger.info(f"[Choreography] Trial @ {court} for case {case_id}")

        # Collect participant IDs from payload
        participants = {
            "judge": payload.get("judge_id"),
            "plaintiff_lawyer": payload.get("plaintiff_lawyer_id"),
            "defendant_lawyer": payload.get("defendant_lawyer_id"),
            "plaintiff": payload.get("plaintiff_id"),
            "defendant": payload.get("defendant_id"),
        }

        # Stand everyone who's currently sitting
        for role, agent_id in participants.items():
            if agent_id:
                await self.map_engine.stand_agent(agent_id)

        # Spawn judge and other participants if not already on map
        for role, agent_id in participants.items():
            if not agent_id:
                continue
            
            # If not on map, spawn them
            if agent_id not in self.map_engine._agent_states:
                agent = self.registry.get_agent(agent_id)
                if agent:
                    logger.info(f"[Choreography] Spawning missing participant for trial: {agent.name} ({role})")
                    if role in {"plaintiff", "defendant"}:
                        character_name = self._get_character_name_for_client(agent, role)
                    elif "lawyer" in role:
                        character_name = self._get_character_name_for_lawyer(agent)
                    else:
                        character_name = self._get_or_assign_character_name(agent)

                    await self.map_engine.spawn_agent(
                        agent_id=agent_id,
                        name=agent.name,
                        character_name=character_name,
                        birth_loc_id=self._get_birth_location_for_agent(agent_id),
                        role=role,
                    )

        # Map role → court chair location ID
        role_to_loc = {
            "judge": f"{court}_judgeA",
            "plaintiff_lawyer": f"{court}_plaintiff_lawyer",
            "defendant_lawyer": f"{court}_defendant_lawyer",
            "plaintiff": f"{court}_plaintiff",
            "defendant": f"{court}_defendant",
        }
        role_to_seat_direction = {
            "plaintiff_lawyer": "right",
            "plaintiff": "right",
            "defendant_lawyer": "left",
            "defendant": "left",
        }

        # Move all participants to their court positions in parallel
        move_tasks = []
        for role, agent_id in participants.items():
            if agent_id and role in role_to_loc:
                loc_id = role_to_loc[role]
                move_tasks.append(self.map_engine.move_to_location(agent_id, loc_id))

        if move_tasks:
            await asyncio.gather(*move_tasks)

        judge_id = participants.get("judge")
        if judge_id:
            await self._stand_agent_on_location(
                judge_id,
                role_to_loc["judge"],
                direction="down",
                y_offset=-16.0,
            )

        # Only seat the two parties and their lawyers. The judge remains standing.
        for role, agent_id in participants.items():
            if agent_id and role in role_to_seat_direction:
                await self.map_engine.sit_agent(
                    agent_id,
                    role_to_loc[role],
                    direction_override=role_to_seat_direction[role],
                )

    async def _choreograph_case_closed(self, payload: dict) -> None:
        """CASE_CLOSED: everyone stands, moves to birth point, despawns."""
        if not self.map_engine:
            return

        case_id = payload.get("case_id", "")
        agent_ids = payload.get("participant_ids", [])
        logger.info(f"[Choreography] CASE_CLOSED: {case_id}, despawning {len(agent_ids)} agents")
        await self._return_agents_to_birth_and_despawn(agent_ids)

    # ══════════════════════════════════════════════════════════
    #  新增场景处理方法
    # ══════════════════════════════════════════════════════════

    async def _on_first_instance_verdict(self, payload: dict) -> None:
        """TRIAL_FIRST_INSTANCE_COMPLETED: 一审庭审结束，发布判决书下达事件。"""
        from ..core.event_bus import EventType
        await self.event_bus.publish(EventType.FIRST_INSTANCE_VERDICT_ISSUED, payload)

    async def _start_appeal_decision(self, payload: dict) -> None:
        """FIRST_INSTANCE_VERDICT_ISSUED: 一审判决下达，当事人决定是否上诉。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        trial_slot_released = False
        participant_ids = payload.get("participant_ids") or self._collect_case_participant_ids(
            case_id,
            extra_ids=[payload.get("judge_id", "")],
        )
        parties = self._get_case_parties(case_id)
        plaintiff_path = parties["plaintiff"].get("client_path") or payload.get("client_path", "")

        logger.info(f"[Orchestrator] 根据数据集判断上诉方: {case_id}")

        if not plaintiff_path:
            logger.error("[Orchestrator] 无法定位案件配置，无法执行上诉判定: %s", case_id)
            if case_id:
                await self._release_trial_slot("courtA", case_id)
            return

        try:
            _data_loader, case, _ = self._load_case_data(plaintiff_path)
            appeal_roles = self._resolve_appeal_roles(case)
            appellant_role = appeal_roles.get("appellant_role", "")
            appellee_role = appeal_roles.get("appellee_role", "")
            will_appeal = bool(appellant_role)

            if self.map_engine:
                logger.info("[Choreography] 一审结束后，所有角色返回出生点并消失: %s", case_id)
                await self._return_agents_to_birth_and_despawn(participant_ids)

            await self._release_trial_slot("courtA", case_id)
            trial_slot_released = True

            appellant_bundle = parties.get(appellant_role, {})
            appellee_bundle = parties.get(appellee_role, {})
            appellant_lawyer = appellant_bundle.get("lawyer")
            appellee_lawyer = appellee_bundle.get("lawyer")

            decision_payload = {
                "case_id": case_id,
                "client_path": appellant_bundle.get("client_path") or plaintiff_path,
                "will_appeal": will_appeal,
                "participant_ids": participant_ids,
                "appellant_role": appellant_role,
                "appellee_role": appellee_role,
                "appellant_name": appeal_roles.get("appellant_name", ""),
                "appellee_name": appeal_roles.get("appellee_name", ""),
                "appellant_client_id": getattr(appellant_bundle.get("client"), "agent_id", ""),
                "appellant_client_path": appellant_bundle.get("client_path", ""),
                "appellant_lawyer_id": appellant_bundle.get("lawyer_id", ""),
                "appellant_map_prefix": self._resolve_map_prefix_from_lawyer(appellant_lawyer) if appellant_lawyer else "lawfirmA",
                "appellee_client_id": getattr(appellee_bundle.get("client"), "agent_id", ""),
                "appellee_client_path": appellee_bundle.get("client_path", ""),
                "appellee_lawyer_id": appellee_bundle.get("lawyer_id", ""),
                "appellee_map_prefix": self._resolve_map_prefix_from_lawyer(appellee_lawyer) if appellee_lawyer else "lawfirmB",
                "plaintiff_id": getattr(parties["plaintiff"].get("client"), "agent_id", ""),
                "defendant_id": getattr(parties["defendant"].get("client"), "agent_id", ""),
                "plaintiff_lawyer_id": parties["plaintiff"].get("lawyer_id", ""),
                "defendant_lawyer_id": parties["defendant"].get("lawyer_id", ""),
            }
            if self._player_plaintiff_lawyer_enabled() and will_appeal and appellee_role == "plaintiff":
                decision_payload["player_current_appeal_stage"] = "AR"
                decision_payload["client_path"] = appellee_bundle.get("client_path") or plaintiff_path

            logger.info(
                "[Orchestrator] 上诉判定结果: will_appeal=%s, appellant=%s(%s)",
                will_appeal,
                appeal_roles.get("appellant_name", ""),
                appellant_role,
            )
            await self.event_bus.publish(EventType.APPEAL_DECISION_MADE, decision_payload)
        finally:
            if not trial_slot_released and case_id:
                await self._release_trial_slot("courtA", case_id)

    async def _handle_appeal_decision(self, payload: dict) -> None:
        """APPEAL_DECISION_MADE: 处理上诉决策结果。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        will_appeal = payload.get("will_appeal", False)
        participant_ids = payload.get("participant_ids") or self._collect_case_participant_ids(case_id)

        if will_appeal:
            appellee_role = payload.get("appellee_role", "")
            if self._player_plaintiff_lawyer_enabled() and appellee_role == "plaintiff":
                payload["player_current_appeal_stage"] = "AR"
                logger.info(f"[Orchestrator] 玩家原告为被上诉人，跳过对手上诉状起草展示并进入上诉答辩: {case_id}")
                await self._run_appeal_response_drafting(payload)
                return
            logger.info(f"[Orchestrator] 当事人决定上诉，开始起草上诉状: {case_id}")
            # 开始上诉状起草
            await self._run_appeal_drafting(payload)
        else:
            logger.info(f"[Orchestrator] 当事人服判，案件结案: {case_id}")
            # 直接结案
            await self.event_bus.publish(EventType.CASE_CLOSED, {
                "case_id": case_id,
                "client_path": client_path,
                "participant_ids": participant_ids,
            })

    async def _run_appeal_drafting(self, payload: dict) -> None:
        """运行上诉状起草场景。"""
        from ..scenarios.appeal_drafting import AppealDraftingScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("appellant_client_path") or payload.get("client_path", "")
        appellant_role = payload.get("appellant_role", "plaintiff")
        client_id = payload.get("appellant_client_id", "")
        lawyer_id = payload.get("appellant_lawyer_id", "")
        map_prefix = payload.get("appellant_map_prefix", "lawfirmA")

        logger.info(f"[Orchestrator] 开始上诉状起草: {case_id}")

        # 获取上诉人与其律师
        client = self.registry.get_agent(client_id) if client_id else None
        if not client:
            client, _ = self._find_client_for_case(case_id, party_role=appellant_role or "plaintiff")
        if not client or not client_path:
            logger.error(f"[Orchestrator] 未找到上诉人: {case_id}")
            return

        config = self.storage.load_agent_config(client_path)
        if not lawyer_id:
            lawyer_id = config.get("assigned_lawyer_id", "")
        lawyer = self.registry.get_agent(lawyer_id)

        if not lawyer:
            logger.error(f"[Orchestrator] 未找到律师: {lawyer_id}")
            return

        client_chair = ""
        lawyer_chair = ""
        birth_loc_id = self._resolve_birth_location_for_map_prefix(map_prefix)
        if self.map_engine:
            logger.info("[Choreography] 上诉人 %s 与律师 %s 从社区前往 %s", client.name, lawyer.name, map_prefix)
            client_chair, lawyer_chair = await self._seat_client_and_lawyer_for_drafting(
                client=client,
                lawyer=lawyer,
                map_prefix=map_prefix,
                client_role=appellant_role or "plaintiff",
            )

        if (
            self._player_plaintiff_lawyer_enabled()
            and appellant_role == "plaintiff"
            and not self._player_ai_surrogate_enabled()
        ):
            await self._run_player_appellate_document_drafting(
                case_id=case_id,
                client_path=client_path,
                client=client,
                lawyer=lawyer,
                lawyer_id=lawyer_id,
                firm_id=getattr(lawyer, "firm_id", "law_firm_A"),
                client_chair=client_chair,
                lawyer_chair=lawyer_chair,
                stage="AD",
                document_type="appeal",
                document_label="上诉状",
                prompt="请以原告/上诉人律师身份完成《民事上诉状》。你可以调用后端文书辅助接口生成草稿，审核修改后确认文书。",
                completion_event=EventType.APPEAL_DRAFTING_COMPLETED,
                completion_payload={**payload, "case_id": case_id, "client_path": client_path, "lawyer_id": lawyer_id},
            )
            return

        # 加载一审判决书
        case_output_dir = self._get_case_output_dir(case_id)
        verdict_path = case_output_dir / "CI_result.json"
        first_instance_verdict = ""
        if verdict_path.exists():
            with open(verdict_path, "r", encoding="utf-8") as f:
                verdict_data = json.load(f)
                first_instance_verdict = verdict_data.get("final_judgment", "")

        # 构建场景数据
        data_loader, case, config = self._load_case_data(client_path)
        appellant_info = data_loader.extract_appellant_appeal(case)
        appeal_claims = appellant_info.get("claim", [])
        appeal_reasons = appellant_info.get("reasons", "")
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        appellee_role = payload.get("appellee_role") or ("defendant" if appellant_role == "plaintiff" else "plaintiff")
        appellant_profile = plaintiff_profile if appellant_role == "plaintiff" else defendant_profile
        appellee_profile = defendant_profile if appellee_role == "defendant" else plaintiff_profile
        client_profile = appellant_profile
        appellant_new_evidence = data_loader.extract_second_instance_evidence(case, side="appellant")
        
        scenario_data = {
            "first_instance_judgment": first_instance_verdict,
            "appellant_name": appellant_profile.get("name", "") or getattr(client, "name", ""),
            "appellant_gender": appellant_profile.get("gender", "") or getattr(client, "gender", ""),
            "appellant_birth_date": appellant_profile.get("birth_date", ""),
            "appellant_ethnicity": appellant_profile.get("ethnicity", ""),
            "appellant_address": appellant_profile.get("address", ""),
            "appellant_representative": appellant_profile.get("representative", ""),
            "appellee_name": appellee_profile.get("name", ""),
            "appellee_gender": appellee_profile.get("gender", ""),
            "appellee_birth_date": appellee_profile.get("birth_date", ""),
            "appellee_ethnicity": appellee_profile.get("ethnicity", ""),
            "appellee_address": appellee_profile.get("address", ""),
            "appellee_representative": appellee_profile.get("representative", ""),
            "case_background": data_loader.extract_case_background(case),
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "case_cause": data_loader.extract_case_cause(case),
            "appeal_claims": "\n".join(f"{i+1}. {c}" for i, c in enumerate(appeal_claims)) if isinstance(appeal_claims, list) else appeal_claims,
            "appeal_reasons": appeal_reasons,
            "new_evidence": appellant_new_evidence or "（暂无新证据）",
        }

        # 激活 Agents (使用 PromptAssembler)
        client_scenario = PromptAssembler.build_scenario_prompt("client", "AD", scenario_data)
        client_memory = self._get_client_prompt_memory(client, case_id)
        client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(client, client_profile),
            long_term_memory=client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=client_scenario,
        )
        scenario_data["case_output_dir"] = str(case_output_dir.resolve())
        client.scenario_data = scenario_data
        client.activate(client_prompt)

        lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "AD", scenario_data)
        lawyer_config = self.storage.load_agent_config(lawyer.config_path) if lawyer and lawyer.config_path else {}
        lawyer_memory = self._get_lawyer_prompt_memory(lawyer, case_id)
        lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(lawyer),
            long_term_memory=lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=lawyer_scenario,
        )
        lawyer.scenario_type = "AD"
        lawyer.scenario_data = dict(scenario_data)
        lawyer.activate(lawyer_prompt)
        self._configure_stage_tools(
            "AD",
            {
                "appellant": client,
                "lawyer": lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "AD",
            "AD",
            [client, lawyer],
        )
        self._collect_stage_prompts(case_id, "AD", client, lawyer, reset=True)
        self._mark_case_stage_active(case_id, "AD", [client.agent_id, lawyer.agent_id])
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="AD",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="AD",
            case_cause=scenario_data.get("case_cause", ""),
            case_background=scenario_data.get("case_background", ""),
            trace_recorder=trace_recorder,
        )

        # 执行场景
        scenario_succeeded = False
        try:
            output_path = str(case_output_dir / "AD_result.json")
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: AppealDraftingScenario(
                    appellant_agent=client,
                    lawyer_agent=lawyer,
                    max_turns=self._resolve_stage_max_turns("AD", 20),
                    output_path=output_path,
                    verbose=False,
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "appellant": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                trace_recorder=trace_recorder,
                trace_stage_code="AD",
                trace_stage_key="AD",
                trace_agents=[client, lawyer],
                trace_result_path=output_path,
            )
            self._save_result(case_id, "AD", result or {})
            appeal_statement = str((result or {}).get("appeal_statement") or "")
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="AD",
                document_text=appeal_statement,
                compare_left=first_instance_verdict,
                compare_right=appeal_statement,
                compare_labels=("first_instance_judgment", "appeal"),
                trace_recorder=trace_recorder,
            )
            if self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase="ad_document_received",
                    message="上诉状已生成，系统正在登记文书。",
                    detail="登记完成后会递交中级法院并通知被上诉人。",
                    blocking=False,
                    metadata={"stage": "AD", "scenario_type": "AD"},
                )
            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code="AD",
                stage_label=self.STAGE_DISPLAY_NAMES.get("AD", "AD"),
                agents=[lawyer],
            )
            trace_recorder.export_stage(
                stage_code="AD",
                stage_key="AD",
                agents=[client, lawyer],
                stage_result=result or {},
                stage_result_path=output_path,
                status="completed",
            )
            scenario_succeeded = True

        except Exception as e:
            logger.error(f"[Orchestrator] AD scenario failed: {e}")
            # Recover agents instead of deactivating
            client.recover_from_error()
            lawyer.recover_from_error()
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="AD",
                exc=e,
                stage_label="上诉状起草",
            ):
                return
        finally:
            self._clear_case_stage_active(case_id)
            # Deactivate only if still active
            if client.is_active:
                client.deactivate()
            if lawyer.is_active:
                lawyer.deactivate()

        if not scenario_succeeded:
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)
            if self.map_engine:
                await self._return_agents_to_birth_and_despawn([client.agent_id, lawyer.agent_id])
            return

        if self.map_engine:
            await self.map_engine.stand_agent(lawyer.agent_id)
            await self.map_engine.stand_agent(client.agent_id)
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)

            logger.info("[Choreography] 上诉人 %s 返回社区出生点并消失", client.name)
            await self.map_engine.move_to_location(client.agent_id, birth_loc_id)
            await self.map_engine.despawn_agent(client.agent_id)

            logger.info("[Choreography] 上诉人律师 %s 前往中级人民法院递交上诉状", lawyer.name)
            await self._file_document_and_despawn(
                lawyer=lawyer,
                court_entrance="courtB_entrance",
                birth_loc_id=birth_loc_id,
                document_name="上诉状",
            )

        # 发布完成事件
        await self.event_bus.publish(EventType.APPEAL_DRAFTING_COMPLETED, {
            **payload,
            "case_id": case_id,
            "client_path": client_path,
            "lawyer_id": lawyer_id,
        })

    async def _on_appeal_filed(self, payload: dict) -> None:
        """APPEAL_DRAFTING_COMPLETED: 上诉状起草完成，发布递交事件。"""
        from ..core.event_bus import EventType
        case_id = payload.get("case_id", "")
        appellant_role = payload.get("appellant_role", "plaintiff")
        if self._player_plaintiff_lawyer_enabled() and appellant_role == "plaintiff":
            if case_id and self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase="appeal_filing",
                    message="上诉状正在递交中级法院。",
                    detail="玩家原告流程不展示对手上诉答辩文书，递交完成后将直接准备二审开庭。",
                    blocking=False,
                    metadata={"stage": "AD", "scenario_type": "AD"},
                )
            await self._prepare_player_second_instance_trial_after_document(payload)
            return
        if case_id and self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
            await self.map_engine.broadcast_runtime_progress(
                case_id,
                phase="appeal_filing",
                message="上诉状正在递交中级法院。",
                detail="递交完成后会通知被上诉人准备上诉答辩。",
                blocking=False,
                metadata={"stage": "AD", "scenario_type": "AD"},
        )
        await self.event_bus.publish(EventType.APPEAL_FILED, payload)

    async def _prepare_player_second_instance_trial_after_document(self, payload: dict) -> None:
        """Player plaintiff mode skips the opponent appeal document stage and schedules CIA."""
        from .case_fsm import CaseState
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        if not client_path:
            parties = self._get_case_parties(case_id)
            client_path = parties["plaintiff"].get("client_path", "")
        if not client_path:
            logger.error("[Orchestrator] 玩家原告流程无法定位二审案件配置: case=%s", case_id)
            return

        data_loader, case, _ = self._load_case_data(client_path)
        appeal_roles = self._resolve_appeal_roles(case)
        appellant_role = payload.get("appellant_role") or appeal_roles.get("appellant_role", "plaintiff")
        appellee_role = payload.get("appellee_role") or appeal_roles.get("appellee_role", "defendant")
        parties = self._get_case_parties(case_id)
        plaintiff_lawyer_id = parties["plaintiff"].get("lawyer_id", "")
        if not parties["defendant"].get("lawyer_id"):
            self._ensure_player_trial_opponent_bundle(
                {
                    **payload,
                    "case_id": case_id,
                    "client_path": parties["plaintiff"].get("client_path") or client_path,
                    "plaintiff_lawyer_id": plaintiff_lawyer_id,
                }
            )
            parties = self._get_case_parties(case_id)

        appellant_bundle = parties.get(appellant_role, {})
        appellee_bundle = parties.get(appellee_role, {})
        appellant_lawyer_id = appellant_bundle.get("lawyer_id", "")
        appellee_lawyer_id = appellee_bundle.get("lawyer_id", "")
        if not appellant_lawyer_id or not appellee_lawyer_id:
            logger.error(
                "[Orchestrator] 玩家原告流程二审参与律师不完整: case=%s appellant_lawyer=%s appellee_lawyer=%s",
                case_id,
                appellant_lawyer_id,
                appellee_lawyer_id,
            )
            return

        self._set_shared_case_state(case_id, CaseState.WAITING_FOR_SECOND_TRIAL)
        await self._schedule_trial_entry(
            case_id=case_id,
            court="courtB",
            court_level="intermediate",
            event_type=EventType.ENTER_TRIAL_SECOND_INSTANCE,
            payload={
                **payload,
                "case_id": case_id,
                "client_path": client_path,
                "appellant_role": appellant_role,
                "appellee_role": appellee_role,
                "appellant_client_id": getattr(appellant_bundle.get("client"), "agent_id", ""),
                "appellee_client_id": getattr(appellee_bundle.get("client"), "agent_id", ""),
                "appellant_client_path": appellant_bundle.get("client_path", ""),
                "appellee_client_path": appellee_bundle.get("client_path", ""),
                "appellant_lawyer_id": appellant_lawyer_id,
                "appellee_lawyer_id": appellee_lawyer_id,
                "plaintiff_id": getattr(parties["plaintiff"].get("client"), "agent_id", ""),
                "defendant_id": getattr(parties["defendant"].get("client"), "agent_id", ""),
                "plaintiff_lawyer_id": parties["plaintiff"].get("lawyer_id", ""),
                "defendant_lawyer_id": parties["defendant"].get("lawyer_id", ""),
            },
        )

    async def _activate_appeal_response(self, payload: dict) -> None:
        """APPEAL_FILED: 上诉状递交，激活被上诉人起草上诉答辩状。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")

        logger.info(f"[Orchestrator] 激活被上诉人起草上诉答辩状: {case_id}")
        if case_id and self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
            await self.map_engine.broadcast_runtime_progress(
                case_id,
                phase="appeal_response_preparing",
                message="上诉状已递交中级法院，正在通知被上诉人准备答辩。",
                detail="系统正在读取上诉状与一审判决，正在准备上诉答辩材料。",
                blocking=False,
                metadata={"stage": "AR", "scenario_type": "AR"},
            )

        # 开始上诉答辩状起草
        await self._run_appeal_response_drafting(payload)

    async def _run_appeal_response_drafting(self, payload: dict) -> None:
        """运行上诉答辩状起草场景。"""
        from ..scenarios.appeal_response_drafting import AppealResponseDraftingScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("appellee_client_path") or payload.get("client_path", "")
        appellee_role = payload.get("appellee_role", "defendant")
        client_id = payload.get("appellee_client_id", "")
        lawyer_id = payload.get("appellee_lawyer_id", "")
        map_prefix = payload.get("appellee_map_prefix", "lawfirmB")

        logger.info(f"[Orchestrator] 开始上诉答辩状起草: {case_id}")

        client = self.registry.get_agent(client_id) if client_id else None
        if not client:
            client, _ = self._find_client_for_case(case_id, party_role=appellee_role or "defendant")
        if not client or not client_path:
            logger.error(f"[Orchestrator] 未找到被上诉人: {case_id}")
            return

        config = self.storage.load_agent_config(client_path)
        if not lawyer_id:
            lawyer_id = config.get("assigned_lawyer_id", "")
        lawyer = self.registry.get_agent(lawyer_id)

        if not lawyer:
            logger.error(f"[Orchestrator] 未找到律师: {lawyer_id}")
            return

        client_chair = ""
        lawyer_chair = ""
        birth_loc_id = self._resolve_birth_location_for_map_prefix(map_prefix)
        if self.map_engine:
            logger.info("[Choreography] 被上诉人 %s 与律师 %s 从社区前往 %s", client.name, lawyer.name, map_prefix)
            client_chair, lawyer_chair = await self._seat_client_and_lawyer_for_drafting(
                client=client,
                lawyer=lawyer,
                map_prefix=map_prefix,
                client_role=appellee_role or "defendant",
            )
            if hasattr(self.map_engine, "broadcast_runtime_progress"):
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase="appeal_response_generating",
                    message="被上诉人与律师已就位，正在生成上诉答辩首轮内容。",
                    detail="系统会先展示被上诉人的答辩需求，再进入文书处理。",
                    blocking=False,
                    metadata={"stage": "AR", "scenario_type": "AR"},
                )

        if (
            self._player_plaintiff_lawyer_enabled()
            and appellee_role == "plaintiff"
            and not self._player_ai_surrogate_enabled()
        ):
            await self._run_player_appellate_document_drafting(
                case_id=case_id,
                client_path=client_path,
                client=client,
                lawyer=lawyer,
                lawyer_id=lawyer_id,
                firm_id=getattr(lawyer, "firm_id", "law_firm_A"),
                client_chair=client_chair,
                lawyer_chair=lawyer_chair,
                stage="AR",
                document_type="appeal_response",
                document_label="上诉答辩状",
                prompt="请以原告/被上诉人律师身份完成《民事上诉答辩状》。你可以调用后端文书辅助接口生成草稿，审核修改后确认文书。",
                completion_event=EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED,
                completion_payload={**payload, "case_id": case_id, "client_path": client_path, "lawyer_id": lawyer_id},
            )
            return

        if self._player_plaintiff_lawyer_enabled() and appellee_role == "plaintiff":
            await self._run_player_appellate_document_drafting(
                case_id=case_id,
                client_path=client_path,
                client=client,
                lawyer=lawyer,
                lawyer_id=lawyer_id,
                firm_id=getattr(lawyer, "firm_id", "law_firm_A"),
                client_chair=client_chair,
                lawyer_chair=lawyer_chair,
                stage="AR",
                document_type="appeal_response",
                document_label="上诉答辩状",
                prompt="请以原告/被上诉人律师身份完成《民事上诉答辩状》。你可以调用后端文书辅助接口生成草稿，审核修改后确认文书。",
                completion_event=EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED,
                completion_payload={**payload, "case_id": case_id, "client_path": client_path, "lawyer_id": lawyer_id},
            )
            return

        # 加载上诉状
        case_output_dir = self._get_case_output_dir(case_id)
        appeal_path = case_output_dir / "AD_result.json"
        appeal_statement = ""
        if appeal_path.exists():
            with open(appeal_path, "r", encoding="utf-8") as f:
                appeal_data = json.load(f)
                appeal_statement = resolve_stage_document_text(
                    appeal_data,
                    "appeal_statement",
                )
        appeal_fields = extract_appeal_prompt_fields(appeal_statement)

        # 构建场景数据
        data_loader, case, config = self._load_case_data(client_path)
        appellant_info = data_loader.extract_appellant_appeal(case)
        appeal_claims = appellant_info.get("claim", [])
        appeal_reasons = appellant_info.get("reasons", "")
        appellee_position = data_loader.extract_second_instance_appellee_defense(case)
        appellee_new_evidence = data_loader.extract_second_instance_evidence(case, side="appellee")
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        appellant_role = payload.get("appellant_role") or ("plaintiff" if appellee_role == "defendant" else "defendant")
        appellant_profile = plaintiff_profile if appellant_role == "plaintiff" else defendant_profile
        appellee_profile = defendant_profile if appellee_role == "defendant" else plaintiff_profile
        client_profile = appellee_profile
        party_identity_data = {
            "appellant_name": appellant_profile.get("name", ""),
            "appellant_gender": appellant_profile.get("gender", ""),
            "appellant_birth_date": appellant_profile.get("birth_date", ""),
            "appellant_ethnicity": appellant_profile.get("ethnicity", ""),
            "appellant_address": appellant_profile.get("address", ""),
            "appellant_representative": appellant_profile.get("representative", ""),
            "appellee_name": appellee_profile.get("name", "") or getattr(client, "name", ""),
            "appellee_gender": appellee_profile.get("gender", "") or getattr(client, "gender", ""),
            "appellee_birth_date": appellee_profile.get("birth_date", ""),
            "appellee_ethnicity": appellee_profile.get("ethnicity", ""),
            "appellee_address": appellee_profile.get("address", ""),
            "appellee_representative": appellee_profile.get("representative", ""),
        }

        client_scenario_data = {
            **party_identity_data,
            "first_instance_judgment": "",
            "case_background": data_loader.extract_case_background(case),
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "case_cause": data_loader.extract_case_cause(case),
            "appeal_claims": appeal_fields.get("appeal_claims") or ("\n".join(f"{i+1}. {c}" for i, c in enumerate(appeal_claims)) if isinstance(appeal_claims, list) else appeal_claims),
            "appeal_reasons": appeal_fields.get("appeal_reasons") or appeal_reasons,
            "my_position": appellee_position,
            "new_evidence": appellee_new_evidence or "（暂无新证据）", 
            "appeal_statement": appeal_statement,
        }
        lawyer_scenario_data = {
            **party_identity_data,
            "first_instance_judgment": "",
            "case_background": data_loader.extract_case_background(case),
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "case_cause": data_loader.extract_case_cause(case),
            "appeal_claims": appeal_fields.get("appeal_claims") or ("\n".join(f"{i+1}. {c}" for i, c in enumerate(appeal_claims)) if isinstance(appeal_claims, list) else appeal_claims),
            "appeal_reasons": appeal_fields.get("appeal_reasons") or appeal_reasons,
            "my_position": appellee_position,
            "new_evidence": appellee_new_evidence or "（暂无新证据）",
            "appeal_statement": appeal_statement,
        }

        # 尝试加载一审判决
        verdict_path = self.output_dir / case_id / "CI_result.json"
        if verdict_path.exists():
            with open(verdict_path, "r", encoding="utf-8") as f:
                verdict_data = json.load(f)
                client_scenario_data["first_instance_judgment"] = verdict_data.get("final_judgment", "")
                lawyer_scenario_data["first_instance_judgment"] = verdict_data.get("final_judgment", "")

        # 激活 Agents (使用 PromptAssembler)
        client_scenario = PromptAssembler.build_scenario_prompt("client", "AR", client_scenario_data)
        client_memory = self._get_client_prompt_memory(client, case_id)
        client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(client, client_profile),
            long_term_memory=client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=client_scenario,
        )
        client_scenario_data["case_output_dir"] = str(case_output_dir.resolve())
        lawyer_scenario_data["case_output_dir"] = str(case_output_dir.resolve())
        client.scenario_data = client_scenario_data
        client.activate(client_prompt)

        lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "AR", lawyer_scenario_data)
        lawyer_config = self.storage.load_agent_config(lawyer.config_path) if lawyer and lawyer.config_path else {}
        lawyer_memory = self._get_lawyer_prompt_memory(lawyer, case_id)
        lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(lawyer),
            long_term_memory=lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=lawyer_scenario,
        )
        lawyer.scenario_type = "AR"
        lawyer.scenario_data = dict(lawyer_scenario_data)
        lawyer.activate(lawyer_prompt)
        self._configure_stage_tools(
            "AR",
            {
                "appellee": client,
                "lawyer": lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "AR",
            "AR",
            [client, lawyer],
        )
        self._collect_stage_prompts(case_id, "AR", client, lawyer, reset=True)
        self._mark_case_stage_active(case_id, "AR", [client.agent_id, lawyer.agent_id])
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="AR",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="AR",
            case_cause=lawyer_scenario_data.get("case_cause", ""),
            case_background=lawyer_scenario_data.get("case_background", ""),
            trace_recorder=trace_recorder,
        )

        # 执行场景
        scenario_succeeded = False
        try:
            output_path = str(case_output_dir / "AR_result.json")
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: AppealResponseDraftingScenario(
                    appellee_agent=client,
                    lawyer_agent=lawyer,
                    max_turns=self._resolve_stage_max_turns("AR", 20),
                    output_path=output_path,
                    verbose=SCENARIO_VERBOSE,
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "appellee": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                trace_recorder=trace_recorder,
                trace_stage_code="AR",
                trace_stage_key="AR",
                trace_agents=[client, lawyer],
                trace_result_path=output_path,
            )
            self._save_result(case_id, "AR", result or {})
            appeal_response_statement = str((result or {}).get("appeal_response_statement") or "")
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="AR",
                document_text=appeal_response_statement,
                compare_left=appeal_statement,
                compare_right=appeal_response_statement,
                compare_labels=("appeal", "appeal_response"),
                trace_recorder=trace_recorder,
            )
            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code="AR",
                stage_label=self.STAGE_DISPLAY_NAMES.get("AR", "AR"),
                agents=[lawyer],
            )
            trace_recorder.export_stage(
                stage_code="AR",
                stage_key="AR",
                agents=[client, lawyer],
                stage_result=result or {},
                stage_result_path=output_path,
                status="completed",
            )
            scenario_succeeded = True

        except Exception as e:
            logger.error(f"[Orchestrator] AR scenario failed: {e}")
            # Recover agents instead of deactivating
            client.recover_from_error()
            lawyer.recover_from_error()
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="AR",
                exc=e,
                stage_label="上诉答辩状起草",
            ):
                return
        finally:
            self._clear_case_stage_active(case_id)
            # Deactivate only if still active
            if client.is_active:
                client.deactivate()
            if lawyer.is_active:
                lawyer.deactivate()

        if not scenario_succeeded:
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)
            if self.map_engine:
                await self._return_agents_to_birth_and_despawn([client.agent_id, lawyer.agent_id])
            return

        if self.map_engine:
            await self.map_engine.stand_agent(lawyer.agent_id)
            await self.map_engine.stand_agent(client.agent_id)
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)

            logger.info("[Choreography] 被上诉人 %s 返回社区出生点并消失", client.name)
            await self.map_engine.move_to_location(client.agent_id, birth_loc_id)
            await self.map_engine.despawn_agent(client.agent_id)

            logger.info("[Choreography] 被上诉人律师 %s 前往中级人民法院递交上诉答辩状", lawyer.name)
            await self._file_document_and_despawn(
                lawyer=lawyer,
                court_entrance="courtB_entrance",
                birth_loc_id=birth_loc_id,
                document_name="上诉答辩状",
            )

        # 发布完成事件
        await self.event_bus.publish(EventType.APPEAL_RESPONSE_DRAFTING_COMPLETED, {
            **payload,
            "case_id": case_id,
            "client_path": client_path,
            "lawyer_id": lawyer_id,
        })

    async def _on_appeal_response_filed(self, payload: dict) -> None:
        """APPEAL_RESPONSE_DRAFTING_COMPLETED: 上诉答辩状起草完成，发布递交事件。"""
        from ..core.event_bus import EventType
        await self.event_bus.publish(EventType.APPEAL_RESPONSE_FILED, payload)

    async def _check_appeal_trial_ready(self, payload: dict) -> None:
        """APPEAL_RESPONSE_FILED: 二审文书完成并进入等待开庭状态后，触发二审开庭。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("appellant_client_path") or payload.get("client_path", "")

        logger.info(f"[Orchestrator] 二审开庭条件检查: {case_id}")

        if client_path:
            try:
                client_config = self.storage.load_agent_config(client_path)
                case_state = str(client_config.get("case_state", "") or "")
            except Exception as exc:
                logger.error("[Orchestrator] 无法读取二审案件状态: %s", exc)
                return
            if case_state not in {"等待二审开庭", "二审庭审中"}:
                logger.warning(
                    "[Orchestrator] 二审开庭前状态不足，跳过开庭: case=%s, state=%s",
                    case_id,
                    case_state,
                )
                return

        parties = self._get_case_parties(case_id)
        plaintiff_id = getattr(parties["plaintiff"].get("client"), "agent_id", "")
        defendant_id = getattr(parties["defendant"].get("client"), "agent_id", "")
        plaintiff_lawyer_id = parties["plaintiff"].get("lawyer_id", "")
        defendant_lawyer_id = parties["defendant"].get("lawyer_id", "")
        appellant_lawyer_id = payload.get("appellant_lawyer_id", "")
        appellee_lawyer_id = payload.get("appellee_lawyer_id", "")

        logger.info("[Orchestrator] 二审开庭条件满足，提交二审调度器")
        await self._schedule_trial_entry(
            case_id=case_id,
            court="courtB",
            court_level="intermediate",
            event_type=EventType.ENTER_TRIAL_SECOND_INSTANCE,
            payload={
                **payload,
                "case_id": case_id,
                "plaintiff_id": plaintiff_id,
                "defendant_id": defendant_id,
                "plaintiff_lawyer_id": plaintiff_lawyer_id,
                "defendant_lawyer_id": defendant_lawyer_id,
                "appellant_lawyer_id": appellant_lawyer_id,
                "appellee_lawyer_id": appellee_lawyer_id,
                "client_path": client_path,
            },
        )

    async def _on_final_verdict(self, payload: dict) -> None:
        """TRIAL_SECOND_INSTANCE_COMPLETED: 二审庭审结束，发布终审判决事件。"""
        from ..core.event_bus import EventType
        case_id = payload.get("case_id", "")
        trial_slot_released = False
        participant_ids = payload.get("participant_ids", [])
        try:
            if case_id:
                participant_ids = self._collect_case_participant_ids(
                    case_id,
                    extra_ids=[payload.get("judge_id", "")],
                )
                final_result = {
                    "scenario_type": "FINAL_VERDICT",
                    "source_stage": "CIA",
                    "case_id": case_id,
                    "completed": True,
                    "generated_at": datetime.now().isoformat(),
                    "final_judgment": payload.get("final_judgment", ""),
                    "mediation_result": payload.get("mediation_result", {}),
                }
                if not final_result["final_judgment"]:
                    cia_path = self._get_case_output_dir(case_id) / "CIA_result.json"
                    if cia_path.exists():
                        try:
                            with open(cia_path, "r", encoding="utf-8") as f:
                                cia_result = json.load(f)
                            final_result["final_judgment"] = cia_result.get("final_judgment", "")
                            if not final_result["mediation_result"]:
                                final_result["mediation_result"] = cia_result.get("mediation_result", {})
                        except Exception as e:
                            logger.warning(f"[Orchestrator] 读取 CIA_result.json 以生成终审文件失败: {e}")
                self._save_result(case_id, "FINAL_VERDICT", final_result)
            final_payload = {
                **payload,
                "participant_ids": participant_ids,
            }
            await self.event_bus.publish(EventType.FINAL_VERDICT_ISSUED, final_payload)
            await self.event_bus.publish(EventType.CASE_CLOSED, final_payload)
            if case_id:
                await self._release_trial_slot("courtB", case_id)
                trial_slot_released = True
        finally:
            if case_id and not trial_slot_released:
                await self._release_trial_slot("courtB", case_id)

    async def _move_and_despawn(self, agent_id: str, location: str) -> None:
        """Helper function to move an agent to a location and then despawn them."""
        if not self.map_engine:
            return
        await self.map_engine.move_to_location(agent_id, location)
        await self.map_engine.despawn_agent(agent_id)

    async def _run_complaint_drafting(self, payload: dict) -> None:
        """执行起诉状起草场景。"""
        from ..scenarios.complaint_drafting import ComplaintDraftingScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        lawyer_id = payload.get("lawyer_id", "")
        firm_id = payload.get("firm_id", "law_firm_A")
        client_chair = payload.get("client_chair", "")
        lawyer_chair = payload.get("lawyer_chair", "")

        logger.info(f"[Orchestrator] 开始起诉状起草场景: {case_id}")

        # 1. 获取当事人和律师 Agent
        client, _ = self._find_client_for_case(case_id)
        if not client or not client_path:
            logger.error(f"[Orchestrator] 未找到当事人: {case_id}")
            return

        lawyer = self.registry.get_agent(lawyer_id) if lawyer_id else None
        if not lawyer:
            config = self.storage.load_agent_config(client_path)
            lawyer_id = config.get("assigned_lawyer_id", "")
            lawyer = self.registry.get_agent(lawyer_id)

        if not lawyer:
            logger.error(f"[Orchestrator] 未找到律师: {lawyer_id}")
            return

        if self._player_plaintiff_lawyer_enabled():
            await self._run_player_complaint_drafting(
                case_id=case_id,
                client_path=client_path,
                client=client,
                lawyer=lawyer,
                lawyer_id=lawyer_id,
                firm_id=firm_id,
                client_chair=client_chair,
                lawyer_chair=lawyer_chair,
            )
            return

        if self.map_engine and client and lawyer:
            logger.info(f"[Choreography] {lawyer.name} 和 {client.name} 在座位上进行起诉状起草")


        # 2. 加载案件数据
        data_loader, case, config = self._load_case_data(client_path)
        case_background = data_loader.extract_case_background(case)
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        case_context = self._build_case_party_context(
            case_id,
            party_role="plaintiff",
            case=case,
            default_case_background=case_background,
            default_claims=data_loader.extract_claims(case),
            default_evidence=data_loader.extract_plaintiff_evidence(case),
        )

        # 3. 加载原告咨询对话历史（优先 PLC_result.json，兼容旧 LC_result.json）
        dialog_history = self._load_consultation_history(case_id, "PLC")

        # 4. 构建场景数据
        scenario_data = {
            "plaintiff_name": plaintiff_profile.get("name", "") or getattr(client, "name", ""),
            "plaintiff_gender": plaintiff_profile.get("gender", "") or getattr(client, "gender", ""),
            "plaintiff_birth_date": plaintiff_profile.get("birth_date", ""),
            "plaintiff_ethnicity": plaintiff_profile.get("ethnicity", ""),
            "plaintiff_address": plaintiff_profile.get("address", ""),
            "plaintiff_representative": plaintiff_profile.get("representative", ""),
            "defendant_name": case_context.get("defendant_name") or defendant_profile.get("name", ""),
            "defendant_gender": case_context.get("defendant_gender") or defendant_profile.get("gender", ""),
            "defendant_birth_date": case_context.get("defendant_birth_date") or defendant_profile.get("birth_date", ""),
            "defendant_ethnicity": case_context.get("defendant_ethnicity") or defendant_profile.get("ethnicity", ""),
            "defendant_address": case_context.get("defendant_address") or defendant_profile.get("address", ""),
            "defendant_representative": case_context.get("defendant_representative") or defendant_profile.get("representative", ""),
            "case_background": case_context.get("case_background") or case_background,
            "claims": case_context.get("claims") or data_loader.extract_claims(case),
            "evidence": case_context.get("evidence") or data_loader.extract_plaintiff_evidence(case),
            "court_name": data_loader.extract_court_name(case),
            "case_cause": data_loader.extract_case_cause(case),
            "consultation_history": dialog_history,
            "case_output_dir": str(self._get_case_output_dir(case_id).resolve()),
        }

        # 5. 激活 Agents
        client_scenario = PromptAssembler.build_scenario_prompt("client", "CD", scenario_data)
        client_memory = self._get_client_prompt_memory(client, case_id)
        client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(client, plaintiff_profile),
            long_term_memory=client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=client_scenario,
        )
        client.scenario_data = scenario_data
        client.activate(client_prompt)

        if self._player_plaintiff_lawyer_enabled() and not self._player_ai_surrogate_enabled():
            await self._run_player_complaint_drafting(
                case_id=case_id,
                client_path=client_path,
                client=client,
                lawyer=lawyer,
                lawyer_id=lawyer_id,
                firm_id=firm_id,
                client_chair=client_chair,
                lawyer_chair=lawyer_chair,
            )
            return

        lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "CD", scenario_data)
        lawyer_config = self.storage.load_agent_config(lawyer.config_path) if lawyer.config_path else {}
        lawyer_memory = self._get_lawyer_prompt_memory(lawyer, case_id)
        lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(lawyer),
            long_term_memory=lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=lawyer_scenario,
        )
        lawyer.scenario_type = "CD"
        lawyer.scenario_data = dict(scenario_data)
        lawyer.activate(lawyer_prompt)
        self._configure_stage_tools(
            "CD",
            {
                "plaintiff": client,
                "lawyer": lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "CD",
            "CD",
            [client, lawyer],
        )
        self._collect_stage_prompts(case_id, "CD", client, lawyer, reset=True)
        self._mark_case_stage_active(case_id, "CD", [client.agent_id, lawyer.agent_id])
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="CD",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="CD",
            case_cause=scenario_data.get("case_cause", ""),
            case_background=scenario_data.get("case_background", ""),
            trace_recorder=trace_recorder,
        )

        # 6. 执行 ComplaintDraftingScenario
        scenario_succeeded = False
        try:
            output_path = str(self.output_dir / case_id / "CD_result.json")
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: ComplaintDraftingScenario(
                    plaintiff_agent=client,
                    lawyer_agent=lawyer,
                    max_turns=self._resolve_stage_max_turns("CD", 20),
                    output_path=output_path,
                    verbose=SCENARIO_VERBOSE,
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "plaintiff": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                trace_recorder=trace_recorder,
                trace_stage_code="CD",
                trace_stage_key="CD",
                trace_agents=[client, lawyer],
                trace_result_path=output_path,
            )
            self._save_result(case_id, "CD", result or {})
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="CD",
                document_text=str((result or {}).get("complaint_statement") or ""),
                trace_recorder=trace_recorder,
            )

            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code="CD",
                stage_label=self.STAGE_DISPLAY_NAMES.get("CD", "CD"),
                agents=[lawyer, client],
            )
            trace_recorder.export_stage(
                stage_code="CD",
                stage_key="CD",
                agents=[client, lawyer],
                stage_result=result or {},
                stage_result_path=output_path,
                status="completed",
            )
            scenario_succeeded = True
        except Exception as e:
            logger.error(f"[Orchestrator] CD scenario failed: {e}")
            # Recover agents instead of deactivating
            client.recover_from_error()
            lawyer.recover_from_error()
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="CD",
                exc=e,
                stage_label="起诉状起草",
            ):
                return
        finally:
            self._clear_case_stage_active(case_id)
            # 7. 停用 Agents (only if still active)
            if client.is_active:
                client.deactivate()
            if lawyer.is_active:
                lawyer.deactivate()

        if not scenario_succeeded:
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)
            if self.map_engine:
                await self._return_agents_to_birth_and_despawn([client.agent_id, lawyer.agent_id])
            return

        await self._complete_complaint_drafting_after_success(
            case_id=case_id,
            client_path=client_path,
            client=client,
            lawyer=lawyer,
            lawyer_id=lawyer_id,
            firm_id=firm_id,
            client_chair=client_chair,
            lawyer_chair=lawyer_chair,
        )

    def _build_player_cd_client_opening(self, lawyer: Any) -> str:
        lawyer_name = str(getattr(lawyer, "name", "") or "律师").strip()
        return (
            f"{lawyer_name}，刚才咨询的情况我已经说清楚了。"
            "请您根据这些材料帮我正式起草一份民事起诉状，诉讼请求就围绕医疗费、"
            "误工费、护理费、住院伙食补助费、营养费、交通费、车辆损失费和诉讼费来写；"
            "如果还有缺的信息，我可以后面再补充。"
        )

    def _build_player_document_client_opening(self, *, lawyer: Any, document_label: str, stage: str) -> str:
        lawyer_name = str(getattr(lawyer, "name", "") or "律师").strip()
        label = str(document_label or "文书").strip() or "文书"
        stage_code = str(stage or "").strip().upper()
        if stage_code == "AD":
            return (
                f"{lawyer_name}，我决定继续上诉。"
                f"请您根据一审判决和我们前面整理的材料，帮我起草一份{label}，"
                "重点写清上诉请求、事实理由和证据依据。"
            )
        if stage_code == "AR":
            return (
                f"{lawyer_name}，对方已经提交上诉材料了。"
                f"请您帮我起草一份{label}，把我们不同意对方上诉请求的理由和证据依据写清楚。"
            )
        return f"{lawyer_name}，请您根据当前案件材料帮我起草一份{label}。"

    def _ensure_player_document_client_active(
        self,
        *,
        case_id: str,
        client_path: str,
        client: Any,
        client_role: str,
        stage: str,
    ) -> None:
        if getattr(client, "is_active", False):
            return

        data_loader, case, _config = self._load_case_data(client_path)
        party_role = str(client_role or "plaintiff").strip().lower()
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        client_profile = defendant_profile if party_role == "defendant" else plaintiff_profile
        stage_code = str(stage or "").strip().upper()
        case_output_dir = self._get_case_output_dir(case_id)
        first_instance_judgment = ""
        verdict_path = case_output_dir / "CI_result.json"
        if verdict_path.exists():
            try:
                with verdict_path.open("r", encoding="utf-8") as handle:
                    first_instance_judgment = str((json.load(handle) or {}).get("final_judgment") or "")
            except Exception:
                first_instance_judgment = ""

        scenario_data = {
            "case_background": data_loader.extract_case_background(case),
            "case_cause": data_loader.extract_case_cause(case),
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "first_instance_judgment": first_instance_judgment,
            "case_output_dir": str(case_output_dir.resolve()),
        }
        if stage_code == "AD":
            scenario_data.update({
                "appellant_name": client_profile.get("name", "") or getattr(client, "name", ""),
                "appeal_claims": "",
                "appeal_reasons": "",
                "new_evidence": data_loader.extract_second_instance_evidence(case, side="appellant") or "（暂无新证据）",
            })
        elif stage_code == "AR":
            appeal_info = data_loader.extract_appellant_appeal(case)
            appeal_claims = appeal_info.get("claim", [])
            scenario_data.update({
                "appellee_name": client_profile.get("name", "") or getattr(client, "name", ""),
                "appeal_claims": "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(appeal_claims)) if isinstance(appeal_claims, list) else appeal_claims,
                "appeal_reasons": appeal_info.get("reasons", ""),
                "my_position": "",
                "new_evidence": data_loader.extract_second_instance_evidence(case, side="appellee") or "（暂无新证据）",
            })

        client_scenario = PromptAssembler.build_scenario_prompt("client", stage_code or "CD", scenario_data)
        client_memory = self._get_client_prompt_memory(client, case_id)
        client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(client, client_profile),
            long_term_memory=client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=client_scenario,
        )
        client.scenario_data = scenario_data
        client.activate(client_prompt)

    def _build_player_document_summary_dialogue(self, *, document_label: str, stage: str) -> str:
        label = str(document_label or "文书").strip() or "文书"
        stage_code = str(stage or "").strip().upper()
        focus_by_stage = {
            "CD": "诉讼请求、事实理由和证据依据",
            "AD": "上诉请求、事实理由和证据依据",
            "AR": "答辩意见、事实理由和证据依据",
        }
        focus = focus_by_stage.get(stage_code, "核心意见和事实依据")
        return f"我已完成{label}，{focus}已整理清楚，完整文书可以在文书记录中查看。"

    async def _complete_complaint_drafting_after_success(
        self,
        *,
        case_id: str,
        client_path: str,
        client: Any,
        lawyer: Any,
        lawyer_id: str,
        firm_id: str,
        client_chair: str,
        lawyer_chair: str,
    ) -> None:
        """Run the ordinary post-CD filing choreography before advancing the case."""
        from ..core.event_bus import EventType

        if self.map_engine and client and lawyer:
            await self.map_engine.stand_agent(lawyer.agent_id)
            await self.map_engine.stand_agent(client.agent_id)

            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)

            logger.info(f"[Choreography] {client.name} 回到社区中心...")
            asyncio.create_task(
                self._move_and_despawn(
                    client.agent_id,
                    self._get_birth_location_for_agent(client.agent_id),
                )
            )

        if self.map_engine and lawyer:
            logger.info(f"[Choreography] {lawyer.name} 前往法院递交起诉状")

            court_entrance = "courtA_entrance"
            court_loc = self.map_engine.registry.get(court_entrance)
            if court_loc:
                await self.map_engine.move_to_location(lawyer.agent_id, court_entrance)
                logger.info(f"[Choreography] {lawyer.name} 正在递交起诉状...")
                await self.map_engine.play_animation(lawyer.agent_id, "typing", 3.0)
            else:
                logger.warning(f"[Choreography] 法院门口位置 {court_entrance} 不存在")

            logger.info(f"[Choreography] {lawyer.name} 返回社区中心...")
            asyncio.create_task(
                self._move_and_despawn(
                    lawyer.agent_id,
                    self._get_birth_location_for_agent(lawyer.agent_id),
                )
            )

        if getattr(lawyer, "config_path", None):
            try:
                self.storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
                logger.info(f"[Orchestrator] 已清空律师 {lawyer.name} 的当前案件")
            except Exception as exc:
                logger.error(f"[Orchestrator] 清空律师状态失败: {exc}")

        await self._notify_next_waiting_client(lawyer_id, firm_id)
        await self.event_bus.publish(EventType.COMPLAINT_DRAFTING_COMPLETED, {
            "case_id": case_id,
            "client_path": client_path,
            "lawyer_id": lawyer_id,
        })

    async def _run_player_complaint_drafting(
        self,
        *,
        case_id: str,
        client_path: str,
        client: Any,
        lawyer: Any,
        lawyer_id: str,
        firm_id: str,
        client_chair: str,
        lawyer_chair: str,
    ) -> None:
        """Run player-controlled complaint drafting without AI lawyer dialogue."""
        from ..core.event_bus import EventType
        from ..scenarios.complaint_drafting import ComplaintDraftingScenario

        gateway = getattr(self, "_player_gateway", None)
        if gateway is None:
            logger.error("[Orchestrator] Player complaint drafting requested without gateway: case=%s", case_id)
            return

        case_output_dir = self._get_case_output_dir(case_id)
        scenario_succeeded = False
        dialog_history: list[dict[str, Any]] = []
        req = None
        try:
            self._mark_case_stage_active(case_id, "CD", [client.agent_id, lawyer.agent_id])
            client_opening = ComplaintDraftingScenario._sanitize_plaintiff_message(
                self._build_player_cd_client_opening(lawyer)
            )
            if client_opening:
                dialog_history.append({
                    "turn": 0,
                    "role": "plaintiff",
                    "content": client_opening,
                    "timestamp": datetime.now().isoformat(),
                })
                await self._broadcast_dialog_entry(
                    case_id,
                    {
                        "role": "plaintiff",
                        "content": client_opening,
                    },
                    {
                        "plaintiff": client.agent_id,
                        "lawyer": lawyer.agent_id,
                    },
                    1,
                    scenario_type="CD",
                )

            prompt = (
                "请以原告律师身份完成一审《民事起诉状》。你可以先调用后端文书辅助接口生成草稿，"
                "审核修改后确认文书；确认后系统会保存 CD_result.json 并导出 PDF。"
            )
            req = gateway.create_request(
                case_id=case_id,
                stage="CD",
                role="plaintiff_lawyer",
                speaker_label=getattr(lawyer, "name", "原告律师"),
                prompt=prompt,
                context_summary=f"案件 {case_id} · CD 起诉状起草",
            )
            broadcast_fn = getattr(self, "_player_broadcast_fn", None)
            if callable(broadcast_fn):
                broadcast_fn("player_lawyer_input_required", req.to_dict())

            self._register_player_document_followup_session(
                request_id=req.request_id,
                case_id=case_id,
                stage="CD",
                client=client,
                client_role="plaintiff",
                lawyer=lawyer,
                dialog_history=dialog_history,
                role_to_agent_id={
                    "plaintiff": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
            )
            document_text = await asyncio.to_thread(gateway.wait_for_response, req.request_id)
            self._unregister_player_document_followup_session(req.request_id)
            result_path = case_output_dir / "CD_result.json"
            from ..player_lawyer.storage import PlayerLawyerStorage

            existing_payload: dict[str, Any] = {}
            if result_path.exists():
                try:
                    with result_path.open("r", encoding="utf-8") as handle:
                        existing_payload = json.load(handle)
                except Exception:
                    existing_payload = {}
            payload = dict(existing_payload.get("drafted_document_payload") or {})
            if not payload:
                from ..tools.legal.document_drafting_registry import render_document_drafting_payload_for_output_dir
                payload = render_document_drafting_payload_for_output_dir(
                    document_type="complaint",
                    document_text=document_text,
                    case_output_dir=case_output_dir,
                )
            document_summary = self._build_player_document_summary_dialogue(
                document_label="起诉状",
                stage="CD",
            )
            await self._broadcast_dialog_entry(
                case_id,
                {
                    "role": "lawyer",
                    "content": document_summary,
                },
                {
                    "plaintiff": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                len(dialog_history) + 1,
                scenario_type="CD",
            )
            PlayerLawyerStorage(self.sandbox_data_dir).save_document_result(
                case_id=case_id,
                document_text=document_text,
                drafted_document_payload=payload,
                dialog_history=dialog_history,
                dialogue_summary=document_summary,
            )
            scenario_succeeded = True
        except Exception as exc:
            logger.exception("[Orchestrator] Player CD scenario failed")
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="CD",
                exc=exc,
                stage_label="起诉状起草",
            ):
                return
        finally:
            if req is not None:
                self._unregister_player_document_followup_session(req.request_id)
            self._clear_case_stage_active(case_id)
            if getattr(client, "is_active", False):
                client.deactivate()

        if not scenario_succeeded:
            return

        await self._complete_complaint_drafting_after_success(
            case_id=case_id,
            client_path=client_path,
            client=client,
            lawyer=lawyer,
            lawyer_id=lawyer_id,
            firm_id=firm_id,
            client_chair=client_chair,
            lawyer_chair=lawyer_chair,
        )

    async def _run_player_appellate_document_drafting(
        self,
        *,
        case_id: str,
        client_path: str,
        client: Any,
        lawyer: Any,
        lawyer_id: str,
        firm_id: str,
        client_chair: str,
        lawyer_chair: str,
        stage: str,
        document_type: str,
        document_label: str,
        prompt: str,
        completion_event: Any,
        completion_payload: dict,
    ) -> None:
        """Run a player-controlled appellate document stage."""
        gateway = getattr(self, "_player_gateway", None)
        if gateway is None:
            logger.error("[Orchestrator] Player %s requested without gateway: case=%s", stage, case_id)
            return

        scenario_succeeded = False
        case_output_dir = self._get_case_output_dir(case_id)
        result_path = case_output_dir / f"{stage}_result.json"
        dialog_history: list[dict[str, Any]] = []
        req = None
        try:
            self._mark_case_stage_active(case_id, stage, [client.agent_id, lawyer.agent_id])
            client_opening = self._build_player_document_client_opening(
                lawyer=lawyer,
                document_label=document_label,
                stage=stage,
            )
            if client_opening:
                dialog_history.append({
                    "turn": 0,
                    "role": "plaintiff",
                    "content": client_opening,
                    "timestamp": datetime.now().isoformat(),
                })
                await self._broadcast_dialog_entry(
                    case_id,
                    {
                        "role": "plaintiff",
                        "content": client_opening,
                    },
                    {
                        "plaintiff": client.agent_id,
                        "lawyer": lawyer.agent_id,
                    },
                    1,
                    scenario_type=stage,
                )

            client_role = {
                "AD": "appellant",
                "AR": "appellee",
                "DD": "defendant",
            }.get(str(stage or "").upper(), "plaintiff")
            self._ensure_player_document_client_active(
                case_id=case_id,
                client_path=client_path,
                client=client,
                client_role=client_role,
                stage=stage,
            )

            req = gateway.create_request(
                case_id=case_id,
                stage=stage,
                role="plaintiff_lawyer",
                speaker_label=getattr(lawyer, "name", "原告律师"),
                prompt=prompt,
                context_summary=f"案件 {case_id} · {stage} {document_label}起草",
            )
            broadcast_fn = getattr(self, "_player_broadcast_fn", None)
            if callable(broadcast_fn):
                broadcast_fn("player_lawyer_input_required", req.to_dict())

            self._register_player_document_followup_session(
                request_id=req.request_id,
                case_id=case_id,
                stage=stage,
                client=client,
                client_role=client_role,
                lawyer=lawyer,
                dialog_history=dialog_history,
                role_to_agent_id={
                    client_role: client.agent_id,
                    "plaintiff": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
            )
            document_text = await asyncio.to_thread(gateway.wait_for_response, req.request_id)
            self._unregister_player_document_followup_session(req.request_id)
            from ..player_lawyer.storage import PlayerLawyerStorage

            existing_payload: dict[str, Any] = {}
            if result_path.exists():
                try:
                    with result_path.open("r", encoding="utf-8") as handle:
                        existing_payload = json.load(handle)
                except Exception:
                    existing_payload = {}
            payload = dict(existing_payload.get("drafted_document_payload") or {})
            if not payload:
                from ..tools.legal.document_drafting_registry import render_document_drafting_payload_for_output_dir
                payload = render_document_drafting_payload_for_output_dir(
                    document_type=document_type,
                    document_text=document_text,
                    case_output_dir=case_output_dir,
                )
            document_summary = self._build_player_document_summary_dialogue(
                document_label=document_label,
                stage=stage,
            )
            await self._broadcast_dialog_entry(
                case_id,
                {
                    "role": "lawyer",
                    "content": document_summary,
                },
                {
                    "plaintiff": client.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                len(dialog_history) + 1,
                scenario_type=stage,
            )
            PlayerLawyerStorage(self.sandbox_data_dir).save_document_result(
                case_id=case_id,
                document_type=document_type,
                document_text=document_text,
                drafted_document_payload=payload,
                dialog_history=dialog_history,
                dialogue_summary=document_summary,
            )
            if self.map_engine and hasattr(self.map_engine, "broadcast_runtime_progress"):
                await self.map_engine.broadcast_runtime_progress(
                    case_id,
                    phase=f"{stage.lower()}_document_received",
                    message=f"{document_label}已确认，系统正在登记文书。",
                    detail="登记完成后会自动推进到下一诉讼环节。",
                    blocking=False,
                    metadata={"stage": stage, "scenario_type": stage},
                )
            scenario_succeeded = True
        except Exception as exc:
            logger.exception("[Orchestrator] Player %s scenario failed", stage)
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type=stage,
                exc=exc,
                stage_label=f"{document_label}起草",
            ):
                return
        finally:
            if req is not None:
                self._unregister_player_document_followup_session(req.request_id)
            self._clear_case_stage_active(case_id)

        if not scenario_succeeded:
            return

        if lawyer_chair:
            self._release_location(lawyer_chair)
        if client_chair:
            self._release_location(client_chair)
        if getattr(lawyer, "config_path", None):
            try:
                self.storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
            except Exception as exc:
                logger.error("[Orchestrator] 清空律师状态失败: %s", exc)

        await self._notify_next_waiting_client(lawyer_id, firm_id)
        await self.event_bus.publish(completion_event, completion_payload)

    async def _run_defense_drafting(self, payload: dict) -> None:
        """执行答辩状起草场景。"""
        from ..scenarios.defense_drafting import DefenseDraftingScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        lawyer_id = payload.get("lawyer_id", "")
        firm_id = payload.get("firm_id", "law_firm_B")
        client_chair = payload.get("client_chair", "")
        lawyer_chair = payload.get("lawyer_chair", "")

        logger.info(f"[Orchestrator] 开始答辩状起草场景: {case_id}")

        # 1. 获取被告和律师 Agent
        client_id = payload.get("client_id", "")
        defendant = self.registry.get_agent(client_id) if client_id else None
        defendant_id = client_id
        if not defendant:
            _def_agent, _ = self._find_client_for_case(case_id, party_role="defendant")
            defendant_id = _def_agent.agent_id if _def_agent else f"defendant_{case_id}"
            defendant = self.registry.get_agent(defendant_id)

        if not defendant:
            logger.error(f"[Orchestrator] 未找到被告: {defendant_id}")
            return

        lawyer = self.registry.get_agent(lawyer_id) if lawyer_id else None
        if not lawyer:
            logger.error(f"[Orchestrator] 未找到律师: {lawyer_id}")
            return

        if self.map_engine and defendant and lawyer:
            logger.info(f"[Choreography] {lawyer.name} 和 {defendant.name} 在座位上进行答辩状起草")

        # 2. 加载案件数据和起诉状内容
        data_loader, case, config = self._load_case_data(client_path)
        case_background = data_loader.extract_case_background(case)
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        case_context = self._build_case_party_context(
            case_id,
            party_role="defendant",
            case=case,
            default_case_background=case_background,
            default_claims=data_loader.extract_claims(case),
            default_evidence=data_loader.extract_defendant_evidence(case),
        )

        # 加载起诉状
        complaint_path = self.output_dir / case_id / "CD_result.json"
        complaint_statement = ""
        if complaint_path.exists():
            with open(complaint_path, "r", encoding="utf-8") as f:
                cd_data = json.load(f)
                complaint_statement = resolve_stage_document_text(
                    cd_data,
                    "complaint_statement",
                )
        complaint_fields = extract_complaint_prompt_fields(complaint_statement)

        # 3. 构建场景数据
        scenario_data = {
            "plaintiff_name": plaintiff_profile.get("name", ""),
            "plaintiff_gender": plaintiff_profile.get("gender", ""),
            "plaintiff_birth_date": plaintiff_profile.get("birth_date", ""),
            "plaintiff_ethnicity": plaintiff_profile.get("ethnicity", ""),
            "plaintiff_address": plaintiff_profile.get("address", ""),
            "plaintiff_representative": plaintiff_profile.get("representative", ""),
            "defendant_name": case_context.get("defendant_name") or defendant_profile.get("name", "") or getattr(defendant, "name", ""),
            "defendant_gender": case_context.get("defendant_gender") or defendant_profile.get("gender", "") or getattr(defendant, "gender", ""),
            "defendant_birth_date": case_context.get("defendant_birth_date") or defendant_profile.get("birth_date", ""),
            "defendant_ethnicity": case_context.get("defendant_ethnicity") or defendant_profile.get("ethnicity", ""),
            "defendant_address": case_context.get("defendant_address") or defendant_profile.get("address", ""),
            "defendant_representative": case_context.get("defendant_representative") or defendant_profile.get("representative", ""),
            "case_background": case_context.get("case_background") or case_background,
            "claims": complaint_fields.get("claims") or case_context.get("claims") or data_loader.extract_claims(case),
            "my_position": data_loader.extract_defendant_defense(case),
            "facts_and_reasons": complaint_fields.get("facts_and_reasons") or data_loader.extract_facts_and_reasons(case),
            "evidence": case_context.get("evidence") or data_loader.extract_defendant_evidence(case),
            "court_name": data_loader.extract_court_name(case),
            "case_number": data_loader.extract_case_number(case),
            "complaint_statement": complaint_statement,
            "case_cause": data_loader.extract_case_cause(case),
            "case_output_dir": str(self._get_case_output_dir(case_id).resolve()),
        }

        # 4. 激活 Agents (使用 PromptAssembler)
        client_scenario = PromptAssembler.build_scenario_prompt("client", "DD", scenario_data)
        # config is loaded in _load_case_data, defendant might not have it saved, but we'll try
        client_memory = self._get_client_prompt_memory(defendant, case_id)
        client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(defendant, defendant_profile),
            long_term_memory=client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=client_scenario,
        )
        defendant.scenario_data = scenario_data
        defendant.activate(client_prompt)

        lawyer_scenario = PromptAssembler.build_scenario_prompt("lawyer", "DD", scenario_data)
        lawyer_config = self.storage.load_agent_config(lawyer.config_path) if lawyer and lawyer.config_path else {}
        lawyer_memory = self._get_lawyer_prompt_memory(lawyer, case_id)
        lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(lawyer),
            long_term_memory=lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=lawyer_scenario,
        )
        lawyer.scenario_type = "DD"
        lawyer.scenario_data = dict(scenario_data)
        lawyer.activate(lawyer_prompt)
        self._configure_stage_tools(
            "DD",
            {
                "defendant": defendant,
                "lawyer": lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "DD",
            "DD",
            [defendant, lawyer],
        )
        self._collect_stage_prompts(case_id, "DD", defendant, lawyer, reset=True)
        self._mark_case_stage_active(case_id, "DD", [defendant.agent_id, lawyer.agent_id])
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="DD",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="DD",
            case_cause=scenario_data.get("case_cause", ""),
            case_background=scenario_data.get("case_background", ""),
            trace_recorder=trace_recorder,
        )

        # 5. 执行 DefenseDraftingScenario
        scenario_succeeded = False
        try:
            output_path = str(self.output_dir / case_id / "DD_result.json")
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: DefenseDraftingScenario(
                    defendant_agent=defendant,
                    lawyer_agent=lawyer,
                    max_turns=self._resolve_stage_max_turns("DD", 20),
                    output_path=output_path,
                    verbose=SCENARIO_VERBOSE,
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "defendant": defendant.agent_id,
                    "lawyer": lawyer.agent_id,
                },
                trace_recorder=trace_recorder,
                trace_stage_code="DD",
                trace_stage_key="DD",
                trace_agents=[defendant, lawyer],
                trace_result_path=output_path,
            )
            self._save_result(case_id, "DD", result or {})
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="DD",
                document_text=str((result or {}).get("defense_statement") or ""),
                compare_left=complaint_statement,
                compare_right=str((result or {}).get("defense_statement") or ""),
                compare_labels=("complaint", "defense"),
                trace_recorder=trace_recorder,
            )

            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code="DD",
                stage_label=self.STAGE_DISPLAY_NAMES.get("DD", "DD"),
                agents=[lawyer, defendant],
            )
            trace_recorder.export_stage(
                stage_code="DD",
                stage_key="DD",
                agents=[defendant, lawyer],
                stage_result=result or {},
                stage_result_path=output_path,
                status="completed",
            )
            scenario_succeeded = True
        except Exception as e:
            logger.error(f"[Orchestrator] DD scenario failed: {e}")
            # Recover agents instead of deactivating
            defendant.recover_from_error()
            lawyer.recover_from_error()
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="DD",
                exc=e,
                stage_label="答辩状起草",
            ):
                return
        finally:
            self._clear_case_stage_active(case_id)
            # 6. 停用 Agents (only if still active)
            if defendant.is_active:
                defendant.deactivate()
            if lawyer.is_active:
                lawyer.deactivate()

        if not scenario_succeeded:
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)
            if self.map_engine:
                await self._return_agents_to_birth_and_despawn([defendant.agent_id, lawyer.agent_id])
            return

        # 前端动画：律师和当事人站起来，当事人离开
        if self.map_engine and defendant and lawyer:
            await self.map_engine.stand_agent(lawyer.agent_id)
            await self.map_engine.stand_agent(defendant.agent_id)
            
            if lawyer_chair:
                self._release_location(lawyer_chair)
            if client_chair:
                self._release_location(client_chair)

            logger.info(f"[Choreography] {defendant.name} 回到社区中心...")
            asyncio.create_task(
                self._move_and_despawn(
                    defendant.agent_id,
                    self._get_birth_location_for_agent(defendant.agent_id),
                )
            )

        # 7. 律师移动到法院门口递交文书
        if self.map_engine and lawyer:
            logger.info(f"[Choreography] {lawyer.name} 前往法院递交答辩状")

            # 律师生成（如果还未生成）
            await self.map_engine.spawn_agent(
                agent_id=lawyer.agent_id,
                name=lawyer.name,
                character_name=self._get_character_name_for_lawyer(lawyer),
                birth_loc_id=self._get_birth_location_for_agent(lawyer.agent_id),
                role="lawyer",
            )

            # 移动到法院门口
            court_entrance = "courtA_entrance"
            court_loc = self.map_engine.registry.get(court_entrance)
            if court_loc:
                await self.map_engine.move_to_location(lawyer.agent_id, court_entrance)
                logger.info(f"[Choreography] {lawyer.name} 正在递交答辩状...")
                await self.map_engine.play_animation(lawyer.agent_id, "typing", 3.0)
            else:
                logger.warning(f"[Choreography] 法院门口位置 {court_entrance} 不存在")

            # 律师返回并消失
            await self.map_engine.move_to_location(
                lawyer.agent_id,
                self._get_birth_location_for_agent(lawyer.agent_id),
            )
            await self.map_engine.despawn_agent(lawyer.agent_id)

        # 8. 清理律师状态并通知下一个等候者
        if lawyer.config_path:
            try:
                self.storage.update_agent_field(lawyer.config_path, "current_handling_case", None)
                logger.info(f"[Orchestrator] 已清空律师 {lawyer.name} 的当前案件")
            except Exception as e:
                logger.error(f"[Orchestrator] 清空律师状态失败: {e}")

        await self._notify_next_waiting_client(lawyer_id, firm_id)

        # 9. 发布完成事件
        await self.event_bus.publish(EventType.DEFENSE_DRAFTING_COMPLETED, {
            "case_id": case_id,
            "client_path": client_path,
            "lawyer_id": lawyer_id,
        })

    async def _run_first_instance_trial(self, payload: dict) -> None:
        """执行一审庭审场景 (CourtInvestigationScenario)。"""
        from ..scenarios.court_investigation import CourtInvestigationScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        judge_id = payload.get("judge_id", "")
        plaintiff_lawyer_id = payload.get("plaintiff_lawyer_id", "")
        defendant_lawyer_id = payload.get("defendant_lawyer_id", "")
        plaintiff_id = payload.get("plaintiff_id", "")
        defendant_id = payload.get("defendant_id", "")
        parties = self._get_case_parties(case_id)
        plaintiff_bundle = parties.get("plaintiff", {})
        defendant_bundle = parties.get("defendant", {})
        client_path = client_path or plaintiff_bundle.get("client_path", "")
        plaintiff_id = plaintiff_id or getattr(plaintiff_bundle.get("client"), "agent_id", "")
        defendant_id = defendant_id or getattr(defendant_bundle.get("client"), "agent_id", "")
        plaintiff_lawyer_id = plaintiff_lawyer_id or plaintiff_bundle.get("lawyer_id", "")
        defendant_lawyer_id = defendant_lawyer_id or defendant_bundle.get("lawyer_id", "")
        if client_path:
            data_loader, case, _ = self._load_case_data(client_path)
        else:
            logger.error(f"[Orchestrator] 缺少一审案件配置路径: {case_id}")
            return
        if not judge_id:
            judge = next(
                (j for j in self.registry.get_agents_by_type("judge") if getattr(j, "court_level", "") == "basic"),
                None,
            )
            judge_id = getattr(judge, "agent_id", "")
        participant_ids = self._collect_case_participant_ids(
            case_id,
            extra_ids=[judge_id],
        )

        logger.info(f"[Orchestrator] 开始一审庭审场景: {case_id}")

        judge = self.registry.get_agent(judge_id)
        plaintiff = self.registry.get_agent(plaintiff_id)
        defendant = self.registry.get_agent(defendant_id)
        plaintiff_lawyer = self.registry.get_agent(plaintiff_lawyer_id)
        defendant_lawyer = self.registry.get_agent(defendant_lawyer_id)
        if plaintiff_lawyer_id == defendant_lawyer_id:
            logger.error(
                "[Orchestrator] 同一律师被绑定到原被告双方，拒绝启动一审庭审: case=%s, lawyer=%s",
                case_id,
                plaintiff_lawyer_id,
            )
            return
        if not judge or not plaintiff or not defendant or not plaintiff_lawyer or not defendant_lawyer:
            logger.error(
                "[Orchestrator] 无法找到庭审参与者: judge=%s, plaintiff=%s, defendant=%s, plaintiff_lawyer=%s, defendant_lawyer=%s",
                judge_id,
                plaintiff_id,
                defendant_id,
                plaintiff_lawyer_id,
                defendant_lawyer_id,
            )
            return
        if self._player_plaintiff_lawyer_enabled() and not self._player_ai_surrogate_enabled():
            plaintiff_lawyer = self._build_player_lawyer_adapter(plaintiff_lawyer, case_id=case_id, stage="CI")

        case_output_dir = self._get_case_output_dir(case_id)
        shared_case_background = data_loader.extract_case_background(case)
        plaintiff_config = self.storage.load_agent_config(plaintiff.config_path) if plaintiff.config_path else {}
        defendant_config = self.storage.load_agent_config(defendant.config_path) if defendant.config_path else {}
        plaintiff_profile = data_loader.extract_plaintiff_profile(case)
        defendant_profile = data_loader.extract_defendant_profile(case)
        judge_scenario_data = {
            "case_cause": data_loader.extract_case_cause(case),
            "case_number": data_loader.extract_case_number(case),
            "plaintiff_name": str(
                plaintiff_bundle.get("config", {}).get("profile", {}).get("name", "")
                or plaintiff_profile.get("name", "")
                or getattr(plaintiff, "name", "")
            ).strip(),
            "plaintiff_gender": str(
                plaintiff_bundle.get("config", {}).get("profile", {}).get("gender", "")
                or plaintiff_profile.get("gender", "")
                or getattr(plaintiff, "gender", "")
            ).strip(),
            "plaintiff_birth_date": "",
            "plaintiff_address": "",
            "defendant_name": str(
                defendant_bundle.get("config", {}).get("profile", {}).get("name", "")
                or defendant_profile.get("name", "")
                or getattr(defendant, "name", "")
            ).strip(),
            "defendant_gender": str(
                defendant_bundle.get("config", {}).get("profile", {}).get("gender", "")
                or defendant_profile.get("gender", "")
                or getattr(defendant, "gender", "")
            ).strip(),
            "defendant_birth_date": "",
            "defendant_address": "",
            "case_background": shared_case_background,
            "plaintiff_claim": "",
            "case_output_dir": str(case_output_dir.resolve()),
        }

        judge_scenario = PromptAssembler.build_scenario_prompt(
            "judge",
            "CI",
            judge_scenario_data,
        )
        judge.scenario_type = "CI"
        judge.scenario_data = dict(judge_scenario_data)
        judge_prompt = PromptAssembler.build(
            profile={
                "name": judge.name,
                "occupation": "审判长",
                "court_name": getattr(judge, "court_name", ""),
                "court_level": getattr(judge, "court_level", ""),
                "years_of_experience": getattr(judge, "years_of_experience", None),
            },
            scenario_prompt=judge_scenario,
        )
        judge.activate(judge_prompt)

        plaintiff_scenario = PromptAssembler.build_scenario_prompt(
            "client",
            "CI",
            {},
            court_role="plaintiff",
        )
        plaintiff_memory = self._get_client_prompt_memory(plaintiff, case_id)
        plaintiff_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(plaintiff, plaintiff_profile),
            long_term_memory=plaintiff_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=plaintiff_scenario,
        )
        plaintiff.activate(plaintiff_prompt)

        p_scenario_str = PromptAssembler.build_scenario_prompt(
            "lawyer",
            "CI",
            {
                "court_name": data_loader.extract_court_name(case),
            },
            court_role="plaintiff",
        )
        plaintiff_lawyer_memory = self._get_lawyer_prompt_memory(plaintiff_lawyer, case_id)
        plaintiff_lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(plaintiff_lawyer),
            long_term_memory=plaintiff_lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=p_scenario_str,
        )
        plaintiff_lawyer.activate(plaintiff_lawyer_prompt)

        defendant_scenario = PromptAssembler.build_scenario_prompt(
            "client",
            "CI",
            {},
            court_role="defendant",
        )
        defendant_memory = self._get_client_prompt_memory(defendant, case_id)
        defendant_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(defendant, defendant_profile),
            long_term_memory=defendant_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=defendant_scenario,
        )
        defendant.activate(defendant_prompt)

        if self._player_plaintiff_lawyer_enabled():
            d_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CI",
                self._build_ci_opponent_lawyer_scenario_data(data_loader, case, case_output_dir),
                template_key="CI-opponent-defendant",
            )
        else:
            d_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CI",
                {
                    "court_name": data_loader.extract_court_name(case),
                },
                court_role="defendant",
            )
        defendant_lawyer_memory = self._get_lawyer_prompt_memory(defendant_lawyer, case_id)
        defendant_lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(defendant_lawyer),
            long_term_memory=defendant_lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=d_scenario_str,
        )
        defendant_lawyer.activate(defendant_lawyer_prompt)
        self._configure_stage_tools(
            "CI",
            {
                "judge": judge,
                "plaintiff": plaintiff,
                "defendant": defendant,
                "plaintiff_lawyer": plaintiff_lawyer,
                "defendant_lawyer": defendant_lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "CI",
            "CI",
            [judge, plaintiff, defendant, plaintiff_lawyer, defendant_lawyer],
        )
        self._collect_stage_prompts(case_id, "CI", judge, plaintiff, defendant, plaintiff_lawyer, defendant_lawyer, reset=True)
        self._mark_case_stage_active(case_id, "CI", [judge_id, plaintiff_lawyer_id, defendant_lawyer_id, plaintiff_id, defendant_id])
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="CI",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="CI",
            case_cause=judge_scenario_data.get("case_cause", ""),
            case_background=judge_scenario_data.get("case_background", ""),
            trace_recorder=trace_recorder,
        )

        scenario_failed = False
        try:
            output_path = str(case_output_dir / "CI_result.json")
            first_instance = data_loader.extract_first_instance_info(case)
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: CourtInvestigationScenario(
                    judge_agent=judge,
                    plaintiff_agent=plaintiff,
                    defendant_agent=defendant,
                    plaintiff_lawyer_agent=plaintiff_lawyer,
                    defendant_lawyer_agent=defendant_lawyer,
                    plaintiff_witnesses=[],
                    defendant_witnesses=[],
                    max_debate_rounds=4,
                    output_path=output_path,
                    verbose=False,
                    court_finding=first_instance.get("court_finding", ""),
                    court_opinion=first_instance.get("court_opinion", ""),
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "judge": judge_id,
                    "plaintiff": plaintiff_id,
                    "defendant": defendant_id,
                    "plaintiff_lawyer": plaintiff_lawyer_id,
                    "defendant_lawyer": defendant_lawyer_id,
                },
                gap=0.75,
                trace_recorder=trace_recorder,
                trace_stage_code="CI",
                trace_stage_key="CI",
                trace_agents=[judge, plaintiff, defendant, plaintiff_lawyer, defendant_lawyer],
                trace_result_path=output_path,
            )
            self._save_result(case_id, "CI", result or {})
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="CI",
                document_text=str((result or {}).get("final_judgment") or ""),
                trace_recorder=trace_recorder,
            )
            await self._checkpoint_stage_memories(
                case_id=case_id,
                stage_code="CI",
                stage_label=self.STAGE_DISPLAY_NAMES.get("CI", "CI"),
                agents=[plaintiff, defendant, plaintiff_lawyer, defendant_lawyer],
            )
            trace_recorder.export_stage(
                stage_code="CI",
                stage_key="CI",
                agents=[judge, plaintiff, defendant, plaintiff_lawyer, defendant_lawyer],
                stage_result=result or {},
                stage_result_path=output_path,
                status="completed",
            )
            logger.info(f"[Orchestrator] 一审庭审场景完成: {case_id}")
        except Exception as e:
            logger.error(f"[Orchestrator] 一审庭审场景执行失败: {e}")
            judge.recover_from_error()
            plaintiff.recover_from_error()
            defendant.recover_from_error()
            plaintiff_lawyer.recover_from_error()
            defendant_lawyer.recover_from_error()
            scenario_failed = True
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="CI",
                exc=e,
                stage_label="一审庭审",
            ):
                return
            return
        finally:
            self._clear_case_stage_active(case_id)
            if judge.is_active:
                judge.deactivate()
            if plaintiff.is_active:
                plaintiff.deactivate()
            if defendant.is_active:
                defendant.deactivate()
            if plaintiff_lawyer.is_active:
                plaintiff_lawyer.deactivate()
            if defendant_lawyer.is_active:
                defendant_lawyer.deactivate()
            if scenario_failed:
                await self._release_trial_slot("courtA", case_id)

        await self.event_bus.publish(EventType.TRIAL_FIRST_INSTANCE_COMPLETED, {
            **payload,
            "case_id": case_id,
            "client_path": client_path,
            "judge_id": judge_id,
            "participant_ids": participant_ids,
        })


    async def _run_second_instance_trial(self, payload: dict) -> None:
        """执行二审庭审场景 (AppealCourtInvestigationScenario)。"""
        from ..scenarios.appeal_court_investigation import AppealCourtInvestigationScenario
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("appellant_client_path") or payload.get("client_path", "")
        judge_id = payload.get("judge_id", "")
        appellant_lawyer_id = payload.get("appellant_lawyer_id") or payload.get("plaintiff_lawyer_id", "")
        appellee_lawyer_id = payload.get("appellee_lawyer_id") or payload.get("defendant_lawyer_id", "")
        participant_ids = self._collect_case_participant_ids(
            case_id,
            extra_ids=[judge_id],
        )

        logger.info(f"[Orchestrator] 开始二审庭审场景: {case_id}")

        if not client_path:
            parties = self._get_case_parties(case_id)
            appellant_role = payload.get("appellant_role", "plaintiff")
            client_path = (
                parties.get(appellant_role, {}).get("client_path")
                or parties.get("plaintiff", {}).get("client_path", "")
            )

        if not client_path:
            logger.error(f"[Orchestrator] 缺少二审案件配置路径: {case_id}")
            return

        # 1. 获取二审法官、上诉人律师、被上诉人律师 Agent
        judge = self.registry.get_agent(judge_id)
        appellant_lawyer = self.registry.get_agent(appellant_lawyer_id)
        appellee_lawyer = self.registry.get_agent(appellee_lawyer_id)

        if not judge or not appellant_lawyer or not appellee_lawyer:
            logger.error(
                "[Orchestrator] 无法找到二审庭审参与者: judge=%s, appellant_lawyer=%s, appellee_lawyer=%s",
                judge_id,
                appellant_lawyer_id,
                appellee_lawyer_id,
            )
            return

        case_output_dir = self._get_case_output_dir(case_id)
        data_loader, case, _ = self._load_case_data(client_path)
        appeal_roles = self._resolve_appeal_roles(case)
        appellant_role = payload.get("appellant_role") or appeal_roles.get("appellant_role", "plaintiff")
        appellee_role = payload.get("appellee_role") or appeal_roles.get("appellee_role", "defendant")
        parties = self._get_case_parties(case_id)
        appellant_bundle = parties.get(appellant_role, {})
        appellee_bundle = parties.get(appellee_role, {})
        appellant_client_id = payload.get("appellant_client_id") or getattr(appellant_bundle.get("client"), "agent_id", "")
        appellee_client_id = payload.get("appellee_client_id") or getattr(appellee_bundle.get("client"), "agent_id", "")
        appellant_client = self.registry.get_agent(appellant_client_id) if appellant_client_id else None
        appellee_client = self.registry.get_agent(appellee_client_id) if appellee_client_id else None
        if not appellant_client or not appellee_client:
            logger.error(
                "[Orchestrator] 无法找到二审当事人: appellant=%s, appellee=%s",
                appellant_client_id,
                appellee_client_id,
            )
            return
        if self._player_plaintiff_lawyer_enabled() and not self._player_ai_surrogate_enabled():
            if appellant_role == "plaintiff":
                appellant_lawyer = self._build_player_lawyer_adapter(appellant_lawyer, case_id=case_id, stage="CIA")
            elif appellee_role == "plaintiff":
                appellee_lawyer = self._build_player_lawyer_adapter(appellee_lawyer, case_id=case_id, stage="CIA")
        appellant_config = self.storage.load_agent_config(appellant_client.config_path) if appellant_client and appellant_client.config_path else appellant_bundle.get("config", {})
        appellee_config = self.storage.load_agent_config(appellee_client.config_path) if appellee_client and appellee_client.config_path else appellee_bundle.get("config", {})
        second_instance = case.get("extracted_info", {}).get("second_instance", {})

        first_instance_verdict = ""
        verdict_path = case_output_dir / "CI_result.json"
        if verdict_path.exists():
            try:
                with open(verdict_path, "r", encoding="utf-8") as f:
                    verdict_data = json.load(f)
                    first_instance_verdict = verdict_data.get("final_judgment", "")
            except Exception as e:
                logger.error(f"[Orchestrator] 加载一审判决结果失败: {e}")

        party_info = case.get("extracted_info", {}).get("party_info", {})
        plaintiff_profile = party_info.get("plaintiff", {}) if isinstance(party_info.get("plaintiff", {}), dict) else {}
        defendant_raw = party_info.get("defendant", {})
        if isinstance(defendant_raw, list):
            defendant_profile = defendant_raw[0] if defendant_raw else {}
        else:
            defendant_profile = defendant_raw if isinstance(defendant_raw, dict) else {}
        appellant_profile = plaintiff_profile if appellant_role == "plaintiff" else defendant_profile
        appellee_profile = defendant_profile if appellant_role == "plaintiff" else plaintiff_profile
        new_evidence = second_instance.get("new_evidence", {})
        appellant_witnesses = extract_second_instance_witness_entries(new_evidence, side="appellant")
        appellee_witnesses = extract_second_instance_witness_entries(new_evidence, side="appellee")

        common_case_background = data_loader.extract_case_background(case)
        appeal_requests_text = self._stringify_prompt_value(
            second_instance.get("appellant_claim") or "",
        )

        judge_scenario_data = {
            "case_cause": data_loader.extract_case_cause(case),
            "case_number": data_loader.extract_case_number(case, instance="second") or data_loader.extract_case_number(case),
            "appellant_name": payload.get("appellant_name") or appeal_roles.get("appellant_name", "") or appellant_config.get("profile", {}).get("name", ""),
            "appellant_gender": appellant_profile.get("gender", "") or appellant_config.get("profile", {}).get("gender", ""),
            "appellant_birth_date": appellant_profile.get("birth_date", ""),
            "appellant_address": appellant_profile.get("address", ""),
            "appellee_name": payload.get("appellee_name") or appeal_roles.get("appellee_name", "") or appellee_config.get("profile", {}).get("name", ""),
            "appellee_gender": appellee_profile.get("gender", "") or appellee_config.get("profile", {}).get("gender", ""),
            "appellee_birth_date": appellee_profile.get("birth_date", ""),
            "appellee_address": appellee_profile.get("address", ""),
            "case_background": common_case_background,
            "first_instance_judgment": first_instance_verdict,
            "appeal_requests": appeal_requests_text,
            "case_output_dir": str(case_output_dir.resolve()),
        }
        judge.scenario_type = "CIA"
        judge.scenario_data = dict(judge_scenario_data)
        judge_scenario = PromptAssembler.build_scenario_prompt("judge", "CIA", judge_scenario_data)
        judge_prompt = PromptAssembler.build(
            profile={
                "name": judge.name,
                "court_name": getattr(judge, "court_name", ""),
                "court_level": getattr(judge, "court_level", ""),
            },
            scenario_prompt=judge_scenario,
        )
        judge.activate(judge_prompt)

        # 3. 激活 Agents（注入二审庭审信息并使用 PromptAssembler 加载记忆）
        appellant_client_scenario = PromptAssembler.build_scenario_prompt(
            "client",
            "CIA",
            {},
            court_role="appellant",
        )
        appellant_client_memory = self._get_client_prompt_memory(appellant_client, case_id)
        appellant_client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(appellant_client, appellant_profile),
            long_term_memory=appellant_client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=appellant_client_scenario,
        )
        appellant_client.activate(appellant_client_prompt)

        player_plaintiff_mode = self._player_plaintiff_lawyer_enabled()
        appellant_scenario_data = {
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "first_instance_judgment": first_instance_verdict,
        }
        if player_plaintiff_mode and appellant_role != "plaintiff":
            a_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CIA",
                self._build_cia_opponent_lawyer_scenario_data(
                    data_loader,
                    case,
                    court_role="appellant",
                    first_instance_verdict=first_instance_verdict,
                    case_output_dir=case_output_dir,
                ),
                template_key="CIA-opponent-appellant",
            )
        else:
            a_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CIA",
                appellant_scenario_data,
                court_role="appellant",
            )
        appellant_lawyer_memory = self._get_lawyer_prompt_memory(appellant_lawyer, case_id)
        appellant_lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(appellant_lawyer),
            long_term_memory=appellant_lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=a_scenario_str,
        )
        appellant_lawyer.activate(appellant_lawyer_prompt)

        appellee_client_scenario = PromptAssembler.build_scenario_prompt(
            "client",
            "CIA",
            {},
            court_role="appellee",
        )
        appellee_client_memory = self._get_client_prompt_memory(appellee_client, case_id)
        appellee_client_prompt = PromptAssembler.build(
            profile=self._build_client_prompt_profile(appellee_client, appellee_profile),
            long_term_memory=appellee_client_memory,
            memory_owner=CLIENT_MEMORY_OWNER,
            scenario_prompt=appellee_client_scenario,
        )
        appellee_client.activate(appellee_client_prompt)

        appellee_scenario_data = {
            "court_name": data_loader.extract_court_name(case, instance="second") or data_loader.extract_court_name(case),
            "first_instance_judgment": first_instance_verdict,
        }
        if player_plaintiff_mode and appellee_role != "plaintiff":
            e_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CIA",
                self._build_cia_opponent_lawyer_scenario_data(
                    data_loader,
                    case,
                    court_role="appellee",
                    first_instance_verdict=first_instance_verdict,
                    case_output_dir=case_output_dir,
                ),
                template_key="CIA-opponent-appellee",
            )
        else:
            e_scenario_str = PromptAssembler.build_scenario_prompt(
                "lawyer",
                "CIA",
                appellee_scenario_data,
                court_role="appellee",
            )
        appellee_lawyer_memory = self._get_lawyer_prompt_memory(appellee_lawyer, case_id)
        appellee_lawyer_prompt = PromptAssembler.build(
            profile=self._build_lawyer_profile(appellee_lawyer),
            long_term_memory=appellee_lawyer_memory,
            memory_owner=LAWYER_MEMORY_OWNER,
            scenario_prompt=e_scenario_str,
        )
        appellee_lawyer.activate(appellee_lawyer_prompt)
        self._configure_stage_tools(
            "CIA",
            {
                "judge": judge,
                "appellant": appellant_client,
                "appellee": appellee_client,
                "appellant_lawyer": appellant_lawyer,
                "appellee_lawyer": appellee_lawyer,
            },
        )
        trace_recorder = self._bind_case_stage_trace_agents(
            case_id,
            "CIA",
            "CIA",
            [judge, appellant_client, appellee_client, appellant_lawyer, appellee_lawyer],
        )
        self._collect_stage_prompts(
            case_id,
            "CIA",
            judge,
            appellant_client,
            appellee_client,
            appellant_lawyer,
            appellee_lawyer,
            reset=True,
        )
        self._mark_case_stage_active(case_id, "CIA", participant_ids)
        await self._emit_runtime_stage_start(
            case_id=case_id,
            stage_code="CIA",
            trace_recorder=trace_recorder,
        )
        await self._emit_runtime_stage_research(
            case_id=case_id,
            stage_code="CIA",
            case_cause=data_loader.extract_case_cause(case),
            case_background=data_loader.extract_case_background(case),
            trace_recorder=trace_recorder,
        )

        # 4. 执行 AppealCourtInvestigationScenario
        result_data: dict[str, Any] = {}
        scenario_failed = False
        try:
            output_path = str(case_output_dir / "CIA_result.json")
            result = await self._run_sync_scenario_with_live_bubbles(
                case_id=case_id,
                scenario_factory=lambda bubble_publisher: AppealCourtInvestigationScenario(
                    judge_agent=judge,
                    appellant_agent=appellant_client,
                    appellee_agent=appellee_client,
                    appellant_lawyer_agent=appellant_lawyer,
                    appellee_lawyer_agent=appellee_lawyer,
                    appellant_witnesses=appellant_witnesses,
                    appellee_witnesses=appellee_witnesses,
                    max_debate_rounds=4,
                    output_path=output_path,
                    verbose=False,
                    court_finding=second_instance.get("court_finding", "") or second_instance.get("court_findings", ""),
                    court_opinion=second_instance.get("court_opinion", "") or second_instance.get("judgment", ""),
                    bubble_publisher=bubble_publisher,
                ),
                role_to_agent_id={
                    "judge": judge_id,
                    "appellant": appellant_client_id,
                    "appellee": appellee_client_id,
                    "appellant_lawyer": appellant_lawyer_id,
                    "appellee_lawyer": appellee_lawyer_id,
                },
                gap=0.75,
                trace_recorder=trace_recorder,
                trace_stage_code="CIA",
                trace_stage_key="CIA",
                trace_agents=[judge, appellant_client, appellee_client, appellant_lawyer, appellee_lawyer],
                trace_result_path=output_path,
            )
            result_data = result or {}
            self._save_result(case_id, "CIA", result_data)
            await self._emit_runtime_document_complete(
                case_id=case_id,
                stage_code="CIA",
                document_text=str(result_data.get("final_judgment") or ""),
                compare_left=first_instance_verdict,
                compare_right=str(result_data.get("final_judgment") or ""),
                compare_labels=("first_instance_judgment", "second_instance_judgment"),
                trace_recorder=trace_recorder,
            )
            # 二审庭审结束后案件直接进入终审收尾，不再额外更新长期记忆或阶段摘要。
            logger.info(f"[Orchestrator] 二审庭审场景完成: {case_id}")
        except Exception as e:
            logger.error(f"[Orchestrator] 二审庭审场景执行失败: {e}")
            # Recover agents instead of deactivating
            judge.recover_from_error()
            appellant_client.recover_from_error()
            appellee_client.recover_from_error()
            appellant_lawyer.recover_from_error()
            appellee_lawyer.recover_from_error()
            scenario_failed = True
            if await self._report_runtime_issue(
                case_id=case_id,
                scenario_type="CIA",
                exc=e,
                stage_label="二审庭审",
            ):
                return
            return
        finally:
            # 5. 停用 Agents (only if still active)
            self._clear_case_stage_active(case_id)
            if judge.is_active:
                judge.deactivate()
            if appellant_client.is_active:
                appellant_client.deactivate()
            if appellee_client.is_active:
                appellee_client.deactivate()
            if appellant_lawyer.is_active:
                appellant_lawyer.deactivate()
            if appellee_lawyer.is_active:
                appellee_lawyer.deactivate()
            if scenario_failed:
                await self._release_trial_slot("courtB", case_id)

        # 6. 发布完成事件
        await self.event_bus.publish(EventType.TRIAL_SECOND_INSTANCE_COMPLETED, {
            **payload,
            "case_id": case_id,
            "client_path": client_path,
            "judge_id": judge_id,
            "participant_ids": participant_ids,
            "final_judgment": result_data.get("final_judgment", ""),
            "mediation_result": result_data.get("mediation_result", {}),
        })

    async def _auto_close_case(self, payload: dict) -> None:
        """咨询完成后的流程：发布文书起草事件（当事人留在座位上等待起草完毕）。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        client_id = payload.get("client_id", "")
        lawyer_id = payload.get("lawyer_id", "")
        party_role = payload.get("party_role", "plaintiff")
        firm_id = payload.get("firm_id", "law_firm_A")
        client_chair = payload.get("client_chair", "")
        lawyer_chair = payload.get("lawyer_chair", "")

        logger.info(f"[Orchestrator] 咨询完成，开始后续文书起草流程: {case_id}")

        # ═══ 发布文书起草事件 ═══
        if party_role == "plaintiff":
            logger.info(f"[Orchestrator] 发布进入起诉状起草事件: {case_id}")
            await self.event_bus.publish(EventType.ENTER_COMPLAINT_DRAFTING, {
                "case_id": case_id,
                "lawyer_id": lawyer_id,
                "client_path": client_path,
                "firm_id": firm_id,
                "client_id": client_id,
                "client_chair": client_chair,
                "lawyer_chair": lawyer_chair,
            })
        else:  # defendant
            logger.info(f"[Orchestrator] 发布进入答辩状起草事件: {case_id}")
            await self.event_bus.publish(EventType.ENTER_DEFENSE_DRAFTING, {
                "case_id": case_id,
                "lawyer_id": lawyer_id,
                "client_path": client_path,
                "firm_id": firm_id,
                "client_id": client_id,
                "client_chair": client_chair,
                "lawyer_chair": lawyer_chair,
            })

    async def _choreograph_client_called(self, payload: dict) -> None:
        """CLIENT_CALLED: 等候的当事人被通知，从沙发移动到咨询椅。"""
        client_id = payload.get("client_id", "")
        lawyer_id = payload.get("lawyer_id", "")
        case_id = payload.get("case_id", "")
        firm_id = payload.get("firm_id", "law_firm_A")
        party_role = payload.get("party_role", "plaintiff")  # 从payload读取party_role

        client = self.registry.get_agent(client_id)
        lawyer = self.registry.get_agent(lawyer_id)
        if not client or not lawyer:
            return

        logger.info(f"[Choreography] CLIENT_CALLED: {client.name} ({party_role}) 被通知前往咨询区")

        # 重新执行咨询流程（从沙发站起 → 移动到椅子）
        await self._start_consultation_immediately(
            client, lawyer, case_id, firm_id, party_role, payload
        )

    async def _on_complaint_filed(self, payload: dict) -> None:
        """COMPLAINT_DRAFTING_COMPLETED: 起诉状递交完成，发布 LAWSUIT_FILED。"""
        from ..core.event_bus import EventType
        if self._player_plaintiff_lawyer_enabled():
            await self._prepare_player_first_instance_trial_after_complaint(payload)
            return
        await self.event_bus.publish(EventType.LAWSUIT_FILED, payload)

    async def _prepare_player_first_instance_trial_after_complaint(self, payload: dict) -> None:
        """Player plaintiff mode skips opponent pretrial work and schedules CI directly."""
        from .case_fsm import CaseState
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        plaintiff_agent, plaintiff_path = self._find_client_for_case(case_id, party_role="plaintiff")
        plaintiff_path = plaintiff_path or client_path
        if not plaintiff_agent or not plaintiff_path:
            logger.error("[Orchestrator] 玩家原告流程无法定位原告: case=%s", case_id)
            return

        plaintiff_config = self.storage.load_agent_config(plaintiff_path)
        plaintiff_lawyer_id = str(
            payload.get("lawyer_id")
            or plaintiff_config.get("assigned_lawyer_id", "")
            or ""
        ).strip()
        opponent_bundle = self._ensure_player_trial_opponent_bundle(
            {
                **payload,
                "case_id": case_id,
                "client_path": plaintiff_path,
                "plaintiff_lawyer_id": plaintiff_lawyer_id,
            }
        )
        defendant_id = opponent_bundle.get("defendant_id", "")
        defendant_lawyer_id = opponent_bundle.get("defendant_lawyer_id", "")
        if not plaintiff_lawyer_id or not defendant_id or not defendant_lawyer_id:
            logger.error(
                "[Orchestrator] 玩家原告流程一审参与者不完整: case=%s plaintiff_lawyer=%s defendant=%s defendant_lawyer=%s",
                case_id,
                plaintiff_lawyer_id,
                defendant_id,
                defendant_lawyer_id,
            )
            return

        self._set_shared_case_state(case_id, CaseState.WAITING_FOR_FIRST_TRIAL)
        await self._schedule_trial_entry(
            case_id=case_id,
            court="courtA",
            court_level="basic",
            event_type=EventType.ENTER_TRIAL_FIRST_INSTANCE,
            payload={
                **payload,
                "case_id": case_id,
                "plaintiff_id": plaintiff_agent.agent_id,
                "defendant_id": defendant_id,
                "plaintiff_lawyer_id": plaintiff_lawyer_id,
                "defendant_lawyer_id": defendant_lawyer_id,
                "client_path": plaintiff_path,
            },
        )

    async def _on_defense_filed(self, payload: dict) -> None:
        """DEFENSE_DRAFTING_COMPLETED: 答辩状递交完成，发布 DEFENSE_FILED。"""
        from ..core.event_bus import EventType
        await self.event_bus.publish(EventType.DEFENSE_FILED, payload)

    async def _activate_defendant(self, payload: dict) -> None:
        """LAWSUIT_FILED: 激活被告并让其前往律所。"""
        from ..agents.client_agent import ClientAgent
        from ..core.event_bus import EventType
        from .case_fsm import CaseState

        case_id = payload.get("case_id", "")
        client_path = payload.get("client_path", "")
        plaintiff_config = self.storage.load_agent_config(client_path) if client_path else {}

        logger.info(f"[Orchestrator] 激活被告: case_id={case_id}, client_path={client_path}")

        defendant, defendant_path = self._find_client_for_case(case_id, party_role="defendant")

        if defendant:
            defendant_id = defendant.agent_id
            logger.info(f"[Orchestrator] ✓ 找到配置的被告: {defendant.name} ({defendant_id}), config_path={defendant.config_path}")
        else:
            logger.warning(f"[Orchestrator] ✗ 未找到配置的被告，将创建临时 Agent")
            logger.info(f"[Orchestrator] 当前registry中的clients: {[(c.agent_id, c.name, getattr(c, 'role', 'N/A')) for c in self.registry.get_agents_by_type('client')]}")
            try:
                _, case, _ = self._load_case_data(client_path)
                party_info = case.get("extracted_info", {}).get("party_info", {})
                defendant_data = party_info.get("defendant", [])

                # Handle both list and dict formats for defendant_info
                if isinstance(defendant_data, list) and len(defendant_data) > 0:
                    defendant_info = defendant_data[0]
                elif isinstance(defendant_data, dict):
                    defendant_info = defendant_data
                else:
                    defendant_info = {}

                defendant_info = DataLoader.normalize_party_profile(defendant_info)
                logger.info(f"[Orchestrator] 提取被告信息: {defendant_info}")
            except Exception as e:
                logger.error(f"[Orchestrator] 无法加载案件数据: {e}")
                defendant_info = {}

            # 生成被告 Agent
            defendant_id = f"defendant_{case_id}"
            defendant_name = defendant_info.get("name", "被告")
            defendant_gender = defendant_info.get("gender", "")
            logger.info(f"[Orchestrator] 创建临时被告Agent: {defendant_name} ({defendant_id})")

            # Create config path for new defendant using case-based structure
            defendant_config_path = self.storage.get_case_agent_path(case_id, "defendant")
            defendant_config_path.mkdir(parents=True, exist_ok=True)

            # Save defendant config
            defendant_config = {
                "case_id": case_id,
                "party_role": "defendant",
                "profile": {
                    "name": defendant_name,
                    "type": defendant_info.get("type", "") or defendant_info.get("party_type", ""),
                    "party_type": defendant_info.get("party_type", "") or defendant_info.get("type", ""),
                    "gender": defendant_gender,
                    "ethnicity": defendant_info.get("ethnicity", ""),
                    "birth_date": defendant_info.get("birth_date", ""),
                    "address": defendant_info.get("address", ""),
                    "representative": defendant_info.get("representative", ""),
                    "legal_persona_profile": defendant_info.get("legal_persona_profile", {}) or {},
                },
                "case_state": "空闲",
                "dataset_path": plaintiff_config.get("dataset_path", ""),
            }
            self.storage.save_agent_config(str(defendant_config_path), defendant_config)
            defendant_path = str(defendant_config_path)

            defendant = ClientAgent(
                agent_id=defendant_id,
                name=defendant_name,
                gender=defendant_gender,
                role="defendant",
                event_bus=self.event_bus,
                storage=self.storage,
                config_path=defendant_path,
            )

            # 注册到 registry
            self.registry._agents[defendant_id] = defendant

        # 生成并移动到律所
        defendant_config = self.storage.load_agent_config(defendant.config_path) if defendant and defendant.config_path else {}
        if defendant and defendant.config_path:
            try:
                self.storage.update_agent_field(defendant.config_path, "case_state", CaseState.DEFENDANT_SUMMONED)
                defendant_config["case_state"] = CaseState.DEFENDANT_SUMMONED
            except Exception as exc:
                logger.warning("[Orchestrator] 更新被告传唤状态失败: %s", exc)
        existing_assigned_firm = str(defendant_config.get("assigned_firm", "") or "").strip()
        preferred_firm = existing_assigned_firm or self._infer_case_firm_for_defendant(
            payload,
            plaintiff_config,
        )
        target_firm, map_prefix = self._choose_case_firm(
            config_path=defendant.config_path if defendant else None,
            preferred_firm=preferred_firm,
            force_random=not preferred_firm,
        )

        if self.map_engine:
            await self.map_engine.spawn_agent(
                agent_id=defendant_id,
                name=defendant.name,
                character_name=self._get_character_name_for_client(defendant, "defendant"),
                birth_loc_id=self._resolve_birth_location_for_map_prefix(map_prefix),
                role="defendant",
            )

            # 等待一下让前端显示出生动画
            await asyncio.sleep(1.0)

        # 发布被告到达事件
        await self.event_bus.publish(EventType.DEFENDANT_ARRIVED, {
            "client_id": defendant_id,
            "case_id": case_id,
            "target_firm": target_firm,
            "map_prefix": map_prefix,
            "party_role": "defendant",
            "client_path": defendant_path if defendant_path else client_path,  # 使用被告的config路径
        })

    async def _on_defendant_arrived(self, payload: dict) -> None:
        """DEFENDANT_ARRIVED: 被告到达律所，记录日志（实际接待由ReceptionistAgent处理）。"""
        client_id = payload.get("client_id", "")
        case_id = payload.get("case_id", "")
        logger.info(f"[Orchestrator] 被告 {client_id} 已到达律所，案件 {case_id}，等待前台接待")

    async def _check_trial_ready(self, payload: dict) -> None:
        """DEFENSE_FILED: 一审文书完成并进入等待开庭状态后，触发一审开庭。"""
        from ..core.event_bus import EventType

        case_id = payload.get("case_id", "")
        logger.info(f"[Orchestrator] 一审开庭条件检查: {case_id}")

        plaintiff_agent, plaintiff_path = self._find_client_for_case(case_id, party_role="plaintiff")
        defendant_agent, defendant_path = self._find_client_for_case(case_id, party_role="defendant")
        if not plaintiff_agent or not plaintiff_path or not defendant_agent or not defendant_path:
            logger.error(f"[Orchestrator] 无法收集开庭参与者: case_id={case_id}")
            return

        try:
            plaintiff_config, defendant_config, case_state = self._normalize_first_instance_state(
                case_id,
                plaintiff_path,
                defendant_path,
            )
        except Exception as e:
            logger.error(f"[Orchestrator] 无法加载案件状态: {e}")
            return

        if case_state not in {"等待一审开庭", "一审庭审中"}:
            logger.warning(
                "[Orchestrator] 一审开庭前状态不足，跳过开庭: case=%s, plaintiff=%s, defendant=%s",
                case_id,
                plaintiff_config.get("case_state", ""),
                defendant_config.get("case_state", ""),
            )
            return

        plaintiff_id = plaintiff_agent.agent_id
        defendant_id = defendant_agent.agent_id
        plaintiff_lawyer_id = plaintiff_config.get("assigned_lawyer_id", "")
        defendant_lawyer_id = defendant_config.get("assigned_lawyer_id", "")

        if not defendant_lawyer_id:
            lawyers = self.registry.get_agents_by_type("lawyer")
            for lawyer in lawyers:
                if not lawyer.config_path:
                    continue
                lawyer_cfg = self.storage.load_agent_config(lawyer.config_path)
                if lawyer_cfg.get("current_handling_case") == case_id and lawyer.agent_id != plaintiff_lawyer_id:
                    defendant_lawyer_id = lawyer.agent_id
                    break

        if not plaintiff_lawyer_id or not defendant_lawyer_id:
            logger.error(
                "[Orchestrator] 开庭参与律师不完整，拒绝进入庭审: case=%s, plaintiff_lawyer=%s, defendant_lawyer=%s",
                case_id,
                plaintiff_lawyer_id,
                defendant_lawyer_id,
            )
            return

        if plaintiff_lawyer_id == defendant_lawyer_id:
            logger.error(
                "[Orchestrator] 同一律师被绑定到原被告双方，拒绝进入庭审: case=%s, lawyer=%s",
                case_id,
                plaintiff_lawyer_id,
            )
            return

        logger.info("[Orchestrator] 开庭条件满足，提交一审调度器")
        await self._schedule_trial_entry(
            case_id=case_id,
            court="courtA",
            court_level="basic",
            event_type=EventType.ENTER_TRIAL_FIRST_INSTANCE,
            payload={
                "case_id": case_id,
                "plaintiff_id": plaintiff_id,
                "defendant_id": defendant_id,
                "plaintiff_lawyer_id": plaintiff_lawyer_id,
                "defendant_lawyer_id": defendant_lawyer_id,
                "client_path": plaintiff_path,
            },
        )
