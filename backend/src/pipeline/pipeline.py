"""Legal Pipeline for full-process legal simulation.

This module provides the LegalPipeline class for orchestrating complete
legal simulation processes from data loading to scenario execution.
"""

import json
import logging
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime

from ..data.data_loader import DataLoader
from .stage_tool_resolver import apply_stage_tool_permissions
from ..utils.agent_trace import CaseAgentTraceRecorder, bind_agent_trace_context
from ..utils.model_config import resolve_openai_chat_model
from ..utils.runtime_flags import scenario_verbose_enabled

# Stage summary utility for cross-scenario memory
import sys
_src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from utils.drafted_document_sections import (
    extract_appeal_prompt_fields,
    resolve_stage_document_text,
)


logger = logging.getLogger(__name__)
SCENARIO_VERBOSE = scenario_verbose_enabled()
DEFAULT_LAWYER_NAME = "律师"
DEFAULT_OPPONENT_LAWYER_NAME = "对侧律师"
DEFAULT_PLAINTIFF_NAME = "原告"
DEFAULT_DEFENDANT_NAME = "被告"
DEFAULT_APPELLEE_NAME = "被上诉人"
DEFAULT_COURT_NAME = "模拟法庭"
DEFAULT_JUDGE_NAME = "审判长"
PLAINTIFF_ROLE_LABEL = "原告"
DEFENDANT_ROLE_LABEL = "被告"
DEFAULT_FIXED_SIMULATION_MODEL = "qwen3.5-plus"


class LegalPipeline:
    """Full-process Legal Simulation Pipeline.
    
    This pipeline orchestrates the complete legal simulation process:
    1. Load case data
    2. Create agents for each stage with appropriate data
    3. Execute scenarios in controlled sequence
    4. Handle conditional branching (e.g., appeal decisions)
    5. Save results
    
    Design:
    - Non-linear: Uses code-controlled flow in run() instead of linear stages list
    - Supports conditional branching and loops for appeal scenarios
    - Uses stage_output dictionary for data passing between stages
    
    Attributes:
        data_loader: DataLoader instance for case data
        case_data: Current case data dictionary
        output_dir: Directory for output files
        stage_output: Dictionary for passing data between stages
        stage_results: Dictionary storing results from each stage
    """
    
    # Valid stages:
    # LC=法律咨询, DRAFT=文书起草(CD/DD), CI=民事一审,
    # SD=二审判定, APPEAL_DRAFT=二审文书起草(AD/AR), CIA=模拟二审
    VALID_STAGES: Set[str] = {"LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"}
    
    # 闃舵鎵ц椤哄簭
    STAGE_ORDER = ["LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"]
    
    def __init__(
        self,
        data_loader: DataLoader,
        case_index: int = 0,
        output_dir: str = "./output",
        start_stage: Optional[str] = None,
        end_stage: Optional[str] = None,
        party_role: str = "plaintiff",  # 褰撲簨浜鸿鑹? plaintiff(鍘熷憡) 鎴?defendant(琚憡)
        enable_console_output: bool = True,
        evaluated_lawyer_model_name: Optional[str] = None,
        fixed_simulation_model_name: str = DEFAULT_FIXED_SIMULATION_MODEL,
    ):
        """Initialize legal pipeline.
        
        Args:
            data_loader: DataLoader instance with case data
            case_index: Index of case to process (default: 0)
            output_dir: Directory for output files
            start_stage: 浠庢闃舵寮€濮嬫墽琛?(None=浠庡ご寮€濮?
            end_stage: 鎵ц鍒版闃舵鍚庡仠姝?(None=杩愯鍒扮粨灏?
            party_role: 褰撲簨浜鸿鑹? 'plaintiff' 鎴?'defendant'
        """
        self.data_loader = data_loader
        self.case_data = data_loader.get_case(case_index)
        self.output_dir = output_dir
        self.party_role = party_role.lower()  # 褰撲簨浜鸿鑹?        
        self.enable_console_output = bool(enable_console_output)
        self.evaluated_lawyer_model_name = resolve_openai_chat_model(
            explicit_model=evaluated_lawyer_model_name,
        )
        self.fixed_simulation_model_name = (
            str(fixed_simulation_model_name or "").strip()
            or DEFAULT_FIXED_SIMULATION_MODEL
        )
        # 闃舵閫夋嫨鍙傛暟
        self.start_stage = start_stage.upper() if start_stage else None
        self.end_stage = end_stage.upper() if end_stage else None
        
        # 楠岃瘉闃舵鍚嶇О
        if self.start_stage and self.start_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid start_stage: {start_stage}. Valid stages: {self.VALID_STAGES}")
        if self.end_stage and self.end_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid end_stage: {end_stage}. Valid stages: {self.VALID_STAGES}")
        
        # 杩借釜宸叉墽琛岀殑闃舵
        # Track stage execution state
        self._stages_executed: Set[str] = set()
        self._reached_start = self.start_stage is None
        self._reached_end = False
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Data passing between stages (鏅€氬瓧鍏?
        self.stage_output: Dict[str, Any] = {}
        
        # Results from each stage
        self.stage_results: Dict[str, Any] = {}
        
        # Agent prompts collection for debugging/export
        # Structure: {stage_name: [{agent_id, agent_name, agent_class, system_prompt}, ...]}
        self.agent_prompts: Dict[str, List[Dict[str, Any]]] = {}
        self.agent_trace_recorder = CaseAgentTraceRecorder(Path(output_dir))
        self._stage_trace_agents: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Pipeline initialized for case: {self.data_loader.extract_case_id(self.case_data)}")
        if self.start_stage:
            logger.info(f"  Start from stage: {self.start_stage}")
        if self.end_stage:
            logger.info(f"  Stop after stage: {self.end_stage}")

    def _emit_console(self, message: str = "") -> None:
        if self.enable_console_output:
            print(message, flush=True)

    def _resolve_agent_model_name(self, agent_id: str) -> str:
        normalized_agent_id = str(agent_id or "").strip()
        if normalized_agent_id == "lawyer":
            return str(self.evaluated_lawyer_model_name or "").strip() or resolve_openai_chat_model()
        return self.fixed_simulation_model_name

    @staticmethod
    def _stringify_material_value(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(
                f"{index + 1}. {str(item).strip()}"
                for index, item in enumerate(value)
                if str(item or "").strip()
            )
        return str(value or "").strip()

    def _build_first_instance_party_materials(self, party_role: str | None = None) -> dict[str, str]:
        normalized_role = str(party_role or self.party_role or "plaintiff").strip().lower()
        claims = self._stringify_material_value(self.data_loader.extract_claims(self.case_data))
        if normalized_role == "defendant":
            return {
                "claims": claims,
                "my_position": self._stringify_material_value(
                    self.data_loader.extract_defendant_defense(self.case_data)
                ),
                "evidence": self._stringify_material_value(
                    self.data_loader.extract_defendant_evidence(self.case_data)
                ),
            }
        return {
            "claims": claims,
            "my_position": claims,
            "evidence": self._stringify_material_value(
                self.data_loader.extract_plaintiff_evidence(self.case_data)
            ),
        }
    
    def _get_agent_output_dir(self, agent_id: str) -> str:
        """Get output directory for specific agent.
        
        Args:
            agent_id: Agent identifier (e.g., 'lawyer', 'plaintiff')
            
        Returns:
            Path to agent's output directory
        """
        agent_dir = os.path.join(self.output_dir, agent_id)
        Path(agent_dir).mkdir(parents=True, exist_ok=True)
        return agent_dir

    def _bind_stage_trace_agents(self, stage_code: str, slot_to_agent: Dict[str, Any]) -> None:
        normalized_stage_code = str(stage_code or "").strip().upper()
        bindings: Dict[str, Any] = {}
        for slot, agent in dict(slot_to_agent or {}).items():
            if agent is None:
                continue
            bind_agent_trace_context(
                agent,
                recorder=self.agent_trace_recorder,
                output_dir=Path(self.output_dir) / "_debug" / "agent_traces" / str(getattr(agent, "agent_id", slot) or slot),
                stage_code=normalized_stage_code,
                stage_key=normalized_stage_code,
            )
            bindings[str(getattr(agent, "agent_id", slot) or slot)] = agent
        self._stage_trace_agents[normalized_stage_code] = bindings

    def _trace_result_path_for_stage(self, stage_code: str) -> Path:
        mapping = {
            "LC": Path(self.output_dir) / "LC_result.json",
            "CD": Path(self.output_dir) / "CD_result.json",
            "DD": Path(self.output_dir) / "DD_result.json",
            "CI": Path(self.output_dir) / "CI_result.json",
            "AD": Path(self.output_dir) / "AD_result.json",
            "AR": Path(self.output_dir) / "AR_result.json",
            "CIA": Path(self.output_dir) / "CIA_result.json",
        }
        return mapping[str(stage_code or "").strip().upper()]

    def _export_stage_agent_traces(
        self,
        stage_code: str,
        stage_result: Any,
        *,
        status: str = "completed",
        error: str = "",
    ) -> None:
        normalized_stage_code = str(stage_code or "").strip().upper()
        stage_agents = list(self._stage_trace_agents.get(normalized_stage_code, {}).values())
        self.agent_trace_recorder.export_stage(
            stage_code=normalized_stage_code,
            stage_key=normalized_stage_code,
            agents=stage_agents,
            stage_result=stage_result,
            stage_result_path=self._trace_result_path_for_stage(normalized_stage_code),
            status=status,
            error=error,
        )
    
    def _collect_agent_prompts(self, stage: str, *agents) -> None:
        """Collect system prompts from agents for a specific stage.
        
        Args:
            stage: Stage name (e.g., 'LC', 'CI')
            *agents: Agent instances to collect prompts from
        """
        if stage not in self.agent_prompts:
            self.agent_prompts[stage] = []
        
        for agent in agents:
            if agent is not None and hasattr(agent, 'get_prompt_info'):
                prompt_info = agent.get_prompt_info()
                self.agent_prompts[stage].append(prompt_info)
    
    def export_agent_prompts(self, filepath: str) -> None:
        """Export all collected agent prompts to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        export_data = {
            "case_id": self.data_loader.extract_case_id(self.case_data),
            "party_role": self.party_role,
            "export_time": datetime.now().isoformat(),
            "stages": {}
        }
        
        for stage, agents in self.agent_prompts.items():
            export_data["stages"][stage] = {
                "stage_name": stage,
                "agents": agents
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Agent prompts exported to {filepath}")

    @staticmethod
    def _persist_agent_state(
        agent: Any,
        stage_name: str,
        ltm_path: str | None = None,
    ) -> None:
        """Persist long-term memory for one agent."""
        del stage_name
        if ltm_path and hasattr(agent, "extract_and_save_long_term_memory"):
            agent.extract_and_save_long_term_memory(ltm_path)

    @staticmethod
    def _has_meaningful_long_term_memory(memory_payload: Any) -> bool:
        if isinstance(memory_payload, dict):
            return any(LegalPipeline._has_meaningful_long_term_memory(value) for value in memory_payload.values())
        if isinstance(memory_payload, (list, tuple, set)):
            return any(LegalPipeline._has_meaningful_long_term_memory(value) for value in memory_payload)
        return bool(str(memory_payload or "").strip())

    @classmethod
    def _load_long_term_memory(cls, ltm_path: str | None) -> Dict[str, Any]:
        if not ltm_path or not os.path.exists(ltm_path):
            return {}
        try:
            with open(ltm_path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _resolve_lawyer_case_background(
        cls,
        default_background: Any,
        *,
        ltm_path: str | None,
    ) -> str:
        if cls._has_meaningful_long_term_memory(cls._load_long_term_memory(ltm_path)):
            return ""
        return str(default_background or "").strip()

    @staticmethod
    def _run_postprocess_tasks(tasks: List[tuple[str, Callable[[], None]]]) -> None:
        """Run independent LLM post-processing tasks concurrently."""
        active_tasks = [(name, task) for name, task in tasks if task is not None]
        if not active_tasks:
            return
        if len(active_tasks) == 1:
            active_tasks[0][1]()
            return

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(active_tasks), 4)) as executor:
            future_to_name = {
                executor.submit(task): name
                for name, task in active_tasks
            }
            for future in as_completed(future_to_name):
                task_name = future_to_name[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("Post-process task failed: %s", task_name)
                    errors.append(task_name)

        if errors:
            joined = ", ".join(errors)
            raise RuntimeError(f"Post-process task failed: {joined}")

    @staticmethod
    def _configure_stage_tools(stage_code: str, role_to_agent: Dict[str, Any]) -> dict[str, list[str]]:
        """Apply manifest-declared tool permissions for participating agents."""
        return apply_stage_tool_permissions(stage_code, role_to_agent)
    def _should_run_stage(self, stage: str) -> bool:
        """Decide whether the given stage should run under start/end-stage controls."""
        if self._reached_end:
            self._emit_console(f"  [跳过] {stage} - 已达到结束阶段")
            return False

        if self.start_stage and stage == self.start_stage:
            self._reached_start = True

        if not self._reached_start:
            self._emit_console(f"  [跳过] {stage} - 尚未到起始阶段 {self.start_stage}")
            return False

        return True

    def _mark_stage_done(self, stage: str) -> None:
        """Mark a stage as completed and update end-stage status."""
        self._stages_executed.add(stage)
        if self.end_stage and stage == self.end_stage:
            self._reached_end = True
            self._emit_console(f"  [完成] 已到达结束阶段 {stage}")

    def _notify_stage_started(self, stage: str) -> None:
        """Hook for subclasses that need stage start notifications."""
        return None
    def run(self) -> Dict[str, Any]:
        """Execute the full-process pipeline."""
        start_time = datetime.now()
        case_id = self.data_loader.extract_case_id(self.case_data)

        self._emit_console("=" * 60)
        self._emit_console("案件流程模拟 Pipeline")
        self._emit_console(f"案件ID: {case_id}")
        if self.start_stage:
            self._emit_console(f"起始阶段: {self.start_stage}")
        if self.end_stage:
            self._emit_console(f"结束阶段: {self.end_stage}")
        self._emit_console("=" * 60)

        if self._should_run_stage("LC"):
            self._notify_stage_started("LC")
            self._emit_console("")
            self._emit_console("[阶段] LC - 法律咨询")
            self._emit_console("-" * 40)
            lc_result = self._execute_lc()
            self.stage_results["LC"] = lc_result
            self.stage_output["lc_dialog"] = lc_result.get("dialog_history", [])
            self._emit_console(f"  轮次: {lc_result.get('turn_count', 0)} 轮")
            self._mark_stage_done("LC")

        if self._should_run_stage("DRAFT"):
            self._notify_stage_started("DRAFT")
            if self.party_role == "plaintiff":
                self._emit_console("")
                self._emit_console("[阶段] DRAFT (CD) - 起诉状起草")
                self._emit_console("-" * 40)
                draft_result = self._execute_cd()
                self.stage_results["DRAFT"] = draft_result
                complaint_text = resolve_stage_document_text(draft_result, "complaint_statement")
                self.stage_output["complaint"] = complaint_text
                self._emit_console(f"  轮次: {draft_result.get('turn_count', 0)} 轮")
                if complaint_text:
                    self._emit_console(
                        f"  文书长度: {len(complaint_text)} 字"
                    )
            else:
                self._emit_console("")
                self._emit_console("[阶段] DRAFT (DD) - 答辩状起草")
                self._emit_console("-" * 40)
                draft_result = self._execute_dd()
                self.stage_results["DRAFT"] = draft_result
                defense_text = resolve_stage_document_text(draft_result, "defense_statement")
                self.stage_output["defense_brief"] = defense_text
                self._emit_console(f"  轮次: {draft_result.get('turn_count', 0)} 轮")
                if defense_text:
                    self._emit_console(
                        f"  文书长度: {len(defense_text)} 字"
                    )
            self._mark_stage_done("DRAFT")

        if self._should_run_stage("CI"):
            self._notify_stage_started("CI")
            self._emit_console("")
            self._emit_console("[阶段] CI - 民事一审")
            self._emit_console("-" * 40)
            ci_result = self._execute_ci()
            self.stage_results["CI"] = ci_result
            self.stage_output["judgment"] = ci_result.get("final_judgment", "")
            self._emit_console(f"  轮次: {ci_result.get('total_turns', 0)} 轮")
            if ci_result.get("final_judgment"):
                self._emit_console(
                    f"  判决书长度: {len(str(ci_result.get('final_judgment', '')))} 字"
                )
            self._mark_stage_done("CI")

        if self._should_run_stage("SD"):
            self._notify_stage_started("SD")
            self._emit_console("")
            self._emit_console("[阶段] SD - 二审判定")
            self._emit_console("-" * 40)
            sd_result = self._execute_sd()
            self.stage_results["SD"] = sd_result
            self.stage_output["second_instance_info"] = sd_result
            self._emit_console(f"  上诉人: {sd_result.get('appellant_name', 'Unknown')}")
            self._emit_console(f"  当前身份: {sd_result.get('my_role', 'Unknown')}")
            self._emit_console(
                f"  是否为上诉方: {'是' if sd_result.get('is_appellant') else '否'}"
            )
            self._mark_stage_done("SD")

        if self._should_run_stage("APPEAL_DRAFT"):
            self._notify_stage_started("APPEAL_DRAFT")
            self._emit_console("")
            self._emit_console("[阶段] APPEAL_DRAFT - 二审文书起草")
            self._emit_console("-" * 40)
            ad_result = self._execute_appeal_draft()
            self.stage_results["APPEAL_DRAFT"] = ad_result
            appeal_text = resolve_stage_document_text(ad_result, "appeal_statement")
            appeal_response_text = resolve_stage_document_text(ad_result, "appeal_response_statement")
            if appeal_text:
                self.stage_output["appeal_statement"] = appeal_text
            if appeal_response_text:
                self.stage_output["appeal_response_statement"] = appeal_response_text
            self._mark_stage_done("APPEAL_DRAFT")

        if self._should_run_stage("CIA"):
            self._notify_stage_started("CIA")
            self._emit_console("")
            self._emit_console("[阶段] CIA - 二审庭审")
            self._emit_console("-" * 40)
            cia_result = self._execute_cia()
            self.stage_results["CIA"] = cia_result
            self.stage_output["appeal_judgment"] = cia_result.get("final_judgment", "")
            self._emit_console(f"  轮次: {cia_result.get('total_turns', 0)} 轮")
            if cia_result.get("final_judgment"):
                self._emit_console(
                    f"  二审判决书长度: {len(str(cia_result.get('final_judgment', '')))} 字"
                )
            self._mark_stage_done("CIA")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        final_result = self._build_final_result()
        final_result["execution_time_seconds"] = duration

        final_output_path = os.path.join(self.output_dir, "pipeline_result.json")
        self._save_result(final_result, final_output_path)

        self._emit_console("")
        self._emit_console("=" * 60)
        self._emit_console("Pipeline 完成")
        self._emit_console(f"耗时: {duration:.2f} 秒")
        self._emit_console(f"完成阶段: {', '.join(self.stage_results.keys())}")
        self._emit_console(f"结果文件: {final_output_path}")
        self._emit_console("=" * 60)

        return final_result

    def _execute_lc(self) -> Dict[str, Any]:
        """Execute Legal Consultation (LC) stage.
        
        Returns:
            LC stage result dictionary
        """
        from ..agents import ClientAgent, LawyerAgent
        from ..scenarios import LegalConsultationScenario
        
        # Extract data for LC stage
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        case_background = self.data_loader.extract_case_background(self.case_data)
        party_materials = self._build_first_instance_party_materials("plaintiff")
        
        # Build scenario_data for client
        scenario_data = {
            "case_background": case_background,
            "questions": plaintiff_profile.get("questions", []),
            "claims": party_materials.get("claims", ""),
            "my_position": party_materials.get("my_position", ""),
            "evidence": party_materials.get("evidence", ""),
            "case_cause": self.data_loader.extract_case_cause(self.case_data),
        }
        
        # Work memory paths - per identity, stored in agent subdirectories
        plaintiff_output_dir = self._get_agent_output_dir("plaintiff")
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        # Stage summaries paths (replacing work_memory for cross-scenario context)
        
        # Create client agent
        client_agent = ClientAgent(
            agent_id="plaintiff",
            name=plaintiff_profile.get("name", DEFAULT_PLAINTIFF_NAME),
            party_type=plaintiff_profile.get("party_type", "") or plaintiff_profile.get("type", ""),
            representative=plaintiff_profile.get("representative", ""),
            gender=plaintiff_profile.get("gender", ""),
            ethnicity=plaintiff_profile.get("ethnicity", ""),
            birth_date=plaintiff_profile.get("birth_date", ""),
            address=plaintiff_profile.get("address", ""),
            personality=plaintiff_profile.get("personality", ""),
            speaking_style=plaintiff_profile.get("speaking_style", ""),
            interaction_guidelines=plaintiff_profile.get("interaction_guidelines", ""),
            legal_persona_profile=plaintiff_profile.get("legal_persona_profile", {}),
            scenario_type="LC",
            scenario_data=scenario_data,
            role="plaintiff",
            model_type=self._resolve_agent_model_name("plaintiff"),
        )
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        
        # Create lawyer agent
        lawyer_agent = LawyerAgent(
            agent_id="lawyer",
            name=DEFAULT_LAWYER_NAME,
            scenario_type="LC",
            scenario_data=scenario_data,
            long_term_memory_path=lawyer_ltm_path,
            model_type=self._resolve_agent_model_name("lawyer"),
        )
        self._configure_stage_tools(
            "LC",
            {
                "client": client_agent,
                "lawyer": lawyer_agent,
            },
        )
        self._bind_stage_trace_agents(
            "LC",
            {
                "plaintiff": client_agent,
                "lawyer": lawyer_agent,
            },
        )
        
        # Execute scenario
        output_path = os.path.join(self.output_dir, "LC_result.json")
        scenario = LegalConsultationScenario(
            client_agent=client_agent,
            lawyer_agent=lawyer_agent,
            output_path=output_path,
            verbose=SCENARIO_VERBOSE,
            trace_recorder=self.agent_trace_recorder,
            trace_stage_code="LC",
            trace_stage_key="LC",
        )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("LC", client_agent, lawyer_agent)
        
        result = scenario.execute()
        self._run_postprocess_tasks([
            (
                "lc_client_summary",
                lambda: self._persist_agent_state(
                    client_agent,
                    "法律咨询",
                ),
            ),
            (
                "lc_lawyer_state",
                lambda: self._persist_agent_state(
                    lawyer_agent,
                    "法律咨询",
                    ltm_path=lawyer_ltm_path,
                ),
            ),
        ])
        self._export_stage_agent_traces("LC", result)
        return result
        
        # Extract and save long term memory for Lawyer
        lawyer_agent.extract_and_save_long_term_memory(lawyer_ltm_path)
        
        # Generate and save stage summaries (replacing work_memory export)
        
        return result
    
    def _execute_cd(self) -> Dict[str, Any]:
        """Execute Complaint Drafting (CD) stage.
        
        Returns:
            CD stage result dictionary
        """
        from ..agents import ClientAgent, LawyerAgent
        from ..scenarios import ComplaintDraftingScenario
        
        # Extract data for CD stage
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        
        # Build scenario_data for plaintiff
        # ?????????????????????????
        scenario_data = {
            "plaintiff_name": plaintiff_profile.get("name", ""),
            "plaintiff_gender": plaintiff_profile.get("gender", ""),
            "plaintiff_birth_date": plaintiff_profile.get("birth_date", ""),
            "plaintiff_ethnicity": plaintiff_profile.get("ethnicity", ""),
            "plaintiff_address": plaintiff_profile.get("address", ""),
            "plaintiff_representative": plaintiff_profile.get("representative", ""),
            "defendant_name": defendant_profile.get("name", ""),
            "defendant_gender": defendant_profile.get("gender", ""),
            "defendant_birth_date": defendant_profile.get("birth_date", ""),
            "defendant_ethnicity": defendant_profile.get("ethnicity", ""),
            "defendant_address": defendant_profile.get("address", ""),
            "defendant_representative": defendant_profile.get("representative", ""),
            "case_background": self.data_loader.extract_case_background(self.case_data),
            "claims": self.data_loader.extract_claims(self.case_data),
            "evidence": self.data_loader.extract_plaintiff_evidence(self.case_data),
            "court_name": self.data_loader.extract_court_name(self.case_data),
            "case_cause": self.data_loader.extract_case_cause(self.case_data),
            "case_output_dir": str(Path(self.output_dir).resolve()),
        }
        
        # Work memory paths - per identity, stored in agent subdirectories
        plaintiff_output_dir = self._get_agent_output_dir("plaintiff")
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        # Stage summaries paths
        
        # Create plaintiff agent
        plaintiff_agent = ClientAgent(
            agent_id="plaintiff",
            name=plaintiff_profile.get("name", DEFAULT_PLAINTIFF_NAME),
            party_type=plaintiff_profile.get("party_type", "") or plaintiff_profile.get("type", ""),
            representative=plaintiff_profile.get("representative", ""),
            gender=plaintiff_profile.get("gender", ""),
            ethnicity=plaintiff_profile.get("ethnicity", ""),
            birth_date=plaintiff_profile.get("birth_date", ""),
            address=plaintiff_profile.get("address", ""),
            personality=plaintiff_profile.get("personality", ""),
            speaking_style=plaintiff_profile.get("speaking_style", ""),
            interaction_guidelines=plaintiff_profile.get("interaction_guidelines", ""),
            legal_persona_profile=plaintiff_profile.get("legal_persona_profile", {}),
            scenario_type="CD",
            scenario_data=scenario_data,
            role="plaintiff",
            model_type=self._resolve_agent_model_name("plaintiff"),
        )
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        
        # Create lawyer agent
        lawyer_agent = LawyerAgent(
            agent_id="lawyer",
            name=DEFAULT_LAWYER_NAME,
            scenario_type="CD",
            scenario_data=scenario_data,
            long_term_memory_path=lawyer_ltm_path,
            model_type=self._resolve_agent_model_name("lawyer"),
        )
        self._configure_stage_tools(
            "CD",
            {
                "plaintiff": plaintiff_agent,
                "lawyer": lawyer_agent,
            },
        )
        self._bind_stage_trace_agents(
            "CD",
            {
                "plaintiff": plaintiff_agent,
                "lawyer": lawyer_agent,
            },
        )
        
        # Execute scenario
        output_path = os.path.join(self.output_dir, "CD_result.json")
        scenario = ComplaintDraftingScenario(
            plaintiff_agent=plaintiff_agent,
            lawyer_agent=lawyer_agent,
            output_path=output_path,
            verbose=SCENARIO_VERBOSE,
            trace_recorder=self.agent_trace_recorder,
            trace_stage_code="CD",
            trace_stage_key="CD",
        )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("CD", plaintiff_agent, lawyer_agent)
        
        result = scenario.execute()
        self._run_postprocess_tasks([
            (
                "cd_plaintiff_summary",
                lambda: self._persist_agent_state(
                    plaintiff_agent,
                    "起诉状起草",
                ),
            ),
            (
                "cd_lawyer_state",
                lambda: self._persist_agent_state(
                    lawyer_agent,
                    "起诉状起草",
                    ltm_path=lawyer_ltm_path,
                ),
            ),
        ])
        self._export_stage_agent_traces("CD", result)
        return result
        
        # Extract and save long term memory for Lawyer
        lawyer_agent.extract_and_save_long_term_memory(lawyer_ltm_path)
        
        # Generate and save stage summaries (replacing work_memory export)
        
        return result
    
    def _execute_ci(self) -> Dict[str, Any]:
        """Execute Civil First Instance (CI) stage.
        
        Agent identity design:
        - 寰呮祴璇勫緥甯?(evaluated lawyer): uses agent_id="lawyer", memory dir="lawyer/"
        - 瀵规墜寰嬪笀 (opponent lawyer): uses agent_id="opponent_lawyer", memory dir="opponent_lawyer/"
        - Which one plays plaintiff/defendant depends on party_role.
        
        Returns:
            CI stage result dictionary
        """
        from ..agents import ClientAgent, LawyerAgent, JudgeAgent
        from ..scenarios import CourtInvestigationScenario
        
        # 1. Prepare Data
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        
        # New fields from dataset
        judge_name = self.data_loader.extract_judge_name(self.case_data) or DEFAULT_JUDGE_NAME
        court_name = self.data_loader.extract_court_name(self.case_data) or DEFAULT_COURT_NAME
        case_cause = self.data_loader.extract_case_cause(self.case_data)
        case_number = self.data_loader.extract_case_number(self.case_data)
        
        # Judge Data
        judge_data = {
            "plaintiff_name": plaintiff_profile.get("name", DEFAULT_PLAINTIFF_NAME),
            "plaintiff_gender": plaintiff_profile.get("gender", ""),
            "plaintiff_birth_date": plaintiff_profile.get("birth_date", ""),
            "plaintiff_address": plaintiff_profile.get("address", ""),
            "defendant_name": defendant_profile.get("name", DEFAULT_DEFENDANT_NAME),
            "defendant_gender": defendant_profile.get("gender", ""),
            "defendant_birth_date": defendant_profile.get("birth_date", ""),
            "defendant_address": defendant_profile.get("address", ""),
            "case_background": "",
            "plaintiff_claim": [],
            "court_finding": self.data_loader.extract_first_instance_info(self.case_data).get("court_finding", ""),
            "court_opinion": self.data_loader.extract_court_opinion(self.case_data),
            "case_cause": case_cause,
            "case_number": case_number,
            "case_output_dir": str(Path(self.output_dir).resolve()),
        }
        
        p_lawyer_data = {
            "court_name": court_name,
            "case_cause": case_cause,
        }
        
        d_lawyer_data = {
            "court_name": court_name,
            "case_cause": case_cause,
        }

        # Memory paths for parties
        plaintiff_output_dir = self._get_agent_output_dir("plaintiff")
        defendant_output_dir = self._get_agent_output_dir("defendant")
        plaintiff_ltm_path = os.path.join(plaintiff_output_dir, "memory.yaml")
        defendant_ltm_path = os.path.join(defendant_output_dir, "memory.yaml")

        # Memory paths for lawyers
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        opponent_output_dir = self._get_agent_output_dir("opponent_lawyer")
        
        # Stage summaries paths for lawyers
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        opponent_ltm_path = os.path.join(opponent_output_dir, "memory.yaml")
        # 2. Create Agents
        
        # Judge (no cross-scenario memory needed, uses ChatAgent's built-in session memory)
        judge = JudgeAgent(
            agent_id="judge",
            name=judge_name,
            court_name=court_name,
            years_of_experience=20,
            scenario_type="CI",
            scenario_data=judge_data,
            model_type=self._resolve_agent_model_name("judge"),
        )

        plaintiff_client = ClientAgent(
            agent_id="plaintiff",
            name=plaintiff_profile.get("name", DEFAULT_PLAINTIFF_NAME),
            party_type=plaintiff_profile.get("party_type", "") or plaintiff_profile.get("type", ""),
            representative=plaintiff_profile.get("representative", ""),
            gender=plaintiff_profile.get("gender", ""),
            ethnicity=plaintiff_profile.get("ethnicity", ""),
            birth_date=plaintiff_profile.get("birth_date", ""),
            address=plaintiff_profile.get("address", ""),
            personality=plaintiff_profile.get("personality", ""),
            speaking_style=plaintiff_profile.get("speaking_style", ""),
            interaction_guidelines=plaintiff_profile.get("interaction_guidelines", ""),
            legal_persona_profile=plaintiff_profile.get("legal_persona_profile", {}),
            role="plaintiff",
            scenario_type="CI",
            scenario_data={},
            long_term_memory_path=plaintiff_ltm_path,
            model_type=self._resolve_agent_model_name("plaintiff"),
        )

        defendant_client = ClientAgent(
            agent_id="defendant",
            name=defendant_profile.get("name", DEFAULT_DEFENDANT_NAME),
            party_type=defendant_profile.get("party_type", "") or defendant_profile.get("type", ""),
            representative=defendant_profile.get("representative", ""),
            gender=defendant_profile.get("gender", ""),
            ethnicity=defendant_profile.get("ethnicity", ""),
            birth_date=defendant_profile.get("birth_date", ""),
            address=defendant_profile.get("address", ""),
            personality=defendant_profile.get("personality", ""),
            speaking_style=defendant_profile.get("speaking_style", ""),
            interaction_guidelines=defendant_profile.get("interaction_guidelines", ""),
            legal_persona_profile=defendant_profile.get("legal_persona_profile", {}),
            role="defendant",
            scenario_type="CI",
            scenario_data={},
            long_term_memory_path=defendant_ltm_path,
            model_type=self._resolve_agent_model_name("defendant"),
        )
        
        # Determine which lawyer is evaluated vs opponent based on party_role
        if self.party_role == "plaintiff":
            # 寰呮祴璇勫緥甯?= 鍘熷憡寰嬪笀, 瀵规墜寰嬪笀 = 琚憡寰嬪笀
            plaintiff_lawyer = LawyerAgent(
                agent_id="lawyer",
                name=DEFAULT_LAWYER_NAME,
                scenario_type="CI",
                court_role="plaintiff",
                scenario_data=p_lawyer_data,
                long_term_memory_path=lawyer_ltm_path,
                model_type=self._resolve_agent_model_name("lawyer"),
            )
            
            defendant_lawyer = LawyerAgent(
                agent_id="opponent_lawyer",
                name=DEFAULT_OPPONENT_LAWYER_NAME,
                scenario_type="CI",
                court_role="defendant",
                scenario_data=d_lawyer_data,
                long_term_memory_path=opponent_ltm_path,
                model_type=self._resolve_agent_model_name("opponent_lawyer"),
            )
            
            evaluated_lawyer = plaintiff_lawyer
            opponent_lawyer = defendant_lawyer
        else:
            # 寰呮祴璇勫緥甯?= 琚憡寰嬪笀, 瀵规墜寰嬪笀 = 鍘熷憡寰嬪笀
            plaintiff_lawyer = LawyerAgent(
                agent_id="opponent_lawyer",
                name=DEFAULT_OPPONENT_LAWYER_NAME,
                scenario_type="CI",
                court_role="plaintiff",
                scenario_data=p_lawyer_data,
                long_term_memory_path=opponent_ltm_path,
                model_type=self._resolve_agent_model_name("opponent_lawyer"),
            )
            
            defendant_lawyer = LawyerAgent(
                agent_id="lawyer",
                name=DEFAULT_LAWYER_NAME,
                scenario_type="CI",
                court_role="defendant",
                scenario_data=d_lawyer_data,
                long_term_memory_path=lawyer_ltm_path,
                model_type=self._resolve_agent_model_name("lawyer"),
            )
            
            evaluated_lawyer = defendant_lawyer
            opponent_lawyer = plaintiff_lawyer

        self._configure_stage_tools(
            "CI",
            {
                "judge": judge,
                "plaintiff": plaintiff_client,
                "defendant": defendant_client,
                "plaintiff_lawyer": plaintiff_lawyer,
                "defendant_lawyer": defendant_lawyer,
            },
        )
        self._bind_stage_trace_agents(
            "CI",
            {
                "judge": judge,
                "plaintiff": plaintiff_client,
                "defendant": defendant_client,
                "lawyer": evaluated_lawyer,
                "opponent_lawyer": opponent_lawyer,
            },
        )

        # 3. Execution
        output_path = os.path.join(self.output_dir, "CI_result.json")
        scenario = CourtInvestigationScenario(
            judge_agent=judge,
            plaintiff_agent=plaintiff_client,
            defendant_agent=defendant_client,
            plaintiff_lawyer_agent=plaintiff_lawyer,
            defendant_lawyer_agent=defendant_lawyer,
            plaintiff_witnesses=party_witnesses if self.party_role == "plaintiff" else [],
            defendant_witnesses=party_witnesses if self.party_role == "defendant" else [],
            court_finding=judge_data.get("court_finding", ""),
            court_opinion=judge_data.get("court_opinion", ""),
            max_debate_rounds=4,
            output_path=output_path,
            verbose=SCENARIO_VERBOSE,
            trace_recorder=self.agent_trace_recorder,
            trace_stage_code="CI",
            trace_stage_key="CI",
        )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("CI", judge, plaintiff_client, defendant_client, plaintiff_lawyer, defendant_lawyer)
        
        result = scenario.execute()
        self._run_postprocess_tasks([
            (
                "ci_plaintiff_state",
                lambda: self._persist_agent_state(
                    plaintiff_client,
                    "民事一审",
                    ltm_path=plaintiff_ltm_path,
                ),
            ),
            (
                "ci_defendant_state",
                lambda: self._persist_agent_state(
                    defendant_client,
                    "民事一审",
                    ltm_path=defendant_ltm_path,
                ),
            ),
            (
                "ci_lawyer_state",
                lambda: self._persist_agent_state(
                    evaluated_lawyer,
                    "民事一审",
                    ltm_path=lawyer_ltm_path,
                ),
            ),
            (
                "ci_opponent_lawyer_state",
                lambda: self._persist_agent_state(
                    opponent_lawyer,
                    "民事一审",
                    ltm_path=opponent_ltm_path,
                ),
            ),
        ])
        self._export_stage_agent_traces("CI", result)
        return result
        
        # Extract and save long term memory for parties
        plaintiff_client.extract_and_save_long_term_memory(plaintiff_ltm_path)
        defendant_client.extract_and_save_long_term_memory(defendant_ltm_path)

        # Extract and save long term memory for Lawyers
        evaluated_lawyer.extract_and_save_long_term_memory(lawyer_ltm_path)
        opponent_lawyer.extract_and_save_long_term_memory(opponent_ltm_path)
        
        # Generate and save stage summaries for all participating agents

        
        return result
    
    def _check_appeal_condition(self, ci_result: Dict[str, Any]) -> bool:
        """Check if appeal is needed based on first instance result.
        
        Args:
            ci_result: First instance (CI) stage result
            
        Returns:
            True if appeal should be filed
        """
        # Appeal gating is disabled in this public pipeline entry point.
        # Downstream users can replace this with their own procedural policy.
        return False
    
    def _check_further_appeal(self, second_trial_result: Dict[str, Any]) -> bool:
        """Check if further appeal is needed after second instance.
        
        Args:
            second_trial_result: Second instance result
            
        Returns:
            True if further appeal should be filed
        """
        # Further-appeal gating is disabled in this public pipeline entry point.
        return False
    
    def _build_final_result(self) -> Dict[str, Any]:
        """Build the final pipeline result dictionary.
        
        Returns:
            Final result dictionary
        """
        return {
            "case_id": self.data_loader.extract_case_id(self.case_data),
            "case_cause": self.data_loader.extract_case_cause(self.case_data),
            "stages_completed": list(self.stage_results.keys()),
            "stage_results": self.stage_results,
            "stage_output": self.stage_output,
            "timestamp": datetime.now().isoformat(),
        }
    
    def _execute_dd(self) -> Dict[str, Any]:
        """Execute Defense Drafting (DD) stage.
        
        Returns:
            DD stage result dictionary
        """
        from ..agents import ClientAgent, LawyerAgent
        from ..scenarios import DefenseDraftingScenario
        
        # Extract data for DD stage
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        
        # Build scenario_data for defendant
        # Defendant sees plaintiff's claims and their own defenses
        court_name = self.data_loader.extract_court_name(self.case_data)
        party_materials = self._build_first_instance_party_materials("defendant")
            
        scenario_data = {
            "plaintiff_name": plaintiff_profile.get("name", ""),
            "plaintiff_gender": plaintiff_profile.get("gender", ""),
            "plaintiff_birth_date": plaintiff_profile.get("birth_date", ""),
            "plaintiff_ethnicity": plaintiff_profile.get("ethnicity", ""),
            "plaintiff_address": plaintiff_profile.get("address", ""),
            "plaintiff_representative": plaintiff_profile.get("representative", ""),
            "defendant_name": defendant_profile.get("name", ""),
            "defendant_gender": defendant_profile.get("gender", ""),
            "defendant_birth_date": defendant_profile.get("birth_date", ""),
            "defendant_ethnicity": defendant_profile.get("ethnicity", ""),
            "defendant_address": defendant_profile.get("address", ""),
            "defendant_representative": defendant_profile.get("representative", ""),
            "case_background": self.data_loader.extract_case_background(self.case_data),
            "claims": self.data_loader.extract_claims(self.case_data),
            "my_position": party_materials.get("my_position", ""),
            "court_name": court_name,
            "evidence": self.data_loader.extract_defendant_evidence(self.case_data),
            "facts_and_reasons": self.data_loader.extract_facts_and_reasons(self.case_data),
            "case_number": self.data_loader.extract_case_number(self.case_data),
            "case_cause": self.data_loader.extract_case_cause(self.case_data),
            "case_output_dir": str(Path(self.output_dir).resolve()),
        }
        
        # Stage summaries paths
        defendant_output_dir = self._get_agent_output_dir("defendant")
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        
        # Create defendant agent
        defendant_agent = ClientAgent(
            agent_id="defendant",
            name=defendant_profile.get("name", DEFAULT_DEFENDANT_NAME),
            party_type=defendant_profile.get("party_type", "") or defendant_profile.get("type", ""),
            representative=defendant_profile.get("representative", ""),
            gender=defendant_profile.get("gender", ""),
            ethnicity=defendant_profile.get("ethnicity", ""),
            birth_date=defendant_profile.get("birth_date", ""),
            address=defendant_profile.get("address", ""),
            personality=defendant_profile.get("personality", ""),
            speaking_style=defendant_profile.get("speaking_style", ""),
            interaction_guidelines=defendant_profile.get("interaction_guidelines", ""),
            legal_persona_profile=defendant_profile.get("legal_persona_profile", {}),
            scenario_type="DD",
            scenario_data=scenario_data,
            role="defendant",
            model_type=self._resolve_agent_model_name("defendant"),
        )
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        
        # Create lawyer agent
        lawyer_agent = LawyerAgent(
            agent_id="lawyer",
            name=DEFAULT_LAWYER_NAME,
            scenario_type="DD",
            scenario_data=scenario_data,
            long_term_memory_path=lawyer_ltm_path,
            model_type=self._resolve_agent_model_name("lawyer"),
        )
        self._configure_stage_tools(
            "DD",
            {
                "defendant": defendant_agent,
                "lawyer": lawyer_agent,
            },
        )
        self._bind_stage_trace_agents(
            "DD",
            {
                "defendant": defendant_agent,
                "lawyer": lawyer_agent,
            },
        )
        
        output_path = os.path.join(self.output_dir, "DD_result.json")
        scenario = DefenseDraftingScenario(
            defendant_agent=defendant_agent,
            lawyer_agent=lawyer_agent,
            output_path=output_path,
            verbose=SCENARIO_VERBOSE,
            trace_recorder=self.agent_trace_recorder,
            trace_stage_code="DD",
            trace_stage_key="DD",
        )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("DD", defendant_agent, lawyer_agent)
        
        result = scenario.execute()
        self._run_postprocess_tasks([
            (
                "dd_defendant_summary",
                lambda: self._persist_agent_state(
                    defendant_agent,
                    "答辩状起草",
                ),
            ),
            (
                "dd_lawyer_state",
                lambda: self._persist_agent_state(
                    lawyer_agent,
                    "答辩状起草",
                    ltm_path=lawyer_ltm_path,
                ),
            ),
        ])
        self._export_stage_agent_traces("DD", result)
        return result
        
        # Extract and save long term memory for Lawyer
        lawyer_agent.extract_and_save_long_term_memory(lawyer_ltm_path)
        
        # Generate and save stage summaries (replacing work_memory export)
        
        return result

    def _execute_sd(self) -> Dict[str, Any]:
        """Execute Second Decision (SD) stage.
        
        Determines who is appealing based on case data.
        The appellant field in dataset is now either "鍘熷憡" or "琚憡".
        
        Returns:
            Dict containing appellant info and my role in second instance.
        """
        extracted_info = self.case_data.get("extracted_info", {})
        # appellant瀛楁鐜板湪鍙湁涓ょ鍊? "鍘熷憡" 鎴?"琚憡"
        appellant_role = extracted_info.get("appellant", "")
        
        print(f"  [SD璋冭瘯] appellant瀛楁鍊? {repr(appellant_role)}")
        print(f"  [SD璋冭瘯] party_role: {self.party_role}")
        
        # 鐩存帴姣旇緝瑙掕壊鍒ゆ柇鏄惁涓轰笂璇変汉
        # party_role: "plaintiff" 瀵瑰簲 "鍘熷憡", "defendant" 瀵瑰簲 "琚憡"
        is_appellant = (
            (self.party_role == "plaintiff" and appellant_role == PLAINTIFF_ROLE_LABEL) or
            (self.party_role == "defendant" and appellant_role == DEFENDANT_ROLE_LABEL)
        )
        
        my_role = "appellant" if is_appellant else "appellee"
        
        print(f"  [SD璋冭瘯] is_appellant: {is_appellant}, my_role: {my_role}")
        
        # 鑾峰彇涓婅瘔浜虹殑瀹為檯濮撳悕鐢ㄤ簬鏄剧ず
        if appellant_role == PLAINTIFF_ROLE_LABEL:
            appellant_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        else:
            appellant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        appellant_name = appellant_profile.get("name", appellant_role)
        
        print(f"  [SD璋冭瘯] appellant_name: {appellant_name}")
        
        return {
            "appellant_name": appellant_name,
            "my_role": my_role, # appellant or appellee
            "is_appellant": is_appellant,
            "timestamp": datetime.now().isoformat()
        }

    def _execute_appeal_draft(self) -> Dict[str, Any]:
        """Execute Appeal Drafting (APPEAL_DRAFT) stage.
        
        Executes AB or AR scenario based on SD result.
        
        Returns:
            Scenario result
        """
        from ..agents import ClientAgent, LawyerAgent
        from ..scenarios import AppealDraftingScenario, AppealResponseDraftingScenario
        
        # Get SD result
        sd_result = self.stage_results.get("SD", {})
        is_appellant = sd_result.get("is_appellant", False)
        
        # Get Judgment from CI stage
        final_judgment = self.stage_output.get("judgment", "")
        if not final_judgment:
            # Fallback if CI was not run or failed to produce judgment text.
            logger.warning("No judgment found from CI stage; using fallback judgment instruction")
            final_judgment = "（未找到一审判决书内容，请假设一审判决结果）"
            
        court_name = self.data_loader.extract_court_name(self.case_data, instance="second")
        if not court_name:
            court_name = self.data_loader.extract_court_name(self.case_data, instance="first")
        if court_name and court_name not in final_judgment:
            final_judgment += f"\n\n一审法院：{court_name}"
        
        # Case number and judgment metadata
        case_number = self.data_loader.extract_case_number(self.case_data, instance="second")
        if not case_number:
            case_number = self.data_loader.extract_case_number(self.case_data, instance="first")
            
        # Common data
        # Note: In Appeal, ClientAgent role is 'appellant' or 'appellee'.
        # But we initialize it with original profile info.
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        
        # Get my profile
        if self.party_role == "plaintiff":
            my_profile = plaintiff_profile
            original_role_desc = PLAINTIFF_ROLE_LABEL
        else:
            my_profile = defendant_profile
            original_role_desc = DEFENDANT_ROLE_LABEL
        extracted_info = self.case_data.get("extracted_info", {})
        appellant_value = str(extracted_info.get("appellant", "") or "").strip()
        plaintiff_name = str(plaintiff_profile.get("name", "") or "").strip()
        defendant_name = str(defendant_profile.get("name", "") or "").strip()
        all_defendant_names = str(defendant_profile.get("all_defendant_names", "") or "").strip()
        plaintiff_is_appellant = (
            appellant_value == PLAINTIFF_ROLE_LABEL
            or (plaintiff_name and plaintiff_name in appellant_value)
        )
        defendant_is_appellant = (
            appellant_value == DEFENDANT_ROLE_LABEL
            or (defendant_name and defendant_name in appellant_value)
            or (all_defendant_names and all_defendant_names in appellant_value)
        )
        if not plaintiff_is_appellant and not defendant_is_appellant:
            plaintiff_is_appellant = (
                (self.party_role == "plaintiff" and is_appellant)
                or (self.party_role == "defendant" and not is_appellant)
            )
        if plaintiff_is_appellant:
            appellant_profile = plaintiff_profile
            appellee_profile = defendant_profile
        else:
            appellant_profile = defendant_profile
            appellee_profile = plaintiff_profile
            
        # Extract appeal data with dataset fallback so AR can still see
        # appellant requests/reasons even when no AD_result.json exists.
        second_instance_info = self.data_loader.extract_second_instance_info(self.case_data)
        appeal_claims_value = second_instance_info.get("appeal_claims", "")
        if isinstance(appeal_claims_value, list):
            appeal_claims_str = "\n".join(
                f"{index + 1}. {str(item).strip()}"
                for index, item in enumerate(appeal_claims_value)
                if str(item or "").strip()
            )
        else:
            appeal_claims_str = str(appeal_claims_value or "").strip()
        appeal_reasons = str(second_instance_info.get("appeal_reasons", "") or "").strip()
        existing_appeal_statement = ""
        appeal_result_path = Path(self.output_dir) / "AD_result.json"
        if appeal_result_path.exists():
            appeal_result = json.loads(appeal_result_path.read_text(encoding="utf-8"))
            existing_appeal_statement = resolve_stage_document_text(
                appeal_result,
                "appeal_statement",
            )
            appeal_fields = extract_appeal_prompt_fields(existing_appeal_statement)
            appeal_claims_str = str(appeal_fields.get("appeal_claims", "") or appeal_claims_str).strip()
            appeal_reasons = str(appeal_fields.get("appeal_reasons", "") or appeal_reasons).strip()
        # ???????
        second_instance = self.data_loader._extract_instance_mapping(self.case_data, "second_instance")
        new_evidence_data = second_instance.get("new_evidence", {})
        new_evidence_data = second_instance.get("new_evidence", {})
        # ???????????????
        if is_appellant:
            my_new_evidence = self._format_evidence(new_evidence_data.get("appellant_evidence", {}))
        else:
            my_new_evidence = self._format_evidence(new_evidence_data.get("appellee_evidence", {}))
        
        # 鏍煎紡鍖栨柊璇佹嵁涓哄瓧绗︿覆
        if my_new_evidence:
            new_evidence_str = "\n".join(f"{i+1}. {e}" for i, e in enumerate(my_new_evidence))
        else:
            new_evidence_str = ""
        appellee_position = (
            self.data_loader.extract_second_instance_appellee_defense(self.case_data)
            or second_instance_info.get("respondent_defense", "")
        )
        
        scenario_data = {
            "first_instance_judgment": final_judgment,
            "original_role": original_role_desc,
            "appellant_name": appellant_profile.get("name", ""),
            "appellant_gender": appellant_profile.get("gender", ""),
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
            "case_background": self.data_loader.extract_case_background(self.case_data),
            "court_name": court_name,
            "case_number": case_number,
            "case_cause": self.data_loader.extract_case_cause(self.case_data),
            "appeal_claims": appeal_claims_str,
            "appeal_reasons": appeal_reasons,
            "my_position": str(appellee_position or "").strip(),
            "new_evidence": new_evidence_str,
            "appeal_statement": existing_appeal_statement,
            "case_output_dir": str(Path(self.output_dir).resolve()),
        }
        
        # Work memory paths
        # We reuse the same agent directories to keep continuity? 
        # Or maybe separate appeal dirs? Let's reuse 'plaintiff'/'defendant' dirs but maybe rename agent_id?
        # Actually ClientAgent constructor takes agent_id.
        # Let's keep agent_id as "client_appeal" to separate or reuse "plaintiff"/"defendant"?
        # User wants continuity? "Before second instance, reading dataset...".
        # Let's use "client" and "lawyer".
        
        agent_role = "appellant" if is_appellant else "appellee"
        scenario_type = "AD" if is_appellant else "AR"
        
        # Stage summaries paths
        agent_output_dir = self._get_agent_output_dir(self.party_role)
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        
        client_agent = ClientAgent(
            agent_id=self.party_role,
            name=my_profile.get(
                "name",
                DEFAULT_PLAINTIFF_NAME if self.party_role == "plaintiff" else DEFAULT_DEFENDANT_NAME,
            ),
            party_type=my_profile.get("party_type", "") or my_profile.get("type", ""),
            representative=my_profile.get("representative", ""),
            gender=my_profile.get("gender", ""),
            ethnicity=my_profile.get("ethnicity", ""),
            birth_date=my_profile.get("birth_date", ""),
            address=my_profile.get("address", ""),
            personality=my_profile.get("personality", ""),
            speaking_style=my_profile.get("speaking_style", ""),
            interaction_guidelines=my_profile.get("interaction_guidelines", ""),
            legal_persona_profile=my_profile.get("legal_persona_profile", {}),
            scenario_type=scenario_type,
            scenario_data=scenario_data,
            role=agent_role,
            model_type=self._resolve_agent_model_name(self.party_role),
        )
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        
        lawyer_agent = LawyerAgent(
            agent_id="lawyer",
            name=DEFAULT_LAWYER_NAME,
            scenario_type=scenario_type,
            scenario_data=scenario_data,
            long_term_memory_path=lawyer_ltm_path,
            model_type=self._resolve_agent_model_name("lawyer"),
        )
        self._configure_stage_tools(
            scenario_type,
            {
                agent_role: client_agent,
                "lawyer": lawyer_agent,
            },
        )
        self._bind_stage_trace_agents(
            scenario_type,
            {
                self.party_role: client_agent,
                "lawyer": lawyer_agent,
            },
        )
        
        output_path = os.path.join(self.output_dir, f"{scenario_type}_result.json")
        
        if is_appellant:
            scenario = AppealDraftingScenario(
                appellant_agent=client_agent,
                lawyer_agent=lawyer_agent,
                output_path=output_path,
                verbose=SCENARIO_VERBOSE,
                trace_recorder=self.agent_trace_recorder,
                trace_stage_code="AD",
                trace_stage_key="AD",
            )
        else:
            scenario = AppealResponseDraftingScenario(
                appellee_agent=client_agent,
                lawyer_agent=lawyer_agent,
                output_path=output_path,
                verbose=SCENARIO_VERBOSE,
                trace_recorder=self.agent_trace_recorder,
                trace_stage_code="AR",
                trace_stage_key="AR",
            )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("APPEAL_DRAFT", client_agent, lawyer_agent)
            
        result = scenario.execute()
        stage_name = "上诉状起草" if is_appellant else "上诉答辩状起草"
        self._run_postprocess_tasks([
            (
                "appeal_client_summary",
                lambda: self._persist_agent_state(
                    client_agent,
                    stage_name,
                ),
            ),
            (
                "appeal_lawyer_state",
                lambda: self._persist_agent_state(
                    lawyer_agent,
                    stage_name,
                    ltm_path=lawyer_ltm_path,
                ),
            ),
        ])
        self._export_stage_agent_traces(scenario_type, result)
        return result
        
        # Extract and save long term memory for Lawyer
        lawyer_agent.extract_and_save_long_term_memory(lawyer_ltm_path)
        
        # Generate and save stage summaries (replacing work_memory export)
        appeal_stage_name = "上诉状起草" if is_appellant else "上诉答辩状起草"
        
        return result
    
    def _save_result(self, result: Dict[str, Any], filepath: str) -> None:
        """Save result to JSON file.
        
        Args:
            result: Result dictionary to save
            filepath: Path to save the file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Result saved to {filepath}")

    def _execute_cia(self) -> Dict[str, Any]:
        """Execute Appeal Court Investigation (CIA) stage.
        
        Agent identity design:
        - 浜屽娉曞畼: uses agent_id="appeal_judge", memory dir="appeal_judge/" (鐙珛浜庝竴瀹℃硶瀹?
        - 寰呮祴璇勫緥甯? uses agent_id="lawyer", memory dir="lawyer/"
        - 瀵规墜寰嬪笀: uses agent_id="opponent_lawyer", memory dir="opponent_lawyer/"
        
        Returns:
            CIA stage result dictionary
        """
        from ..agents import ClientAgent, LawyerAgent, JudgeAgent
        from ..scenarios import AppealCourtInvestigationScenario
        
        # 1. Get SD result
        sd_result = self.stage_output.get("second_instance_info", {})
        is_appellant = sd_result.get("is_appellant", False)
        
        # 2. Prepare basic data
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.case_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.case_data)
        
        # Determine appellant and appellee profiles
        extracted_info = self.case_data.get("extracted_info", {})
        appellant_role = extracted_info.get("appellant", "")
        
        if appellant_role == PLAINTIFF_ROLE_LABEL:
            appellant_profile = plaintiff_profile
            appellee_profile = defendant_profile
        else:
            appellant_profile = defendant_profile
            appellee_profile = plaintiff_profile
        
        # Get second instance data
        second_instance = self.data_loader._extract_instance_mapping(self.case_data, "second_instance")
        court_name = second_instance.get("court", "") or self.data_loader.extract_court_name(self.case_data, instance="first")
        case_number = second_instance.get("case_number", "") or self.data_loader.extract_case_number(self.case_data, instance="first")
        case_cause = self.data_loader.extract_case_cause(self.case_data)
        
        # Get judge names for second instance
        judges = second_instance.get("judges", [])
        judge_name = judges[0] if judges else DEFAULT_JUDGE_NAME
        
        # Get first instance judgment
        first_instance_judgment = self.stage_output.get("judgment", "")
        
        # Get appeal data
        # 3. Prepare Judge Data
        judge_data = {
            "case_cause": case_cause,
            "case_number": case_number,
            "appellant_name": appellant_profile.get("name", "上诉人"),
            "appellant_gender": appellant_profile.get("gender", ""),
            "appellant_birth_date": appellant_profile.get("birth_date", ""),
            "appellant_address": appellant_profile.get("address", ""),
            "appellee_name": appellee_profile.get("name", DEFAULT_APPELLEE_NAME),
            "appellee_gender": appellee_profile.get("gender", ""),
            "appellee_birth_date": appellee_profile.get("birth_date", ""),
            "appellee_address": appellee_profile.get("address", ""),
            "case_background": "",
            "first_instance_judgment": first_instance_judgment,
            "appeal_requests": [],
            "case_output_dir": str(Path(self.output_dir).resolve()),
        }
        
        # 4. Prepare Appellant Lawyer Data
        appellant_lawyer_data = {
            "court_name": court_name,
            "case_cause": case_cause,
            "first_instance_judgment": first_instance_judgment,
        }
        
        # 5. Prepare Appellee Lawyer Data
        appellee_lawyer_data = {
            "court_name": court_name,
            "case_cause": case_cause,
            "first_instance_judgment": first_instance_judgment,
        }

        appellant_output_key = "plaintiff" if appellant_role == PLAINTIFF_ROLE_LABEL else "defendant"
        appellee_output_key = "defendant" if appellant_output_key == "plaintiff" else "plaintiff"
        appellant_output_dir = self._get_agent_output_dir(appellant_output_key)
        appellee_output_dir = self._get_agent_output_dir(appellee_output_key)
        appellant_ltm_path = os.path.join(appellant_output_dir, "memory.yaml")
        appellee_ltm_path = os.path.join(appellee_output_dir, "memory.yaml")
        
        # 6. Memory paths
        lawyer_output_dir = self._get_agent_output_dir("lawyer")
        opponent_output_dir = self._get_agent_output_dir("opponent_lawyer")
        
        # Stage summaries paths for lawyers
        
        lawyer_ltm_path = os.path.join(lawyer_output_dir, "memory.yaml")
        opponent_ltm_path = os.path.join(opponent_output_dir, "memory.yaml")
        # 7. Create Agents
        
        # 浜屽娉曞畼 (鐙珛韬唤锛屾棤璺ㄥ満鏅蹇嗭紝浣跨敤 ChatAgent 鍐呯疆浼氳瘽璁板繂)
        appeal_judge = JudgeAgent(
            agent_id="appeal_judge",
            name=judge_name,
            court_name=court_name,
            years_of_experience=25,
            scenario_type="CIA",
            scenario_data=judge_data,
            model_type=self._resolve_agent_model_name("appeal_judge"),
        )

        appellant_client = ClientAgent(
            agent_id="appellant",
            name=appellant_profile.get("name", "上诉人"),
            party_type=appellant_profile.get("party_type", "") or appellant_profile.get("type", ""),
            representative=appellant_profile.get("representative", ""),
            gender=appellant_profile.get("gender", ""),
            ethnicity=appellant_profile.get("ethnicity", ""),
            birth_date=appellant_profile.get("birth_date", ""),
            address=appellant_profile.get("address", ""),
            personality=appellant_profile.get("personality", ""),
            speaking_style=appellant_profile.get("speaking_style", ""),
            interaction_guidelines=appellant_profile.get("interaction_guidelines", ""),
            legal_persona_profile=appellant_profile.get("legal_persona_profile", {}),
            role="appellant",
            scenario_type="CIA",
            scenario_data={},
            long_term_memory_path=appellant_ltm_path,
            model_type=self._resolve_agent_model_name("appellant"),
        )

        appellee_client = ClientAgent(
            agent_id="appellee",
            name=appellee_profile.get("name", DEFAULT_APPELLEE_NAME),
            party_type=appellee_profile.get("party_type", "") or appellee_profile.get("type", ""),
            representative=appellee_profile.get("representative", ""),
            gender=appellee_profile.get("gender", ""),
            ethnicity=appellee_profile.get("ethnicity", ""),
            birth_date=appellee_profile.get("birth_date", ""),
            address=appellee_profile.get("address", ""),
            personality=appellee_profile.get("personality", ""),
            speaking_style=appellee_profile.get("speaking_style", ""),
            interaction_guidelines=appellee_profile.get("interaction_guidelines", ""),
            legal_persona_profile=appellee_profile.get("legal_persona_profile", {}),
            role="appellee",
            scenario_type="CIA",
            scenario_data={},
            long_term_memory_path=appellee_ltm_path,
            model_type=self._resolve_agent_model_name("appellee"),
        )
        
        # Determine which lawyer is evaluated vs opponent based on is_appellant
        if is_appellant:
            appellant_lawyer = LawyerAgent(
                agent_id="lawyer",
                name=DEFAULT_LAWYER_NAME,
                scenario_type="CIA",
                court_role="appellant",
                scenario_data=appellant_lawyer_data,
                long_term_memory_path=lawyer_ltm_path,
                model_type=self._resolve_agent_model_name("lawyer"),
            )

            appellee_lawyer = LawyerAgent(
                agent_id="opponent_lawyer",
                name=DEFAULT_OPPONENT_LAWYER_NAME,
                scenario_type="CIA",
                court_role="appellee",
                scenario_data=appellee_lawyer_data,
                long_term_memory_path=opponent_ltm_path,
                model_type=self._resolve_agent_model_name("opponent_lawyer"),
            )
        else:
            appellant_lawyer = LawyerAgent(
                agent_id="opponent_lawyer",
                name=DEFAULT_OPPONENT_LAWYER_NAME,
                scenario_type="CIA",
                court_role="appellant",
                scenario_data=appellant_lawyer_data,
                long_term_memory_path=opponent_ltm_path,
                model_type=self._resolve_agent_model_name("opponent_lawyer"),
            )
            
            appellee_lawyer = LawyerAgent(
                agent_id="lawyer",
                name=DEFAULT_LAWYER_NAME,
                scenario_type="CIA",
                court_role="appellee",
                scenario_data=appellee_lawyer_data,
                long_term_memory_path=lawyer_ltm_path,
                model_type=self._resolve_agent_model_name("lawyer"),
            )
        
        self._configure_stage_tools(
            "CIA",
            {
                "judge": appeal_judge,
                "appellant": appellant_client,
                "appellee": appellee_client,
                "appellant_lawyer": appellant_lawyer,
                "appellee_lawyer": appellee_lawyer,
            },
        )
        self._bind_stage_trace_agents(
            "CIA",
            {
                "appeal_judge": appeal_judge,
                "appellant": appellant_client,
                "appellee": appellee_client,
                "lawyer": appellee_lawyer if not is_appellant else appellant_lawyer,
                "opponent_lawyer": appellant_lawyer if not is_appellant else appellee_lawyer,
            },
        )

        # 8. Get court opinion for reference
        court_finding = second_instance.get("court_finding", "") or second_instance.get("court_findings", "")
        court_opinion = second_instance.get("court_opinion", "") or second_instance.get("judgment", "")
        
        # 9. Execute scenario
        output_path = os.path.join(self.output_dir, "CIA_result.json")
        scenario = AppealCourtInvestigationScenario(
            judge_agent=appeal_judge,
            appellant_agent=appellant_client,
            appellee_agent=appellee_client,
            appellant_lawyer_agent=appellant_lawyer,
            appellee_lawyer_agent=appellee_lawyer,
            appellant_witnesses=party_witnesses if is_appellant else [],
            appellee_witnesses=party_witnesses if not is_appellant else [],
            court_finding=court_finding,
            court_opinion=court_opinion,
            max_debate_rounds=4,
            output_path=output_path,
            verbose=SCENARIO_VERBOSE,
            trace_recorder=self.agent_trace_recorder,
            trace_stage_code="CIA",
            trace_stage_key="CIA",
        )
        
        # Collect agent prompts before execution
        self._collect_agent_prompts("CIA", appeal_judge, appellant_client, appellee_client, appellant_lawyer, appellee_lawyer)
        
        result = scenario.execute()
        self._export_stage_agent_traces("CIA", result)

        # CIA is the terminal hearing stage. Once the second-instance trial ends,
        # do not trigger another round of long-term memory extraction or stage-summary
        # generation.
        return result
    
    def _format_evidence(self, evidence_dict: Dict[str, Any]) -> List[str]:
        """Format evidence dictionary to list of strings.
        
        Args:
            evidence_dict: Evidence dictionary from dataset
            
        Returns:
            List of formatted evidence strings
        """
        if not evidence_dict:
            return []
        
        result = []
        for key, value in evidence_dict.items():
            if isinstance(value, dict):
                evidence_text = value.get("evidence", "")
                if evidence_text:
                    result.append(evidence_text)
            elif isinstance(value, str):
                result.append(value)
        return result
    
    def _extract_disputes(self, evidence_dict: Dict[str, Any]) -> str:
        """Extract dispute opinions from evidence dictionary.
        
        Args:
            evidence_dict: Evidence dictionary from dataset
            
        Returns:
            Formatted dispute string
        """
        if not evidence_dict:
            return ""
        
        disputes = []
        for key, value in evidence_dict.items():
            if isinstance(value, dict):
                dispute = value.get("dispute", "")
                if dispute:
                    disputes.append(dispute)
        return "\n".join(disputes)

