from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.lawyer_agent import LawyerAgent
from src.utils.file_io import safe_read_json, utc_timestamp, write_json

from .prompts import (
    build_stage_one_instruction,
    build_stage_one_system_prompt,
    build_stage_three_instruction,
    build_stage_three_system_prompt,
    build_stage_two_instruction,
    build_stage_two_system_prompt,
)
from .reflection_manager import ReflectionManager, sanitize_path_component
from .runtime import apply_reflection_stage
from .skill_writer import SkillWriter
from src.tools.common import ArtifactReader


@dataclass
class SingleCaseSkillGrowthConfig:
    case_dir: str
    skill_owner_id: str
    main_skill_root: str
    private_skill_root: str
    reflection_root: str
    model_name: Optional[str] = None
    step_timeout_seconds: int = 420
    delete_final_reflection_on_success: bool = False


@dataclass
class SingleCaseSkillGrowthContext:
    case_dir: Path
    case_id: str
    case_cause: str
    case_cause_dir: str
    party_role: str
    stages_completed: list[str]
    pipeline_result: dict[str, Any]
    eval_summary: dict[str, Any]
    eval_full_report: dict[str, Any]
    lawyer_stage_summaries: list[Any]
    lawyer_long_term_memory: dict[str, Any]
    lawyer_stage_summaries_path: Path
    lawyer_long_term_memory_path: Path


def _ensure_config(config: SingleCaseSkillGrowthConfig | dict[str, Any]) -> SingleCaseSkillGrowthConfig:
    if isinstance(config, SingleCaseSkillGrowthConfig):
        return config
    return SingleCaseSkillGrowthConfig(**config)


def _require_json_dict(path: Path, label: str) -> dict[str, Any]:
    payload = safe_read_json(path, default=None)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _require_json_list(path: Path, label: str) -> list[Any]:
    payload = safe_read_json(path, default=None)
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array: {path}")
    return payload


def load_single_case_context(config: SingleCaseSkillGrowthConfig | dict[str, Any]) -> SingleCaseSkillGrowthContext:
    cfg = _ensure_config(config)
    case_dir = Path(cfg.case_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

    pipeline_result = _require_json_dict(case_dir / "pipeline_result.json", "pipeline_result.json")
    eval_summary = _require_json_dict(case_dir / "eval_result" / "summary.json", "eval_result/summary.json")
    eval_full_report = _require_json_dict(
        case_dir / "eval_result" / "eval_result_full.json",
        "eval_result/eval_result_full.json",
    )
    lawyer_stage_summaries_path = case_dir / "lawyer" / "stage_summaries.json"
    lawyer_long_term_memory_path = case_dir / "lawyer" / "long_term_memory.json"
    lawyer_stage_summaries = _require_json_list(lawyer_stage_summaries_path, "lawyer/stage_summaries.json")
    lawyer_long_term_memory = _require_json_dict(
        lawyer_long_term_memory_path,
        "lawyer/long_term_memory.json",
    )

    case_id = str(pipeline_result.get("case_id") or case_dir.name.removeprefix("case_") or "unknown_case")
    case_cause = str(pipeline_result.get("case_cause") or "unknown_case_cause")
    return SingleCaseSkillGrowthContext(
        case_dir=case_dir,
        case_id=case_id,
        case_cause=case_cause,
        case_cause_dir=sanitize_path_component(case_cause, fallback="unknown_case_cause"),
        party_role=str(pipeline_result.get("party_role") or ""),
        stages_completed=list(pipeline_result.get("stages_completed") or []),
        pipeline_result=pipeline_result,
        eval_summary=eval_summary,
        eval_full_report=eval_full_report,
        lawyer_stage_summaries=lawyer_stage_summaries,
        lawyer_long_term_memory=lawyer_long_term_memory,
        lawyer_stage_summaries_path=lawyer_stage_summaries_path,
        lawyer_long_term_memory_path=lawyer_long_term_memory_path,
    )


def _build_profile(context: SingleCaseSkillGrowthContext, config: SingleCaseSkillGrowthConfig) -> dict[str, Any]:
    return {
        "name": config.skill_owner_id,
        "seniority": "资深律师，正在进行案件复盘与私有 Skill 提炼",
        "specialty": [context.case_cause],
        "law_firm": "GitSkill Reflection",
        "background": "你当前的任务不是继续办案，而是基于完整案件产物做复盘，并决定是否沉淀 Skill。",
    }


def create_reflection_lawyer_agent(
    context: SingleCaseSkillGrowthContext,
    config: SingleCaseSkillGrowthConfig,
) -> LawyerAgent:
    return LawyerAgent(
        agent_id=config.skill_owner_id,
        name=config.skill_owner_id,
        specialty_areas=[context.case_cause],
        law_firm="GitSkill Reflection",
        long_term_memory_path=str(context.lawyer_long_term_memory_path),
        stage_summaries_path=str(context.lawyer_stage_summaries_path),
        tools=[],
        skill_dirs=[],
        enable_default_tools=False,
    )


def _skill_usage_snapshot(agent: Any) -> dict[str, Any]:
    if hasattr(agent, "get_skill_usage_report"):
        try:
            report = agent.get_skill_usage_report()
            if isinstance(report, dict):
                return report
        except Exception:
            pass
    return {
        "tool_call_count": 0,
        "skill_load_count": 0,
        "skills": [],
        "tool_calls": [],
    }


def _set_agent_step_timeout(agent: Any, timeout_seconds: int) -> None:
    if hasattr(agent, "update_runtime_step_timeout"):
        agent.update_runtime_step_timeout(timeout_seconds)


def _compact_growth_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("stage_outputs", None)
    return compact


def _build_error_result(
    *,
    context: SingleCaseSkillGrowthContext,
    cfg: SingleCaseSkillGrowthConfig,
    reflection_manager: ReflectionManager,
    main_skill_root: Path,
    private_skill_root: Path,
    reflection_deleted: bool,
    stage_outputs: dict[str, str],
    stage_skill_usage: dict[str, dict[str, Any]],
    skill_writer: SkillWriter,
    error_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": error_info,
        "case_id": context.case_id,
        "case_cause": context.case_cause,
        "case_dir": str(context.case_dir),
        "reflection_dir": str(reflection_manager.case_dir),
        "skill_owner_id": cfg.skill_owner_id,
        "main_skill_root": str(main_skill_root),
        "private_skill_root": str(private_skill_root),
        "reflection_path": str(reflection_manager.reflection_path),
        "reflection_exists": reflection_manager.exists(),
        "reflection_deleted": reflection_deleted,
        "developer_trace_path": str(reflection_manager.developer_trace_path),
        "result_path": str(reflection_manager.result_path),
        "skill_results": list(skill_writer.results),
        "skill_result": skill_writer.last_result or {"status": "none", "action": "none"},
        "stage_outputs": dict(stage_outputs),
        "stage_skill_usage": dict(stage_skill_usage),
        "config": asdict(cfg),
    }


def run_single_case_skill_growth(
    config: SingleCaseSkillGrowthConfig | dict[str, Any],
    agent_factory: Optional[Callable[[SingleCaseSkillGrowthContext, SingleCaseSkillGrowthConfig], Any]] = None,
) -> dict[str, Any]:
    cfg = _ensure_config(config)
    context = load_single_case_context(cfg)

    main_skill_root = Path(cfg.main_skill_root).resolve()
    private_skill_root = Path(cfg.private_skill_root).resolve()
    reflection_root = Path(cfg.reflection_root).resolve()
    main_skill_root.mkdir(parents=True, exist_ok=True)
    private_skill_root.mkdir(parents=True, exist_ok=True)
    reflection_root.mkdir(parents=True, exist_ok=True)

    reflection_manager = ReflectionManager(
        reflection_root=reflection_root,
        case_cause=context.case_cause,
        case_id=context.case_id,
    )
    artifact_reader = ArtifactReader(context.case_dir)
    skill_writer = SkillWriter(
        private_skill_root=private_skill_root,
        main_skill_root=main_skill_root,
        case_cause=context.case_cause,
    )
    profile = _build_profile(context, cfg)
    skill_dirs = [str(main_skill_root), str(private_skill_root)]

    stage_outputs: dict[str, str] = {}
    stage_skill_usage: dict[str, dict[str, Any]] = {}
    reflection_deleted = False
    error_info: dict[str, Any] | None = None
    agent: Any | None = None

    developer_trace = {
        "generated_at": utc_timestamp(),
        "case_id": context.case_id,
        "case_cause": context.case_cause,
        "case_dir": str(context.case_dir),
        "reflection_dir": str(reflection_manager.case_dir),
        "skill_owner_id": cfg.skill_owner_id,
        "main_skill_root": str(main_skill_root),
        "private_skill_root": str(private_skill_root),
        "reflection_path": str(reflection_manager.reflection_path),
        "developer_trace_path": str(reflection_manager.developer_trace_path),
        "result_path": str(reflection_manager.result_path),
        "skill_results": [],
        "skill_result": None,
        "stage_skill_usage": {},
        "error": None,
        "config": asdict(cfg),
    }

    def _flush_developer_trace() -> None:
        developer_trace["generated_at"] = utc_timestamp()
        developer_trace["reflection_exists"] = reflection_manager.exists()
        developer_trace["reflection_deleted"] = reflection_deleted
        developer_trace["skill_results"] = list(skill_writer.results)
        developer_trace["skill_result"] = skill_writer.last_result or {"status": "none", "action": "none"}
        developer_trace["stages_completed"] = list(stage_outputs.keys())
        developer_trace["stage_skill_usage"] = dict(stage_skill_usage)
        developer_trace["error"] = error_info
        reflection_manager.write_developer_trace(developer_trace)

    def _persist_stage_output(stage_name: str, markdown: str) -> None:
        stage_outputs[stage_name] = markdown
        reflection_manager.write_reflection(markdown)

    def _persist_result(result: dict[str, Any]) -> dict[str, Any]:
        reflection_manager.write_result(_compact_growth_result(result))
        return result

    try:
        print("\n[GitSkill] Stage 1/3: initial reflection")
        print(f"  case_id: {context.case_id}")
        print(f"  eval_summary_path: {context.case_dir / 'eval_result' / 'summary.json'}")
        print(f"  eval_full_report_path: {context.case_dir / 'eval_result' / 'eval_result_full.json'}")
        print(f"  stage_summaries_count: {len(context.lawyer_stage_summaries)}")

        factory = agent_factory or create_reflection_lawyer_agent
        agent = factory(context, cfg)
        stage_one_prompt = build_stage_one_system_prompt(profile, context.lawyer_long_term_memory)
        stage_one_instruction = build_stage_one_instruction(
            case_metadata={
                "case_id": context.case_id,
                "case_cause": context.case_cause,
                "party_role": context.party_role,
                "stages_completed": context.stages_completed,
            },
            eval_summary=context.eval_summary,
            eval_full_report=context.eval_full_report,
            stage_summaries=context.lawyer_stage_summaries,
        )
        agent.activate(
            stage_one_prompt,
            model_type=cfg.model_name,
            tools=[],
            skill_dirs=[],
            step_timeout_seconds=cfg.step_timeout_seconds,
        )
        _set_agent_step_timeout(agent, cfg.step_timeout_seconds)
        stage_one_response = str(agent.step(stage_one_instruction) or "").strip()
        if not stage_one_response:
            raise ValueError("Stage one reflection output is empty.")
        _persist_stage_output("stage_one", stage_one_response)
        stage_skill_usage["stage_one"] = _skill_usage_snapshot(agent)
        _flush_developer_trace()

        print("\n[GitSkill] Stage 2/3: artifact-backed refinement")
        print(f"  artifact_count: {len(artifact_reader.list_catalog_entries())}")
        stage_two_prompt = build_stage_two_system_prompt(profile, context.lawyer_long_term_memory)
        apply_reflection_stage(
            agent,
            system_prompt=stage_two_prompt,
            skill_dirs=skill_dirs,
            extra_tools=[artifact_reader.get_tool()],
        )
        stage_two_instruction = build_stage_two_instruction(
            reflection_markdown=reflection_manager.read_reflection(),
            artifact_catalog_markdown=artifact_reader.render_catalog(),
        )
        stage_two_response = str(agent.step(stage_two_instruction) or "").strip()
        if not stage_two_response:
            raise ValueError("Stage two reflection output is empty.")
        _persist_stage_output("stage_two", stage_two_response)
        stage_skill_usage["stage_two"] = _skill_usage_snapshot(agent)
        _flush_developer_trace()

        print("\n[GitSkill] Stage 3/3: skill decision and write-back")
        print(f"  visible_skill_dirs: {skill_dirs}")
        skill_writer.reset()
        stage_three_prompt = build_stage_three_system_prompt(profile, context.lawyer_long_term_memory)
        apply_reflection_stage(
            agent,
            system_prompt=stage_three_prompt,
            skill_dirs=skill_dirs,
            extra_tools=[skill_writer.get_tool()],
        )
        stage_three_instruction = build_stage_three_instruction(
            reflection_markdown=reflection_manager.read_reflection(),
            case_cause_dir=context.case_cause_dir,
        )
        stage_three_response = str(agent.step(stage_three_instruction) or "").strip()
        if not stage_three_response:
            raise ValueError("Stage three reflection output is empty.")
        _persist_stage_output("stage_three", stage_three_response)
        stage_skill_usage["stage_three"] = _skill_usage_snapshot(agent)
        _flush_developer_trace()

        if (
            cfg.delete_final_reflection_on_success
            and skill_writer.has_successful_results()
            and not skill_writer.has_error_results()
        ):
            reflection_deleted = reflection_manager.delete_reflection()
            _flush_developer_trace()

        skill_results = list(skill_writer.results)
        skill_result = skill_writer.last_result or {"status": "none", "action": "none"}
        return _persist_result(
            {
                "status": "completed",
                "case_id": context.case_id,
                "case_cause": context.case_cause,
                "case_dir": str(context.case_dir),
                "reflection_dir": str(reflection_manager.case_dir),
                "skill_owner_id": cfg.skill_owner_id,
                "main_skill_root": str(main_skill_root),
                "private_skill_root": str(private_skill_root),
                "reflection_path": str(reflection_manager.reflection_path),
                "reflection_exists": reflection_manager.exists(),
                "reflection_deleted": reflection_deleted,
                "developer_trace_path": str(reflection_manager.developer_trace_path),
                "result_path": str(reflection_manager.result_path),
                "skill_results": skill_results,
                "skill_result": skill_result,
                "stage_outputs": dict(stage_outputs),
                "stage_skill_usage": dict(stage_skill_usage),
                "config": asdict(cfg),
            }
        )
    except TimeoutError as exc:
        error_info = {
            "stage": "unknown",
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        if "stage_three" not in stage_outputs and "stage_two" in stage_outputs:
            error_info["stage"] = "stage_three"
        elif "stage_two" not in stage_outputs and "stage_one" in stage_outputs:
            error_info["stage"] = "stage_two"
        elif "stage_one" not in stage_outputs:
            error_info["stage"] = "stage_one"
        _flush_developer_trace()
        return _persist_result(
            _build_error_result(
                context=context,
                cfg=cfg,
                reflection_manager=reflection_manager,
                main_skill_root=main_skill_root,
                private_skill_root=private_skill_root,
                reflection_deleted=reflection_deleted,
                stage_outputs=stage_outputs,
                stage_skill_usage=stage_skill_usage,
                skill_writer=skill_writer,
                error_info=error_info,
            )
        )
    except Exception as exc:
        error_info = {
            "stage": "unknown",
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        _flush_developer_trace()
        return _persist_result(
            _build_error_result(
                context=context,
                cfg=cfg,
                reflection_manager=reflection_manager,
                main_skill_root=main_skill_root,
                private_skill_root=private_skill_root,
                reflection_deleted=reflection_deleted,
                stage_outputs=stage_outputs,
                stage_skill_usage=stage_skill_usage,
                skill_writer=skill_writer,
                error_info=error_info,
            )
        )
    finally:
        _flush_developer_trace()
        if agent is not None and hasattr(agent, "deactivate"):
            try:
                agent.deactivate()
            except Exception:
                pass
