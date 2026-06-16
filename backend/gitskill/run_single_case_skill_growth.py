from __future__ import annotations

import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from batch_tools.common import batch_timestamp, load_project_env
from gitskill.pipeline import SingleCaseSkillGrowthConfig, run_single_case_skill_growth


load_project_env()


def _build_lawyer_workspace_paths(skill_owner_id: str, run_tag: str) -> dict[str, str]:
    lawyer_root = BACKEND_DIR / "gitskill_data" / "lawyers" / skill_owner_id
    run_root = lawyer_root / "skill_growth_runs" / run_tag
    return {
        "lawyer_workspace_root": str(lawyer_root),
        "run_root": str(run_root),
        "private_skill_root": str(lawyer_root / "skills" / "private"),
        "reflection_root": str(run_root / "reflections"),
    }


def _default_main_skill_root() -> str:
    return str(
        Path(os.environ.get("SIMLAW_MAIN_SKILL_ROOT") or BACKEND_DIR / "legal-skillhub" / "public")
        .expanduser()
        .resolve()
    )


def _default_case_dir() -> str:
    return os.environ.get("SIMLAW_SKILL_GROWTH_CASE_DIR", "backend/batch_runs/<case_id>").strip()


DEFAULT_SKILL_OWNER_ID = "demo_lawyer"
DEFAULT_RUN_TAG = batch_timestamp()


SINGLE_CASE_SKILL_GROWTH_CONFIG = {
    "case_dir": _default_case_dir(),
    "skill_owner_id": DEFAULT_SKILL_OWNER_ID,
    "main_skill_root": _default_main_skill_root(),
    "model_name": None,
    "step_timeout_seconds": 420,
    "delete_final_reflection_on_success": False,
    "run_tag": DEFAULT_RUN_TAG,
}
SINGLE_CASE_SKILL_GROWTH_CONFIG.update(
    _build_lawyer_workspace_paths(
        SINGLE_CASE_SKILL_GROWTH_CONFIG["skill_owner_id"],
        SINGLE_CASE_SKILL_GROWTH_CONFIG["run_tag"],
    )
)


def _normalize_case_dir(path_value: str) -> Path:
    case_dir = Path(path_value).expanduser()
    if not case_dir.is_absolute():
        case_dir = (PROJECT_ROOT / case_dir).resolve()
    else:
        case_dir = case_dir.resolve()
    return case_dir


def _print_embedded_config_summary() -> None:
    print("=" * 80)
    print("Single-case GitSkill reflection / skill growth")
    print(f"case_dir: {SINGLE_CASE_SKILL_GROWTH_CONFIG['case_dir']}")
    print(f"skill_owner_id: {SINGLE_CASE_SKILL_GROWTH_CONFIG['skill_owner_id']}")
    print(f"run_tag: {SINGLE_CASE_SKILL_GROWTH_CONFIG['run_tag']}")
    print(f"lawyer_workspace_root: {SINGLE_CASE_SKILL_GROWTH_CONFIG['lawyer_workspace_root']}")
    print(f"run_root: {SINGLE_CASE_SKILL_GROWTH_CONFIG['run_root']}")
    print(f"main_skill_root: {SINGLE_CASE_SKILL_GROWTH_CONFIG['main_skill_root']}")
    print(f"private_skill_root: {SINGLE_CASE_SKILL_GROWTH_CONFIG['private_skill_root']}")
    print(f"reflection_root: {SINGLE_CASE_SKILL_GROWTH_CONFIG['reflection_root']}")
    print(f"model_name: {SINGLE_CASE_SKILL_GROWTH_CONFIG['model_name'] or '<env/default>'}")
    print(f"step_timeout_seconds: {SINGLE_CASE_SKILL_GROWTH_CONFIG['step_timeout_seconds']}")
    print("=" * 80)


def run_with_embedded_config() -> None:
    if "<case_id>" in SINGLE_CASE_SKILL_GROWTH_CONFIG["case_dir"]:
        raise SystemExit(
            "Set SIMLAW_SKILL_GROWTH_CASE_DIR to a generated case-run directory before running "
            "backend/gitskill/run_single_case_skill_growth.py."
        )
    SINGLE_CASE_SKILL_GROWTH_CONFIG["case_dir"] = str(
        _normalize_case_dir(SINGLE_CASE_SKILL_GROWTH_CONFIG["case_dir"])
    )
    _print_embedded_config_summary()
    config = SingleCaseSkillGrowthConfig(
        case_dir=SINGLE_CASE_SKILL_GROWTH_CONFIG["case_dir"],
        skill_owner_id=SINGLE_CASE_SKILL_GROWTH_CONFIG["skill_owner_id"],
        main_skill_root=SINGLE_CASE_SKILL_GROWTH_CONFIG["main_skill_root"],
        private_skill_root=SINGLE_CASE_SKILL_GROWTH_CONFIG["private_skill_root"],
        reflection_root=SINGLE_CASE_SKILL_GROWTH_CONFIG["reflection_root"],
        model_name=SINGLE_CASE_SKILL_GROWTH_CONFIG["model_name"],
        step_timeout_seconds=SINGLE_CASE_SKILL_GROWTH_CONFIG["step_timeout_seconds"],
        delete_final_reflection_on_success=SINGLE_CASE_SKILL_GROWTH_CONFIG["delete_final_reflection_on_success"],
    )
    result = run_single_case_skill_growth(config)

    print("=" * 80)
    print("Single-case GitSkill reflection finished")
    print(f"status: {result.get('status')}")
    print(f"error: {result.get('error')}")
    print(f"case_id: {result['case_id']}")
    print(f"case_cause: {result['case_cause']}")
    print(f"reflection_exists: {result['reflection_exists']}")
    print(f"reflection_path: {result['reflection_path']}")
    print(f"developer_trace_path: {result['developer_trace_path']}")
    print(f"result_path: {result['result_path']}")
    print(f"skill_results: {result.get('skill_results')}")
    print(f"skill_result: {result['skill_result']}")
    print("=" * 80)


if __name__ == "__main__":
    run_with_embedded_config()
