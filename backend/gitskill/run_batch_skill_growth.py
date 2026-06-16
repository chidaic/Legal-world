from __future__ import annotations

import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from batch_tools.common import batch_timestamp, load_project_env
from gitskill.pipeline import BatchSkillGrowthConfig, run_batch_skill_growth


load_project_env()


def _find_latest_cases_root(skill_owner_id: str) -> Path | None:
    lawyer_root = BACKEND_DIR / "gitskill_data" / "lawyers" / skill_owner_id
    full_flow_root = lawyer_root / "full_flow_runs"
    if not full_flow_root.exists():
        return None

    for item in sorted(full_flow_root.iterdir(), reverse=True):
        if not item.is_dir():
            continue
        for dirname in ("case_runs", "."):
            candidate = item if dirname == "." else item / dirname
            if candidate.exists() and any(child.is_dir() and child.name.startswith("case_") for child in candidate.iterdir()):
                return candidate
    return None


def _build_lawyer_workspace_paths(skill_owner_id: str, run_tag: str) -> dict[str, str]:
    lawyer_root = BACKEND_DIR / "gitskill_data" / "lawyers" / skill_owner_id
    run_root = lawyer_root / "skill_growth_runs" / run_tag
    return {
        "lawyer_workspace_root": str(lawyer_root),
        "run_root": str(run_root),
        "manifest_path": str(run_root / "skill_growth_manifest.json"),
        "reflection_root": str(run_root / "reflections"),
        "private_skill_root": str(lawyer_root / "skills" / "private"),
    }


def _default_main_skill_root() -> str:
    return str(
        Path(os.environ.get("SIMLAW_MAIN_SKILL_ROOT") or BACKEND_DIR / "legal-skillhub" / "public")
        .expanduser()
        .resolve()
    )


DEFAULT_SKILL_OWNER_ID = "gitskill_test_lawyer"
DEFAULT_RUN_TAG = batch_timestamp()
DEFAULT_BATCH_ROOT = _find_latest_cases_root(DEFAULT_SKILL_OWNER_ID)


BATCH_SKILL_GROWTH_CONFIG = {
    "batch_root": str(DEFAULT_BATCH_ROOT) if DEFAULT_BATCH_ROOT else "",
    "skill_owner_id": DEFAULT_SKILL_OWNER_ID,
    "main_skill_root": _default_main_skill_root(),
    "model_name": None,
    "step_timeout_seconds": 420,
    "delete_final_reflection_on_success": False,
    "require_evaluation_summary": True,
    "run_tag": DEFAULT_RUN_TAG,
}
BATCH_SKILL_GROWTH_CONFIG.update(
    _build_lawyer_workspace_paths(
        BATCH_SKILL_GROWTH_CONFIG["skill_owner_id"],
        BATCH_SKILL_GROWTH_CONFIG["run_tag"],
    )
)


def _normalize_batch_root(path_value: str) -> Path:
    batch_root = Path(path_value).expanduser()
    if not batch_root.is_absolute():
        batch_root = (PROJECT_ROOT / batch_root).resolve()
    else:
        batch_root = batch_root.resolve()
    return batch_root


def _print_embedded_config_summary() -> None:
    print("=" * 80)
    print("Batch GitSkill reflection / skill growth")
    print(f"batch_root: {BATCH_SKILL_GROWTH_CONFIG['batch_root']}")
    print(f"skill_owner_id: {BATCH_SKILL_GROWTH_CONFIG['skill_owner_id']}")
    print(f"run_tag: {BATCH_SKILL_GROWTH_CONFIG['run_tag']}")
    print(f"lawyer_workspace_root: {BATCH_SKILL_GROWTH_CONFIG['lawyer_workspace_root']}")
    print(f"run_root: {BATCH_SKILL_GROWTH_CONFIG['run_root']}")
    print(f"reflection_root: {BATCH_SKILL_GROWTH_CONFIG['reflection_root']}")
    print(f"manifest_path: {BATCH_SKILL_GROWTH_CONFIG['manifest_path']}")
    print(f"private_skill_root: {BATCH_SKILL_GROWTH_CONFIG['private_skill_root']}")
    print(f"main_skill_root: {BATCH_SKILL_GROWTH_CONFIG['main_skill_root']}")
    print(f"model_name: {BATCH_SKILL_GROWTH_CONFIG['model_name'] or '<env/default>'}")
    print(f"step_timeout_seconds: {BATCH_SKILL_GROWTH_CONFIG['step_timeout_seconds']}")
    print("=" * 80)


def run_with_embedded_config() -> None:
    if not BATCH_SKILL_GROWTH_CONFIG["batch_root"]:
        raise FileNotFoundError(
            "No batch_root was auto-detected. Set BATCH_SKILL_GROWTH_CONFIG['batch_root'] first."
        )

    normalized_batch_root = _normalize_batch_root(BATCH_SKILL_GROWTH_CONFIG["batch_root"])
    BATCH_SKILL_GROWTH_CONFIG["batch_root"] = str(normalized_batch_root)
    _print_embedded_config_summary()

    config = BatchSkillGrowthConfig(
        batch_root=BATCH_SKILL_GROWTH_CONFIG["batch_root"],
        skill_owner_id=BATCH_SKILL_GROWTH_CONFIG["skill_owner_id"],
        main_skill_root=BATCH_SKILL_GROWTH_CONFIG["main_skill_root"],
        private_skill_root=BATCH_SKILL_GROWTH_CONFIG["private_skill_root"],
        reflection_root=BATCH_SKILL_GROWTH_CONFIG["reflection_root"],
        manifest_path=BATCH_SKILL_GROWTH_CONFIG["manifest_path"],
        model_name=BATCH_SKILL_GROWTH_CONFIG["model_name"],
        step_timeout_seconds=BATCH_SKILL_GROWTH_CONFIG["step_timeout_seconds"],
        delete_final_reflection_on_success=BATCH_SKILL_GROWTH_CONFIG["delete_final_reflection_on_success"],
        require_evaluation_summary=BATCH_SKILL_GROWTH_CONFIG["require_evaluation_summary"],
    )
    manifest = run_batch_skill_growth(config)

    print("=" * 80)
    print("Batch GitSkill reflection finished")
    print(f"status: {manifest.get('status')}")
    print(f"summary: {manifest.get('summary')}")
    print(f"manifest_path: {manifest.get('manifest_path')}")
    for case_record in list(manifest.get("cases", []) or []):
        print("-" * 80)
        print(f"case_id: {case_record.get('case_id')}")
        print(f"case_cause: {case_record.get('case_cause')}")
        print(f"status: {case_record.get('status')}")
        print(f"result_path: {case_record.get('result_path')}")
        print(f"developer_trace_path: {case_record.get('developer_trace_path')}")
        print(f"skill_result: {case_record.get('skill_result')}")
        if case_record.get("error"):
            print(f"error: {case_record.get('error')}")
    print("=" * 80)


if __name__ == "__main__":
    run_with_embedded_config()
