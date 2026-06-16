from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "llm_single_natural_person_cases_question.json"

STAGE_DISPLAY_NAMES = {
    "INIT": "初始化",
    "LC": "法律咨询",
    "CD": "起诉状起草",
    "DD": "答辩状起草",
    "CI": "民事一审",
    "SD": "二审判定",
    "AD": "上诉状起草",
    "AR": "上诉答辩状起草",
    "CIA": "二审法庭调查",
    "DRAFT": "文书起草",
    "APPEAL_DRAFT": "二审文书起草",
}

PIPELINE_STAGE_ORDER = ["LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"]


def ensure_backend_on_path() -> None:
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def load_project_env() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


def configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def utc_timestamp() -> str:
    return datetime.now().isoformat()


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_read_json(path: str | Path, default: Any = None) -> Any:
    filepath = Path(path)
    if not filepath.exists():
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: str | Path, data: Any) -> None:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, filepath)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def infer_draft_stage_code(pipeline_result: dict[str, Any]) -> str:
    stage_result = pipeline_result.get("stage_results", {}).get("DRAFT", {})
    if "complaint_statement" in stage_result:
        return "CD"
    if "defense_statement" in stage_result:
        return "DD"
    party_role = (
        pipeline_result.get("stage_output", {}).get("party_role")
        or pipeline_result.get("party_role")
        or "plaintiff"
    )
    return "CD" if party_role == "plaintiff" else "DD"


def infer_appeal_stage_code(pipeline_result: dict[str, Any]) -> str:
    stage_result = pipeline_result.get("stage_results", {}).get("APPEAL_DRAFT", {})
    if "appeal_statement" in stage_result:
        return "AD"
    if "appeal_response_statement" in stage_result:
        return "AR"
    sd_result = pipeline_result.get("stage_results", {}).get("SD", {})
    return "AD" if sd_result.get("is_appellant", True) else "AR"


def map_eval_stage_code(stage_code: str, pipeline_result: dict[str, Any]) -> str:
    if stage_code == "DRAFT":
        return infer_draft_stage_code(pipeline_result)
    if stage_code == "APPEAL_DRAFT":
        return infer_appeal_stage_code(pipeline_result)
    return stage_code


def stage_label(stage_code: str) -> str:
    return STAGE_DISPLAY_NAMES.get(stage_code, stage_code)


def actual_to_pipeline_stage(actual_stage_code: str) -> str:
    if actual_stage_code in {"CD", "DD"}:
        return "DRAFT"
    if actual_stage_code in {"AD", "AR"}:
        return "APPEAL_DRAFT"
    return actual_stage_code


def next_pipeline_stage(completed_stages: list[str]) -> str | None:
    completed = set(completed_stages)
    for stage_code in PIPELINE_STAGE_ORDER:
        if stage_code not in completed:
            return stage_code
    return None
