"""Infer the furthest safely completed case state from persisted artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATE_PROGRESS_ORDER = [
    "空闲",
    "等待前台接待",
    "原告咨询中",
    "起诉状起草中",
    "起诉状已递交",
    "等待被告",
    "被告已传唤",
    "被告咨询中",
    "答辩状起草中",
    "答辩状已递交",
    "等待一审开庭",
    "一审庭审中",
    "一审判决",
    "上诉决策中",
    "上诉状起草中",
    "上诉状已递交",
    "等待上诉答辩",
    "上诉答辩状起草中",
    "上诉答辩状已递交",
    "等待二审开庭",
    "二审庭审中",
    "终审判决",
    "已结案",
]

_STATE_RANK = {state: idx for idx, state in enumerate(STATE_PROGRESS_ORDER)}


def normalize_case_id(case_id: Any) -> str:
    """Normalize raw case id into ``case_x`` form."""
    case_key = str(case_id or "").strip()
    if not case_key:
        return ""
    return case_key if case_key.startswith("case_") else f"case_{case_key}"


def _case_output_dir(base_dir: str | Path, case_id: Any) -> Path:
    return Path(base_dir) / "output" / normalize_case_id(case_id)


def _state_rank(state: str) -> int:
    return _STATE_RANK.get(str(state or "").strip(), -1)


def normalize_case_state(raw_state: Any, default: str = "空闲") -> str:
    state = str(raw_state or "").strip()
    if state in _STATE_RANK:
        return state
    return default


def infer_case_state_from_artifacts(base_dir: str | Path, config: dict[str, Any]) -> str:
    """Infer the most advanced recoverable state from outputs and summaries."""
    current_state = normalize_case_state(config.get("case_state", "空闲"))
    party_role = str(config.get("party_role", "plaintiff") or "plaintiff").lower()
    output_dir = _case_output_dir(base_dir, config.get("case_id", ""))

    def stage_done(stage_name: str) -> bool:
        del stage_name
        return False

    candidates = [current_state]

    def add_candidate(state: str) -> None:
        if state:
            candidates.append(state)

    if (output_dir / "FINAL_VERDICT_result.json").exists() or (output_dir / "CIA_result.json").exists():
        add_candidate("终审判决")
    elif (output_dir / "AR_result.json").exists():
        add_candidate("等待二审开庭")
    else:
        if stage_done("上诉答辩状起草"):
            add_candidate("上诉答辩状起草中")

        if (output_dir / "AD_result.json").exists():
            add_candidate("上诉状已递交")
        elif stage_done("上诉状起草"):
            add_candidate("上诉状起草中")

    if (output_dir / "CI_result.json").exists():
        add_candidate("一审判决")
    elif (output_dir / "DD_result.json").exists():
        add_candidate("等待一审开庭")

    if (output_dir / "DD_result.json").exists():
        add_candidate("答辩状已递交")
    elif party_role == "defendant" and stage_done("答辩状起草"):
        add_candidate("答辩状起草中")

    if (output_dir / "CD_result.json").exists():
        add_candidate("起诉状已递交")
    elif party_role == "plaintiff" and stage_done("起诉状起草"):
        add_candidate("起诉状起草中")

    if party_role == "defendant" and stage_done("法律咨询"):
        add_candidate("答辩状起草中")
    elif party_role == "plaintiff" and stage_done("法律咨询"):
        add_candidate("起诉状起草中")

    return max(candidates, key=_state_rank)
