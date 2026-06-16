"""Helpers for marking player-lawyer responsibility dialogue."""

from __future__ import annotations

from typing import Any


RESPONSIBILITY_STAGES = {"PLC", "CD", "AD", "AR", "CI", "CIA"}
RESPONSIBILITY_ROLES_BY_STAGE = {
    "PLC": {"lawyer", "plaintiff_lawyer"},
    "CD": {"lawyer", "plaintiff_lawyer"},
    "AD": {"lawyer", "plaintiff_lawyer", "appellant_lawyer"},
    "AR": {"lawyer", "plaintiff_lawyer", "appellee_lawyer"},
    "CI": {"plaintiff_lawyer"},
    "CIA": {"appellant_lawyer", "appellee_lawyer"},
}


def build_player_responsibility_marker(
    *,
    role: str,
    stage: str,
    player_lawyer_enabled: bool,
    ai_surrogate_enabled: bool,
    content: str = "",
) -> dict[str, Any] | None:
    """Return optional UI marker metadata for player-responsibility dialogue."""
    if not player_lawyer_enabled:
        return None

    normalized_stage = str(stage or "").strip().upper()
    normalized_role = str(role or "").strip().lower()
    if normalized_stage not in RESPONSIBILITY_STAGES:
        return None
    if normalized_role not in RESPONSIBILITY_ROLES_BY_STAGE.get(normalized_stage, set()):
        return None
    if normalized_stage == "PLC" and _looks_like_consultation_auto_opening(content):
        return None

    label = "玩家职责" if ai_surrogate_enabled else "纳入评价"
    reason = "当前为 AI 代跑玩家律师发言" if ai_surrogate_enabled else "当前为玩家律师提交/发言"
    return {
        "player_responsibility": True,
        "evaluation_marker_label": label,
        "evaluation_marker_reason": reason,
    }


def _looks_like_consultation_auto_opening(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    opening_markers = ("您好，我是", "请您先说一下", "请您说一下", "想咨询的问题")
    return any(marker in text for marker in opening_markers) and len(text) <= 80


__all__ = ["build_player_responsibility_marker"]
