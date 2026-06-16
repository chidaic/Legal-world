"""Runtime guards shared by legal document drafting scenarios."""

from __future__ import annotations

import re
from typing import Any

from ..tools.legal import extract_document_drafting_tool_payload, get_document_type_for_scenario

DRAFTING_MAX_TURNS = 15


_LAWYER_WAITING_RE = re.compile(
    r"(?:等你|等您|拿到|收到|送过来|找[齐到出]|整理好|补充|联系我|随时联系|保持联系|"
    r"回见|路上|框架|定稿|核对|凭证|材料|证据)"
)
_CLIENT_WAITING_RE = re.compile(
    r"(?:回见|联系你|联系您|送过去|送过来|找[齐到出]|整理好|翻找|翻箱倒柜|"
    r"这两天|尽快|保持联系|凭证|材料|证据|路上.*注意|多费心)"
)
_DOCUMENT_TITLE_RE = re.compile(r"民事(?:起诉状|答辩状|上诉状|上诉答辩状)")


def has_document_payload(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(
        str(payload.get("document_text", "") or "").strip()
        or str(payload.get("pdf_path", "") or "").strip()
    )


def capture_drafting_tool_payload(lawyer: Any, *, scenario_type: str) -> dict[str, str]:
    payload = extract_document_drafting_tool_payload(
        list(getattr(lawyer, "_last_tool_call_records", []) or []),
        document_type=get_document_type_for_scenario(scenario_type),
    )
    return payload if has_document_payload(payload) else {}


def missing_document_error(
    *,
    scenario_type: str,
    document_label: str,
    finish_reason: str,
    turn_count: int,
) -> RuntimeError:
    return RuntimeError(
        f"{scenario_type} drafting failed: no complete {document_label} body was produced "
        f"(finish_reason={finish_reason}, turn_count={turn_count})."
    )


def is_stalled_drafting_dialogue(
    *,
    party_message: str,
    lawyer_response: str,
    turn_count: int,
) -> bool:
    """Detect a drafting conversation that has degraded into waiting/farewell loops."""

    if int(turn_count or 0) < 1:
        return False
    party_text = str(party_message or "").strip()
    lawyer_text = str(lawyer_response or "").strip()
    if not party_text or not lawyer_text:
        return False
    if _DOCUMENT_TITLE_RE.search(party_text) or _DOCUMENT_TITLE_RE.search(lawyer_text):
        return False
    return bool(_LAWYER_WAITING_RE.search(lawyer_text) and _CLIENT_WAITING_RE.search(party_text))


def build_forced_document_prompt(
    *,
    scenario_type: str,
    document_title: str,
    end_marker: str,
) -> str:
    return (
        f"【系统纠偏：{scenario_type} 文书起草必须立即收口】\n"
        "你刚才和当事人的对话已经进入等待材料、保持联系或道别循环。"
        "不得继续寒暄、道别、要求当事人线下补材料，不能再回复“等你找到材料再说”。\n"
        f"现在必须基于当前系统提示词、长期记忆、既有案情、诉讼请求、证据清单和本阶段对话，"
        f"直接输出完整《{document_title}》正文。"
        "缺失的精确日期、门牌号、利息金额或证据细节，可以在正文中写明“待补充”“以银行凭证核算为准”"
        "或“以立案后依法核实为准”，但不得因此拒绝成稿。\n"
        f"正文第一行必须是“{document_title}”。正文末尾必须立刻紧跟“{end_marker}”。"
        "除完整文书正文外，不要输出解释、总结、PDF路径或工具调用说明。"
    )
