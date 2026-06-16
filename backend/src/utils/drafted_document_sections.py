"""Helpers for extracting prompt fields from drafted legal documents.

These helpers intentionally use lightweight parsing instead of LLMs so that
downstream drafting prompts can reuse the *actual drafted document* from the
previous stage rather than falling back to dataset ground truth.
"""

from __future__ import annotations

from typing import Any, Iterable


def _normalize_document_text(document: str) -> str:
    return str(document or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def resolve_stage_document_text(
    stage_result: dict[str, Any] | None,
    *fields: str,
) -> str:
    payload = stage_result if isinstance(stage_result, dict) else {}

    drafted_payload = payload.get("drafted_document_payload", {})
    if isinstance(drafted_payload, dict):
        drafted_text = _normalize_document_text(str(drafted_payload.get("document_text", "") or ""))
        if drafted_text:
            return drafted_text

    for field in fields:
        text = _normalize_document_text(str(payload.get(field, "") or ""))
        if text:
            return text

    top_level_text = _normalize_document_text(str(payload.get("document_text", "") or ""))
    if top_level_text:
        return top_level_text
    return ""


def _find_first_label(text: str, labels: Iterable[str], *, start: int = 0) -> tuple[int, str]:
    best_index = -1
    best_label = ""
    for label in labels:
        index = text.find(label, start)
        if index == -1:
            continue
        if best_index == -1 or index < best_index:
            best_index = index
            best_label = label
    return best_index, best_label


def _extract_section(text: str, start_labels: Iterable[str], end_labels: Iterable[str]) -> str:
    start_index, start_label = _find_first_label(text, start_labels)
    if start_index == -1:
        return ""

    content_start = start_index + len(start_label)
    end_index = len(text)
    for label in end_labels:
        candidate_index = text.find(label, content_start)
        if candidate_index != -1 and candidate_index < end_index:
            end_index = candidate_index

    return text[content_start:end_index].strip().replace("【起草结束】", "").strip()


def extract_complaint_prompt_fields(document: str) -> dict[str, str]:
    text = _normalize_document_text(document)
    if not text:
        return {
            "claims": "",
            "facts_and_reasons": "",
            "evidence": "",
        }

    claims = _extract_section(
        text,
        start_labels=("诉讼请求：", "诉讼请求:", "上诉请求：", "上诉请求:"),
        end_labels=(
            "事实和理由：",
            "事实和理由:",
            "事实与理由：",
            "事实与理由:",
            "证据和证据来源，证人姓名和住所：",
            "证据和证据来源，证人姓名和住所:",
            "证据和证据来源、证人姓名和住所：",
            "证据和证据来源、证人姓名和住所:",
            "证据：",
            "证据:",
            "此致",
        ),
    )
    facts_and_reasons = _extract_section(
        text,
        start_labels=("事实和理由：", "事实和理由:", "事实与理由：", "事实与理由:"),
        end_labels=(
            "证据和证据来源，证人姓名和住所：",
            "证据和证据来源，证人姓名和住所:",
            "证据和证据来源、证人姓名和住所：",
            "证据和证据来源、证人姓名和住所:",
            "证据：",
            "证据:",
            "此致",
        ),
    )
    evidence = _extract_section(
        text,
        start_labels=(
            "证据和证据来源，证人姓名和住所：",
            "证据和证据来源，证人姓名和住所:",
            "证据和证据来源、证人姓名和住所：",
            "证据和证据来源、证人姓名和住所:",
            "证据：",
            "证据:",
        ),
        end_labels=("此致",),
    )
    return {
        "claims": claims,
        "facts_and_reasons": facts_and_reasons,
        "evidence": evidence,
    }


def extract_appeal_prompt_fields(document: str) -> dict[str, str]:
    text = _normalize_document_text(document)
    if not text:
        return {
            "appeal_claims": "",
            "appeal_reasons": "",
        }

    appeal_claims = _extract_section(
        text,
        start_labels=("上诉请求：", "上诉请求:", "诉讼请求：", "诉讼请求:"),
        end_labels=("事实和理由：", "事实和理由:", "事实与理由：", "事实与理由:", "此致"),
    )
    appeal_reasons = _extract_section(
        text,
        start_labels=("事实和理由：", "事实和理由:", "事实与理由：", "事实与理由:"),
        end_labels=("此致",),
    )
    return {
        "appeal_claims": appeal_claims,
        "appeal_reasons": appeal_reasons,
    }
