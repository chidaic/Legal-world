from __future__ import annotations

from typing import Any


def extract_second_instance_witness_entries(new_evidence: Any, *, side: str) -> list[str]:
    if not isinstance(new_evidence, dict):
        return []

    side_key = "appellant_evidence" if str(side or "").strip() == "appellant" else "appellee_evidence"
    evidence_payload = new_evidence.get(side_key, {})
    if not isinstance(evidence_payload, dict):
        return []

    witness_entries: list[str] = []
    for evidence_item in evidence_payload.values():
        if not isinstance(evidence_item, dict):
            continue

        witness_name = str(
            evidence_item.get("witness")
            or evidence_item.get("witness_name")
            or evidence_item.get("证人")
            or ""
        ).strip()
        relation = str(
            evidence_item.get("relation")
            or evidence_item.get("witness_relation")
            or evidence_item.get("关系")
            or ""
        ).strip()
        testimony = str(
            evidence_item.get("testimony")
            or evidence_item.get("witness_testimony")
            or evidence_item.get("证言")
            or ""
        ).strip()

        if not witness_name and not testimony:
            continue

        parts = [witness_name or "证人"]
        if relation:
            parts.append(relation)
        if testimony:
            parts.append(testimony)
        witness_entries.append("｜".join(parts))

    return witness_entries
