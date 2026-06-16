"""Filesystem storage helpers for player-lawyer drafts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import DocumentDraft


class PlayerLawyerStorage:
    """Persist player-lawyer operational data under a sandbox storage root."""

    def __init__(self, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root).resolve()

    def case_output_dir(self, case_id: str) -> Path:
        path = self.storage_root / "output" / str(case_id or "").strip()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def player_dir(self, case_id: str) -> Path:
        path = self.case_output_dir(case_id) / "_player_lawyer"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def draft_dir(self, case_id: str) -> Path:
        path = self.player_dir(case_id) / "document_drafts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def draft_path(self, draft: DocumentDraft) -> Path:
        return self.draft_dir(draft.case_id) / f"{draft.draft_id}.json"

    def save_draft(self, draft: DocumentDraft) -> Path:
        path = self.draft_path(draft)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(draft.to_dict(), handle, ensure_ascii=False, indent=2)
        return path

    def load_draft(self, case_id: str, draft_id: str) -> DocumentDraft:
        path = self.draft_dir(case_id) / f"{draft_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Document draft not found: {draft_id}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return document_draft_from_dict(payload)

    def find_draft(self, draft_id: str) -> DocumentDraft:
        normalized = str(draft_id or "").strip()
        if not normalized:
            raise FileNotFoundError("Document draft not found: empty draft_id")
        for path in sorted((self.storage_root / "output").glob(f"*/_player_lawyer/document_drafts/{normalized}.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return document_draft_from_dict(payload)
        raise FileNotFoundError(f"Document draft not found: {draft_id}")

    def save_document_result(
        self,
        *,
        case_id: str,
        document_type: str = "complaint",
        document_text: str,
        drafted_document_payload: dict[str, Any],
        dialog_history: list[dict[str, Any]] | None = None,
        dialogue_summary: str = "",
    ) -> Path:
        spec = _document_result_spec(document_type)
        output_dir = self.case_output_dir(case_id)
        dialog_history = list(dialog_history or [])
        dialogue_text = str(dialogue_summary or "").strip() or str(document_text or "").strip()
        dialog_history.append(
            {
                "turn": len(dialog_history),
                "role": "lawyer",
                "content": dialogue_text,
            }
        )
        result = {
            "scenario_type": spec["stage"],
            "dialog_history": dialog_history,
            "turn_count": 0,
            "completed": True,
            "finish_reason": "player_confirmed",
            spec["result_field"]: str(document_text or "").strip(),
            "drafted_document_payload": drafted_document_payload,
            "pdf_path": str(drafted_document_payload.get("pdf_path", "") or ""),
        }
        path = output_dir / spec["result_filename"]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return path

    def save_cd_result(
        self,
        *,
        case_id: str,
        document_type: str = "complaint",
        document_text: str,
        drafted_document_payload: dict[str, Any],
        dialog_history: list[dict[str, Any]] | None = None,
        dialogue_summary: str = "",
    ) -> Path:
        return self.save_document_result(
            case_id=case_id,
            document_type=document_type,
            document_text=document_text,
            drafted_document_payload=drafted_document_payload,
            dialog_history=dialog_history,
            dialogue_summary=dialogue_summary,
        )


def _document_result_spec(document_type: str) -> dict[str, str]:
    normalized = str(document_type or "").strip()
    specs = {
        "complaint": {
            "stage": "CD",
            "result_field": "complaint_statement",
            "result_filename": "CD_result.json",
        },
        "defense": {
            "stage": "DD",
            "result_field": "defense_statement",
            "result_filename": "DD_result.json",
        },
        "appeal": {
            "stage": "AD",
            "result_field": "appeal_statement",
            "result_filename": "AD_result.json",
        },
        "appeal_response": {
            "stage": "AR",
            "result_field": "appeal_response_statement",
            "result_filename": "AR_result.json",
        },
    }
    if normalized not in specs:
        raise ValueError(f"Unsupported player document type: {document_type}")
    return specs[normalized]


def document_draft_from_dict(payload: dict[str, Any]) -> DocumentDraft:
    return DocumentDraft(
        draft_id=str(payload.get("draft_id", "") or ""),
        request_id=str(payload.get("request_id", "") or ""),
        sandbox_id=int(payload.get("sandbox_id") or 0),
        case_id=str(payload.get("case_id", "") or ""),
        document_type=str(payload.get("document_type", "") or ""),
        skill_id=str(payload.get("skill_id", "") or ""),
        player_prompt=str(payload.get("player_prompt", "") or ""),
        player_draft=str(payload.get("player_draft", "") or ""),
        document_text=str(payload.get("document_text", "") or ""),
        confirmed=bool(payload.get("confirmed", False)),
        finish_reason=str(payload.get("finish_reason", "") or ""),
        pdf_path=str(payload.get("pdf_path", "") or ""),
        created_at=str(payload.get("created_at", "") or ""),
        confirmed_at=payload.get("confirmed_at"),
    )


__all__ = ["PlayerLawyerStorage", "document_draft_from_dict"]
