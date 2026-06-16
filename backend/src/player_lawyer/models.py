"""Data models for the player-lawyer subsystem.

Defines the core request/response shapes used by the input gateway,
document assist service, and REST/WebSocket routes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class PlayerRequestStatus(str, Enum):
    """Lifecycle status of a single player-lawyer pending request."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class PlayerLawyerRequest:
    """A single pending request for human player input.

    Created when the scenario reaches a plaintiff-lawyer turn,
    resolved when the player submits a response or the request
    times out / is cancelled.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sandbox_id: int = 0
    case_id: str = ""
    stage: str = ""          # "LC" or "CD" in phase 1
    role: str = "plaintiff_lawyer"
    speaker_label: str = ""  # display name for the player's role
    prompt: str = ""         # previous speaker message or task description
    context_summary: str = ""  # compact case/stage context for UI
    status: PlayerRequestStatus = PlayerRequestStatus.PENDING
    message: str = ""        # player's submitted text (filled on resolve)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sandbox_id": self.sandbox_id,
            "case_id": self.case_id,
            "stage": self.stage,
            "role": self.role,
            "speaker_label": self.speaker_label,
            "prompt": self.prompt,
            "context_summary": self.context_summary,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass
class DocumentDraft:
    """An AI-assisted document draft awaiting player confirmation."""

    draft_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = ""
    sandbox_id: int = 0
    case_id: str = ""
    document_type: str = ""   # "complaint" / "CD" in phase 1
    skill_id: str = ""
    player_prompt: str = ""
    player_draft: str = ""
    document_text: str = ""   # AI-generated draft text
    confirmed: bool = False
    finish_reason: str = ""   # "player_confirmed" after confirmation
    pdf_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confirmed_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "request_id": self.request_id,
            "sandbox_id": self.sandbox_id,
            "case_id": self.case_id,
            "document_type": self.document_type,
            "skill_id": self.skill_id,
            "player_prompt": self.player_prompt,
            "player_draft": self.player_draft,
            "document_text": self.document_text,
            "confirmed": self.confirmed,
            "finish_reason": self.finish_reason,
            "pdf_path": self.pdf_path,
            "created_at": self.created_at,
            "confirmed_at": self.confirmed_at,
        }


@dataclass
class ResponseAssist:
    """AI-polish metadata for a player-lawyer text response."""

    request_id: str
    sandbox_id: int = 0
    case_id: str = ""
    stage: str = ""
    role: str = "plaintiff_lawyer"
    speaker_label: str = ""
    prompt: str = ""
    context_summary: str = ""
    hint_ids: list[str] = field(default_factory=list)
    user_original_message: str = ""
    ai_polished_message: str = ""
    final_submitted_message: str = ""
    used_ai_polish: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sandbox_id": self.sandbox_id,
            "case_id": self.case_id,
            "stage": self.stage,
            "role": self.role,
            "speaker_label": self.speaker_label,
            "prompt": self.prompt,
            "context_summary": self.context_summary,
            "hint_ids": list(self.hint_ids),
            "user_original_message": self.user_original_message,
            "ai_polished_message": self.ai_polished_message,
            "final_submitted_message": self.final_submitted_message,
            "used_ai_polish": self.used_ai_polish,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
