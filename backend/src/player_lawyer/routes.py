"""REST & WebSocket routes for the player-lawyer subsystem.

This module provides a FastAPI APIRouter that mounts under
``/api/sandbox/player-lawyer/``.  The routes let the frontend:

1. **List** pending player-lawyer input requests.
2. **Submit** a response to a pending request.
3. **Cancel** a pending request.
4. **Get** the current player mode status.

Integration
───────────
Import ``router`` and include it in the main FastAPI app:

    from src.player_lawyer.routes import router as player_lawyer_router
    app.include_router(player_lawyer_router)

The router depends on a per-sandbox ``PlayerInputGateway`` that is
lazily attached to ``SandboxRuntimeContext`` when
``SIMLAW_PLAYER_LAWYER_MODE=plaintiff`` is set.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import is_player_plaintiff_mode
from .input_gateway import PlayerInputGateway

logger = logging.getLogger(__name__)

# ── Pydantic request bodies ──────────────────────────────────────

class SubmitResponseBody(BaseModel):
    request_id: str
    message: str = ""
    original_message: str = ""
    polished_message: str = ""
    final_message: str = ""
    hint_ids: list[str] = Field(default_factory=list)
    used_ai_polish: bool = False


class PolishResponseBody(BaseModel):
    request_id: str
    original_message: str
    hint_ids: list[str] = Field(default_factory=list)


class DraftResponseBody(BaseModel):
    request_id: str
    hint_ids: list[str] = Field(default_factory=list)


class CancelRequestBody(BaseModel):
    request_id: str


# ── Router ────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/api/sandbox/player-lawyer",
    tags=["player-lawyer"],
)


# ── Dependency helpers ────────────────────────────────────────────
# These are thin wrappers that will be resolved at mount-time.
# The actual ``get_gateway_for_user`` callable is injected from
# ``ws_server.py`` after the router is included.

_gateway_provider: Callable[[Request], PlayerInputGateway] | None = None
_status_provider: Callable[[Request], dict[str, Any]] | None = None
_response_assist_provider: Callable[[Request], Any] | None = None


def set_gateway_provider(fn: Callable[[Request], PlayerInputGateway]) -> None:
    """Called once from ws_server.py to inject the dependency resolver."""
    global _gateway_provider
    _gateway_provider = fn


def set_status_provider(fn: Callable[[Request], dict[str, Any]]) -> None:
    """Called once from ws_server.py to inject runtime-aware status."""
    global _status_provider
    _status_provider = fn


def set_response_assist_provider(fn: Callable[[Request], Any]) -> None:
    """Called once from ws_server.py to inject the response assist service."""
    global _response_assist_provider
    _response_assist_provider = fn


def _require_gateway(request: Request) -> PlayerInputGateway:
    if _gateway_provider is None:
        raise HTTPException(
            status_code=503,
            detail="Player-lawyer subsystem not initialized",
        )
    return _gateway_provider(request)


def _get_response_assist_service(request: Request) -> Any | None:
    if _response_assist_provider is None:
        return None
    return _response_assist_provider(request)


def _player_lawyer_status_payload(request: Request) -> dict[str, Any]:
    if _status_provider is not None:
        return _status_provider(request)
    return {
        "player_mode": "plaintiff" if is_player_plaintiff_mode() else "off",
        "enabled": is_player_plaintiff_mode(),
    }


def _pending_payload(request: Request, *, case_id: str | None = None) -> dict[str, Any]:
    gw = _require_gateway(request)
    pending = gw.list_pending(case_id=case_id)
    return {
        "pending": [req.to_dict() for req in pending],
        "count": len(pending),
    }


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/status")
async def player_lawyer_status(request: Request):
    """Return current player-lawyer mode status."""
    return _player_lawyer_status_payload(request)


@router.get("/runtime")
async def player_lawyer_runtime(
    request: Request,
    case_id: str | None = None,
):
    """Return player-lawyer mode status and pending requests in one request."""
    return {
        **_player_lawyer_status_payload(request),
        **_pending_payload(request, case_id=case_id),
    }


@router.get("/pending")
async def list_pending_requests(
    request: Request,
    case_id: str | None = None,
):
    """List all pending player-lawyer input requests."""
    return _pending_payload(request, case_id=case_id)


@router.post("/respond")
@router.post("/submit")
async def submit_response(request: Request, body: SubmitResponseBody):
    """Submit a player response to a pending input request."""
    gw = _require_gateway(request)
    final_message = str(body.final_message or body.message or "").strip()
    if not final_message:
        raise HTTPException(status_code=400, detail="message is required.")
    try:
        req = gw.resolve(body.request_id, final_message)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    assist_payload = None
    service = _get_response_assist_service(request)
    if service is not None:
        try:
            assist = service.record_submission(
                request_id=req.request_id,
                sandbox_id=req.sandbox_id,
                case_id=req.case_id,
                stage=req.stage,
                role=req.role,
                speaker_label=req.speaker_label,
                prompt=req.prompt,
                context_summary=req.context_summary,
                original_message=body.original_message,
                polished_message=body.polished_message,
                final_message=final_message,
                hint_ids=body.hint_ids,
                used_ai_polish=body.used_ai_polish,
            )
            assist_payload = assist.to_dict()
        except (ValueError, RuntimeError) as exc:
            logger.warning("[PlayerLawyer] Failed to record response assist: %s", exc)
    ledger = getattr(gw, "ledger", None)
    if ledger is not None:
        try:
            ledger.record_submission(
                case_id=req.case_id,
                request_id=req.request_id,
                stage=req.stage,
                submission_type="dialogue",
                original_message=body.original_message,
                polished_message=body.polished_message,
                final_message=final_message,
                hint_ids=body.hint_ids,
                used_ai_polish=body.used_ai_polish,
                submitted_at=req.resolved_at or "",
            )
        except Exception as exc:
            logger.warning("[PlayerLawyer] Failed to record ledger submission: %s", exc)
    return {"success": True, "request": req.to_dict(), "assist": assist_payload}


@router.post("/polish-response")
async def polish_response(request: Request, body: PolishResponseBody):
    """Create an AI-polished editable version of a player text response."""
    gw = _require_gateway(request)
    req = gw.get_request(body.request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {body.request_id}")
    if not str(body.original_message or "").strip():
        raise HTTPException(status_code=400, detail="original_message is required.")
    service = _get_response_assist_service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Response assist service not initialized")
    try:
        assist = service.polish_response(
            request_id=req.request_id,
            sandbox_id=req.sandbox_id,
            case_id=req.case_id,
            stage=req.stage,
            role=req.role,
            speaker_label=req.speaker_label,
            prompt=req.prompt,
            context_summary=req.context_summary,
            original_message=body.original_message,
            hint_ids=body.hint_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"success": True, "assist": assist.to_dict()}


@router.post("/draft-response")
async def draft_response(request: Request, body: DraftResponseBody):
    """Create an AI-generated editable response for quick flow-through testing."""
    gw = _require_gateway(request)
    req = gw.get_request(body.request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Unknown request_id: {body.request_id}")
    service = _get_response_assist_service(request)
    if service is None:
        raise HTTPException(status_code=503, detail="Response assist service not initialized")
    try:
        assist = service.draft_response(
            request_id=req.request_id,
            sandbox_id=req.sandbox_id,
            case_id=req.case_id,
            stage=req.stage,
            role=req.role,
            speaker_label=req.speaker_label,
            prompt=req.prompt,
            context_summary=req.context_summary,
            hint_ids=body.hint_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"success": True, "assist": assist.to_dict()}


@router.post("/cancel")
async def cancel_request(request: Request, body: CancelRequestBody):
    """Cancel a pending input request."""
    gw = _require_gateway(request)
    try:
        req = gw.cancel(body.request_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True, "request": req.to_dict()}


@router.post("/cancel-all")
async def cancel_all_requests(
    request: Request,
    case_id: str | None = None,
):
    """Cancel all pending requests, optionally filtered by case_id."""
    gw = _require_gateway(request)
    if case_id:
        cancelled = gw.cancel_all_for_case(case_id)
    else:
        # Cancel everything pending
        cancelled = []
        for req in gw.list_pending():
            try:
                gw.cancel(req.request_id)
                cancelled.append(req)
            except (ValueError, RuntimeError):
                pass
    return {
        "success": True,
        "cancelled_count": len(cancelled),
    }
