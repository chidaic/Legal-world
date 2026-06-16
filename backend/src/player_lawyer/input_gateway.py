"""Thread-safe input gateway for player-lawyer turns.

The gateway is the central coordination point: scenario threads create
pending requests and block until the player responds (or the request
expires); REST/WebSocket handlers resolve requests from the outside.

Design notes
────────────
- Uses ``threading.Condition`` so it works from both sync and async
  callers (scenario threads are typically sync ``agent.step()`` calls
  wrapped in ``asyncio.to_thread``).
- Each sandbox gets its own ``PlayerInputGateway`` instance so
  concurrent sandboxes are fully isolated.
- Requests are persisted to JSON via a pluggable ``persist_fn``
  callback so debug inspection and restart recovery are possible.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .models import PlayerLawyerRequest, PlayerRequestStatus

logger = logging.getLogger(__name__)

# Default timeout for blocking waits (seconds). Human player turns should not
# fail during ordinary reading/writing time, so the default is intentionally long.
DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60
PLAYER_TIMEOUT_ENV = "SIMLAW_PLAYER_LAWYER_TIMEOUT_SECONDS"


def player_lawyer_timeout_seconds(default: float = DEFAULT_TIMEOUT_SECONDS) -> float:
    """Return the configured player-lawyer wait timeout in seconds."""
    raw_value = os.environ.get(PLAYER_TIMEOUT_ENV)
    if raw_value is None:
        return float(default)
    try:
        parsed = float(raw_value.strip())
    except (TypeError, ValueError):
        logger.warning(
            "[PlayerGateway] Invalid %s=%r; using default %ss",
            PLAYER_TIMEOUT_ENV,
            raw_value,
            default,
        )
        return float(default)
    return max(parsed, 1.0)


class PlayerInputGateway:
    """Thread-safe store + wait/notify hub for player-lawyer requests."""

    def __init__(
        self,
        sandbox_id: int = 0,
        persist_fn: Optional[Callable[[PlayerLawyerRequest], None]] = None,
        timeout_seconds: Optional[float] = None,
        ledger: Any | None = None,
        storage_root: Path | str | None = None,
    ) -> None:
        self.sandbox_id = sandbox_id
        self._persist_fn = persist_fn
        self.ledger = ledger
        self.storage_root = Path(storage_root) if storage_root is not None else None
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else player_lawyer_timeout_seconds()
        )

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # request_id -> PlayerLawyerRequest
        self._requests: dict[str, PlayerLawyerRequest] = {}

    # ── Create ────────────────────────────────────────────────────

    def create_request(
        self,
        *,
        case_id: str,
        stage: str,
        role: str = "plaintiff_lawyer",
        speaker_label: str = "",
        prompt: str = "",
        context_summary: str = "",
    ) -> PlayerLawyerRequest:
        """Create a new pending request and persist it."""
        req = PlayerLawyerRequest(
            sandbox_id=self.sandbox_id,
            case_id=case_id,
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
        )
        with self._lock:
            self._requests[req.request_id] = req
        self._persist(req)
        logger.info(
            "[PlayerGateway] Created pending request %s for %s/%s",
            req.request_id,
            case_id,
            stage,
        )
        return req

    # ── Wait (blocking) ───────────────────────────────────────────

    def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> str:
        """Block until the request is resolved; return the player message.

        Raises ``TimeoutError`` if the player does not respond in time.
        Raises ``ValueError`` if the request does not exist.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        with self._condition:
            req = self._requests.get(request_id)
            if req is None:
                raise ValueError(f"Unknown request_id: {request_id}")

            deadline_reached = False
            while req.status == PlayerRequestStatus.PENDING:
                notified = self._condition.wait(timeout=effective_timeout)
                if not notified:
                    deadline_reached = True
                    break

            if req.status == PlayerRequestStatus.SUBMITTED:
                return req.message

            if deadline_reached and req.status == PlayerRequestStatus.PENDING:
                req.status = PlayerRequestStatus.EXPIRED
                req.resolved_at = datetime.utcnow().isoformat()
                self._persist(req)
                raise TimeoutError(
                    f"Player did not respond within {effective_timeout}s "
                    f"(request_id={request_id})"
                )

            # Cancelled or expired by another thread
            raise RuntimeError(
                f"Request {request_id} ended with status {req.status.value}"
            )

    # ── Resolve ───────────────────────────────────────────────────

    def resolve(self, request_id: str, message: str) -> PlayerLawyerRequest:
        """Submit the player's response and wake the waiting thread.

        Raises ``ValueError`` on unknown request_id.
        Raises ``RuntimeError`` if the request is not pending.
        """
        with self._condition:
            req = self._requests.get(request_id)
            if req is None:
                raise ValueError(f"Unknown request_id: {request_id}")
            if req.status != PlayerRequestStatus.PENDING:
                raise RuntimeError(
                    f"Cannot resolve request {request_id}: "
                    f"current status is {req.status.value}"
                )
            req.status = PlayerRequestStatus.SUBMITTED
            req.message = message
            req.resolved_at = datetime.utcnow().isoformat()
            self._condition.notify_all()

        self._persist(req)
        logger.info(
            "[PlayerGateway] Resolved request %s with %d chars",
            request_id,
            len(message),
        )
        return req

    # ── Cancel ────────────────────────────────────────────────────

    def cancel(self, request_id: str) -> PlayerLawyerRequest:
        """Cancel a single pending request."""
        with self._condition:
            req = self._requests.get(request_id)
            if req is None:
                raise ValueError(f"Unknown request_id: {request_id}")
            if req.status != PlayerRequestStatus.PENDING:
                raise RuntimeError(
                    f"Cannot cancel request {request_id}: "
                    f"current status is {req.status.value}"
                )
            req.status = PlayerRequestStatus.CANCELLED
            req.resolved_at = datetime.utcnow().isoformat()
            self._condition.notify_all()
        self._persist(req)
        return req

    def cancel_all_for_case(self, case_id: str) -> list[PlayerLawyerRequest]:
        """Cancel all pending requests for a specific case."""
        cancelled: list[PlayerLawyerRequest] = []
        with self._condition:
            for req in self._requests.values():
                if req.case_id == case_id and req.status == PlayerRequestStatus.PENDING:
                    req.status = PlayerRequestStatus.CANCELLED
                    req.resolved_at = datetime.utcnow().isoformat()
                    cancelled.append(req)
            if cancelled:
                self._condition.notify_all()
        for req in cancelled:
            self._persist(req)
        return cancelled

    def cancel_all_pending(self) -> list[PlayerLawyerRequest]:
        """Cancel every pending request in this gateway."""
        cancelled: list[PlayerLawyerRequest] = []
        with self._condition:
            for req in self._requests.values():
                if req.status == PlayerRequestStatus.PENDING:
                    req.status = PlayerRequestStatus.CANCELLED
                    req.resolved_at = datetime.utcnow().isoformat()
                    cancelled.append(req)
            if cancelled:
                self._condition.notify_all()
        for req in cancelled:
            self._persist(req)
        return cancelled

    # ── Query ─────────────────────────────────────────────────────

    def list_pending(
        self,
        case_id: Optional[str] = None,
    ) -> list[PlayerLawyerRequest]:
        """Return all pending requests, optionally filtered by case_id."""
        with self._lock:
            results = [
                req
                for req in self._requests.values()
                if req.status == PlayerRequestStatus.PENDING
            ]
        if case_id is not None:
            results = [r for r in results if r.case_id == case_id]
        return results

    def get_request(self, request_id: str) -> Optional[PlayerLawyerRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def find_reusable_request(
        self,
        *,
        case_id: str,
        stage: str,
        prompt: str,
    ) -> Optional[PlayerLawyerRequest]:
        """Find an existing pending/submitted request for the same player turn."""
        normalized_prompt = str(prompt or "").strip()
        with self._lock:
            candidates = [
                req
                for req in self._requests.values()
                if req.case_id == case_id
                and req.stage == stage
                and str(req.prompt or "").strip() == normalized_prompt
                and req.status in {PlayerRequestStatus.PENDING, PlayerRequestStatus.SUBMITTED}
            ]
        if not candidates:
            return None
        submitted = [req for req in candidates if req.status == PlayerRequestStatus.SUBMITTED]
        return sorted(submitted or candidates, key=lambda req: req.created_at)[-1]

    # ── Restore ───────────────────────────────────────────────────

    def restore_requests(self, requests: Iterable[PlayerLawyerRequest]) -> int:
        """Restore persisted unfinished requests into this gateway.

        Submitted requests are kept so a restarted scenario can consume an
        answer that was submitted after the original waiting thread disappeared.
        """
        restored = 0
        with self._condition:
            for req in requests:
                if req.status not in {PlayerRequestStatus.PENDING, PlayerRequestStatus.SUBMITTED}:
                    continue
                if not req.request_id or req.request_id in self._requests:
                    continue
                self._requests[req.request_id] = req
                restored += 1
            if restored:
                self._condition.notify_all()
        return restored

    # ── Persistence helper ────────────────────────────────────────

    def _persist(self, req: PlayerLawyerRequest) -> None:
        if self._persist_fn is not None:
            try:
                self._persist_fn(req)
            except Exception as exc:
                logger.warning(
                    "[PlayerGateway] Persist failed for %s: %s",
                    req.request_id,
                    exc,
                )


def make_json_persister(
    base_dir: Path,
    ledger: Any | None = None,
) -> Callable[[PlayerLawyerRequest], None]:
    """Return a persist_fn that writes request JSON and optional ledger rows."""

    def _persist(req: PlayerLawyerRequest) -> None:
        target_dir = base_dir / req.case_id / "_player_lawyer"
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"{req.request_id}.json"
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(req.to_dict(), fh, ensure_ascii=False, indent=2)
        if ledger is not None:
            ledger.record_responsibility_turn(
                case_id=req.case_id,
                request_id=req.request_id,
                stage=req.stage,
                role=req.role,
                speaker_label=req.speaker_label,
                prompt=req.prompt,
                context_summary=req.context_summary,
                status=req.status.value if hasattr(req.status, "value") else str(req.status),
                created_at=req.created_at,
            )

    return _persist


def _request_from_json_payload(payload: dict[str, Any]) -> PlayerLawyerRequest | None:
    try:
        status = PlayerRequestStatus(str(payload.get("status") or "pending"))
    except ValueError:
        logger.warning(
            "[PlayerGateway] Skip persisted request with invalid status: %r",
            payload.get("status"),
        )
        return None

    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        return None

    sandbox_id_raw = payload.get("sandbox_id", 0)
    try:
        sandbox_id = int(sandbox_id_raw)
    except (TypeError, ValueError):
        sandbox_id = 0

    return PlayerLawyerRequest(
        request_id=request_id,
        sandbox_id=sandbox_id,
        case_id=str(payload.get("case_id") or ""),
        stage=str(payload.get("stage") or ""),
        role=str(payload.get("role") or "plaintiff_lawyer"),
        speaker_label=str(payload.get("speaker_label") or ""),
        prompt=str(payload.get("prompt") or ""),
        context_summary=str(payload.get("context_summary") or ""),
        status=status,
        message=str(payload.get("message") or ""),
        created_at=str(payload.get("created_at") or datetime.utcnow().isoformat()),
        resolved_at=payload.get("resolved_at"),
    )


def load_json_requests(base_dir: Path) -> list[PlayerLawyerRequest]:
    """Load persisted player-lawyer request JSON files under *base_dir*."""
    requests: list[PlayerLawyerRequest] = []
    if not base_dir.exists():
        return requests

    for filepath in sorted(base_dir.glob("*/_player_lawyer/*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            logger.warning("[PlayerGateway] Failed to load %s: %s", filepath, exc)
            continue
        if not isinstance(payload, dict):
            continue
        req = _request_from_json_payload(payload)
        if req is not None:
            requests.append(req)
    return requests


def restore_json_requests(gateway: PlayerInputGateway, base_dir: Path) -> int:
    """Restore persisted pending player-lawyer requests into *gateway*."""
    return gateway.restore_requests(load_json_requests(base_dir))
