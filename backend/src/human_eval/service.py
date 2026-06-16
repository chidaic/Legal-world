from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from io import StringIO
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.core.models import HumanEvalAssignment, HumanEvalRating, User
from src.human_eval.materials import HumanEvalMaterials
from src.human_eval.schemas import ROLE_METRICS, SCORABLE_STAGES, STAGE_METRICS, HumanEvalRatingPayload


DEFAULT_ASSIGNMENT_BATCH_SIZE = 10


class HumanEvalService:
    def __init__(self, materials: HumanEvalMaterials | None = None) -> None:
        self.materials = materials or HumanEvalMaterials()

    def list_cases(self, session: Session, user: User) -> dict[str, Any]:
        all_cases = self._cases_with_rating_status(session=session, user=user)
        assignment = self._get_or_create_active_assignment(
            session=session,
            user=user,
            cases=all_cases,
            batch_size=DEFAULT_ASSIGNMENT_BATCH_SIZE,
        )
        assigned_case_ids = self._assignment_case_ids(assignment)
        assigned_cases = [item for item in all_cases if int(item["case_id"]) in assigned_case_ids]
        assigned_cases.sort(key=lambda item: assigned_case_ids.index(int(item["case_id"])))
        submitted_count = sum(1 for item in assigned_cases if item.get("rating_status") == "submitted")
        assignment_payload = {
            "batch_size": DEFAULT_ASSIGNMENT_BATCH_SIZE,
            "assigned_case_ids": assigned_case_ids,
            "submitted_count": submitted_count,
            "total_count": len(assigned_case_ids),
            "completed": bool(assigned_case_ids) and submitted_count >= len(assigned_case_ids),
            "batch_number": assignment.batch_number,
        }
        return {
            "assigned_cases": assigned_cases,
            "all_cases": all_cases,
            "cases": all_cases,
            "assignment": assignment_payload,
        }

    def _cases_with_rating_status(self, session: Session, user: User) -> list[dict[str, Any]]:
        ratings = {
            rating.case_id: rating
            for rating in session.scalars(
                select(HumanEvalRating).where(HumanEvalRating.user_id == user.id)
            ).all()
        }
        cases = []
        for item in self.materials.list_cases():
            case_id = int(item["case_id"])
            rating = ratings.get(case_id)
            cases.append(
                {
                    **item,
                    "rating_status": rating.status if rating else "not_started",
                    "rating_updated_at": rating.updated_at.isoformat() if rating and rating.updated_at else None,
                    "rating_submitted_at": rating.submitted_at.isoformat() if rating and rating.submitted_at else None,
                }
            )
        return cases

    def _get_or_create_active_assignment(
        self,
        *,
        session: Session,
        user: User,
        cases: list[dict[str, Any]],
        batch_size: int,
    ) -> HumanEvalAssignment:
        active = session.scalar(
            select(HumanEvalAssignment).where(
                HumanEvalAssignment.user_id == user.id,
                HumanEvalAssignment.status == "active",
            )
        )
        if active is not None and not self._assignment_is_completed(session=session, user=user, assignment=active):
            return active
        if active is not None:
            active.status = "completed"
            active.completed_at = datetime.now(timezone.utc)
            session.flush()

        next_batch_number = self._next_assignment_batch_number(session=session, user=user)
        case_ids = self._select_assignment_case_ids(
            session=session,
            user=user,
            cases=cases,
            batch_size=batch_size,
        )
        assignment = HumanEvalAssignment(
            user_id=user.id,
            batch_number=next_batch_number,
            case_ids_json=case_ids,
            status="active",
        )
        session.add(assignment)
        session.flush()
        return assignment

    def _assignment_is_completed(self, *, session: Session, user: User, assignment: HumanEvalAssignment) -> bool:
        case_ids = self._assignment_case_ids(assignment)
        if not case_ids:
            return False
        submitted_case_ids = set(self._submitted_case_ids_for_user(session=session, user=user))
        return set(case_ids).issubset(submitted_case_ids)

    @staticmethod
    def _assignment_case_ids(assignment: HumanEvalAssignment) -> list[int]:
        raw_ids = assignment.case_ids_json or []
        return [int(case_id) for case_id in raw_ids]

    def _next_assignment_batch_number(self, *, session: Session, user: User) -> int:
        max_batch = session.scalar(
            select(func.max(HumanEvalAssignment.batch_number)).where(HumanEvalAssignment.user_id == user.id)
        )
        return int(max_batch or 0) + 1

    def _select_assignment_case_ids(
        self,
        *,
        session: Session,
        user: User,
        cases: list[dict[str, Any]],
        batch_size: int,
    ) -> list[int]:
        all_case_ids = [int(item["case_id"]) for item in cases]
        submitted_by_user = set(self._submitted_case_ids_for_user(session=session, user=user))
        coverage = self._submitted_counts_by_case(session=session)
        first_pass_candidates = [case_id for case_id in all_case_ids if case_id not in submitted_by_user]
        candidates = first_pass_candidates or all_case_ids
        candidates.sort(key=lambda case_id: (coverage.get(case_id, 0), self._stable_case_rank(case_id, user.id)))
        return candidates[:batch_size]

    def _submitted_case_ids_for_user(self, *, session: Session, user: User) -> list[int]:
        return list(
            session.scalars(
                select(HumanEvalRating.case_id).where(
                    HumanEvalRating.user_id == user.id,
                    HumanEvalRating.status == "submitted",
                )
            ).all()
        )

    def _submitted_counts_by_case(self, *, session: Session) -> dict[int, int]:
        rows = session.execute(
            select(HumanEvalRating.case_id, func.count(HumanEvalRating.id))
            .where(HumanEvalRating.status == "submitted")
            .group_by(HumanEvalRating.case_id)
        ).all()
        return {int(case_id): int(count) for case_id, count in rows}

    @staticmethod
    def _stable_case_rank(case_id: int, user_id: str) -> str:
        return hashlib.sha256(f"{user_id}:{case_id}".encode("utf-8")).hexdigest()

    def load_case(self, case_id: int) -> dict[str, Any]:
        return self.materials.load_case(case_id)

    def load_schema(self) -> dict[str, Any]:
        return self.materials.load_schema()

    def get_rating(self, session: Session, user: User, case_id: int) -> HumanEvalRating | None:
        return session.scalar(
            select(HumanEvalRating).where(
                HumanEvalRating.user_id == user.id,
                HumanEvalRating.case_id == case_id,
            )
        )

    def save_rating(
        self,
        session: Session,
        user: User,
        case_id: int,
        payload: HumanEvalRatingPayload,
    ) -> HumanEvalRating:
        case = self.load_case(case_id)
        rating = self.get_rating(session, user, case_id)
        now = datetime.now(timezone.utc)
        serialized_payload = payload.model_dump(mode="json")
        if rating is None:
            rating = HumanEvalRating(
                case_id=case_id,
                case_key=str(case.get("case_key") or f"case_{case_id}"),
                rater_id=payload.rater_id,
                user_id=user.id,
                status=payload.status,
                payload_json=serialized_payload,
                submitted_at=now if payload.status == "submitted" else None,
            )
            session.add(rating)
            session.flush()
            return rating

        rating.case_key = str(case.get("case_key") or f"case_{case_id}")
        rating.rater_id = payload.rater_id
        rating.status = payload.status
        rating.payload_json = serialized_payload
        if payload.status == "submitted":
            rating.submitted_at = now
        session.flush()
        return rating

    @staticmethod
    def serialize_rating(rating: HumanEvalRating) -> dict[str, Any]:
        return {
            "case_id": rating.case_id,
            "case_key": rating.case_key,
            "rater_id": rating.rater_id,
            "status": rating.status,
            "payload": rating.payload_json,
            "updated_at": rating.updated_at.isoformat() if rating.updated_at else None,
            "submitted_at": rating.submitted_at.isoformat() if rating.submitted_at else None,
        }

    def export_csv(self, session: Session) -> str:
        ratings = session.scalars(select(HumanEvalRating).order_by(HumanEvalRating.case_id, HumanEvalRating.rater_id)).all()
        fieldnames = self._csv_fieldnames()
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for rating in ratings:
            writer.writerow(self._rating_to_csv_row(rating))
        return output.getvalue()

    @staticmethod
    def _csv_fieldnames() -> list[str]:
        fields = ["case_id", "case_key", "rater_id", "status", "submitted_at", "updated_at"]
        for stage in SCORABLE_STAGES:
            for metric in STAGE_METRICS:
                fields.append(f"{stage}_{metric}")
                fields.append(f"{stage}_{metric}_reason")
        for metric in ROLE_METRICS:
            fields.append(metric)
            fields.append(f"{metric}_reason")
        return fields

    @staticmethod
    def _rating_to_csv_row(rating: HumanEvalRating) -> dict[str, Any]:
        payload = rating.payload_json or {}
        row: dict[str, Any] = {
            "case_id": rating.case_id,
            "case_key": rating.case_key,
            "rater_id": rating.rater_id,
            "status": rating.status,
            "submitted_at": rating.submitted_at.isoformat() if rating.submitted_at else "",
            "updated_at": rating.updated_at.isoformat() if rating.updated_at else "",
        }
        stage_scores = payload.get("stage_scores") or {}
        for stage in SCORABLE_STAGES:
            stage_payload = stage_scores.get(stage) or {}
            for metric in STAGE_METRICS:
                metric_payload = stage_payload.get(metric) or {}
                row[f"{stage}_{metric}"] = metric_payload.get("score", "")
                row[f"{stage}_{metric}_reason"] = metric_payload.get("reason", "")

        role_scores = payload.get("role_scores") or {}
        for metric in ROLE_METRICS:
            metric_payload = role_scores.get(metric) or {}
            row[metric] = metric_payload.get("score", "")
            row[f"{metric}_reason"] = metric_payload.get("reason", "")
        return row
