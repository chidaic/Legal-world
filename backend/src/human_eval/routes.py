from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from src.core.models import User
from src.human_eval.materials import HumanEvalMaterialNotFoundError
from src.human_eval.schemas import HumanEvalRatingPayload
from src.human_eval.service import HumanEvalService


def create_human_eval_router(
    *,
    current_user_dependency: Callable[..., User],
    session_dependency: Callable[..., Any],
    service: HumanEvalService | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/human-eval", tags=["human-eval"])
    runtime_service = service or HumanEvalService()

    @router.get("/cases")
    async def list_cases(
        current_user: User = Depends(current_user_dependency),
        session: Session = Depends(session_dependency),
    ):
        return runtime_service.list_cases(session=session, user=current_user)

    @router.get("/cases/{case_id}")
    async def get_case(case_id: int, current_user: User = Depends(current_user_dependency)):
        try:
            return {"case": runtime_service.load_case(case_id)}
        except HumanEvalMaterialNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/schema")
    async def get_schema(current_user: User = Depends(current_user_dependency)):
        return {"schema": runtime_service.load_schema()}

    @router.get("/ratings/{case_id}")
    async def get_rating(
        case_id: int,
        current_user: User = Depends(current_user_dependency),
        session: Session = Depends(session_dependency),
    ):
        rating = runtime_service.get_rating(session=session, user=current_user, case_id=case_id)
        if rating is None:
            return {"rating": None}
        return {"rating": runtime_service.serialize_rating(rating)}

    @router.put("/ratings/{case_id}")
    async def save_rating(
        case_id: int,
        payload: HumanEvalRatingPayload,
        current_user: User = Depends(current_user_dependency),
        session: Session = Depends(session_dependency),
    ):
        try:
            rating = runtime_service.save_rating(
                session=session,
                user=current_user,
                case_id=case_id,
                payload=payload,
            )
        except HumanEvalMaterialNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"rating": runtime_service.serialize_rating(rating)}

    @router.get("/export.csv")
    async def export_csv(
        current_user: User = Depends(current_user_dependency),
        session: Session = Depends(session_dependency),
    ):
        csv_body = runtime_service.export_csv(session=session)
        return Response(
            content=csv_body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="human_eval_ratings.csv"'},
        )

    return router
