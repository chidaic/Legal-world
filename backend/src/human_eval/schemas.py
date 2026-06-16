from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SCORABLE_STAGES = ("LC", "DRAFT", "CI", "APPEAL_DRAFT", "CIA")
STAGE_METRICS = ("procedural_compliance", "process_coherence")
ROLE_METRICS = (
    "client_stance_authenticity",
    "client_role_distinguishability",
    "lawyer_stance_authenticity",
    "lawyer_role_distinguishability",
    "judge_stance_authenticity",
    "judge_role_distinguishability",
)


class HumanEvalMetricScore(BaseModel):
    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("评分理由不能为空")
        return stripped


class HumanEvalStageScore(BaseModel):
    procedural_compliance: HumanEvalMetricScore
    process_coherence: HumanEvalMetricScore


class HumanEvalRatingPayload(BaseModel):
    rater_id: str = Field(min_length=1, max_length=128)
    status: Literal["draft", "submitted"] = "draft"
    stage_scores: dict[str, HumanEvalStageScore] = Field(default_factory=dict)
    role_scores: dict[str, HumanEvalMetricScore] = Field(default_factory=dict)

    @field_validator("rater_id")
    @classmethod
    def rater_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("评审编号不能为空")
        return stripped

    @model_validator(mode="after")
    def validate_score_shape(self) -> "HumanEvalRatingPayload":
        extra_stages = sorted(set(self.stage_scores) - set(SCORABLE_STAGES))
        if extra_stages:
            raise ValueError(f"不可评分阶段不能提交评分: {', '.join(extra_stages)}")

        extra_roles = sorted(set(self.role_scores) - set(ROLE_METRICS))
        if extra_roles:
            raise ValueError(f"未知角色评分字段: {', '.join(extra_roles)}")

        if self.status == "submitted":
            missing_stages = [stage for stage in SCORABLE_STAGES if stage not in self.stage_scores]
            if missing_stages:
                raise ValueError(f"提交问卷必须填写全部阶段评分: {', '.join(missing_stages)}")

            missing_roles = [metric for metric in ROLE_METRICS if metric not in self.role_scores]
            if missing_roles:
                raise ValueError(f"提交问卷必须填写全部角色评分: {', '.join(missing_roles)}")

        return self


class HumanEvalRatingResponse(BaseModel):
    case_id: int
    case_key: str
    rater_id: str
    status: str
    payload: dict
    updated_at: datetime | None = None
    submitted_at: datetime | None = None
