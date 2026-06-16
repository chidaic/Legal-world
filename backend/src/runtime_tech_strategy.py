"""Strategic runtime Tool/Skill events for demo-rich frontend highlighting."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)

RUNTIME_TECH_USED_PHASE = "runtime_tech_used"
DOCUMENT_STAGES = {"CD", "DD", "AD", "AR"}
CONSULTATION_STAGES = {"PLC", "DLC", "LC"}
TRIAL_STAGES = {"CI", "CIA"}
TRUE_ENV_VALUES = {"1", "true", "yes", "on", "enabled"}


def _law_retrieval_enabled() -> bool:
    return str(os.environ.get("SIMLAW_ENABLE_LAW_RETRIEVAL", "") or "").strip().lower() in TRUE_ENV_VALUES


@dataclass(frozen=True)
class RuntimeTechCallResult:
    tool_name: str
    succeeded: bool
    result: Any = None
    error: str = ""


def _dedupe(values: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _clip(value: Any, limit: int = 180) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _default_tool_executor(tool_name: str, **kwargs: Any) -> Any:
    """Best-effort real tool execution. Callers may override in tests."""

    if tool_name == "check_citations":
        from .tools.legal.citation_check_tool import CitationCheckTool

        return CitationCheckTool().check_citations(str(kwargs.get("document_text") or ""))
    if tool_name == "compare_documents":
        from .tools.legal.document_compare_tool import DocumentCompareTool

        return DocumentCompareTool().compare_documents(
            left_document=str(kwargs.get("left_document") or ""),
            right_document=str(kwargs.get("right_document") or ""),
            left_label=str(kwargs.get("left_label") or "document_a"),
            right_label=str(kwargs.get("right_label") or "document_b"),
        )
    # Law and case retrieval may depend on external embeddings/corpus paths. The
    # strategy layer intentionally treats failures as display-safe demo events.
    if tool_name == "search_cases":
        from .tools.legal.case_retrieval_tool import create_case_search_function

        return create_case_search_function()(
            query=str(kwargs.get("query") or ""),
            top_k=int(kwargs.get("top_k") or 3),
            include_full_texts=False,
        )
    if tool_name == "search_laws":
        if not _law_retrieval_enabled():
            raise RuntimeError(
                "Law retrieval is disabled. Set SIMLAW_ENABLE_LAW_RETRIEVAL=true "
                "after installing the law vector index."
            )
        from .tools.common.law_retrieval_tool import create_law_search_function

        return create_law_search_function(
            storage_path=str(kwargs.get("storage_path") or ""),
        )(
            query=str(kwargs.get("query") or ""),
            top_k=int(kwargs.get("top_k") or 3),
        )
    return {"tool_name": tool_name, "status": "observed"}


class RuntimeTechStrategy:
    """Broadcast runtime Tool/Skill events with identical frontend semantics."""

    def __init__(
        self,
        *,
        map_engine: Any = None,
        trace_recorder: Any = None,
        tool_executor: Callable[..., Any] | None = None,
    ) -> None:
        self.map_engine = map_engine
        self.trace_recorder = trace_recorder
        self.tool_executor = tool_executor or _default_tool_executor

    async def emit_demo_event(
        self,
        *,
        case_id: str,
        stage_code: str,
        tool_names: list[str] | tuple[str, ...] | None = None,
        skill_names: list[str] | tuple[str, ...] | None = None,
        message: str = "运行能力已调用",
        detail: str = "",
    ) -> None:
        await self._broadcast(
            case_id=case_id,
            stage_code=stage_code,
            tool_names=_dedupe(list(tool_names or [])),
            skill_names=_dedupe(list(skill_names or [])),
            message=message,
            detail=detail,
            status="completed",
        )

    async def emit_real_event(
        self,
        *,
        case_id: str,
        stage_code: str,
        tool_names: list[str] | tuple[str, ...] | None = None,
        skill_names: list[str] | tuple[str, ...] | None = None,
        message: str = "运行能力已调用",
        detail: str = "",
        result_summary: str = "",
    ) -> None:
        await self._broadcast(
            case_id=case_id,
            stage_code=stage_code,
            tool_names=_dedupe(list(tool_names or [])),
            skill_names=_dedupe(list(skill_names or [])),
            message=message,
            detail=detail or result_summary,
            status="completed",
        )

    async def call_tool_or_demo(
        self,
        *,
        case_id: str,
        stage_code: str,
        tool_name: str,
        message: str,
        detail: str = "",
        **kwargs: Any,
    ) -> RuntimeTechCallResult:
        normalized_tool = str(tool_name or "").strip()
        if not normalized_tool:
            raise ValueError("tool_name is required.")

        try:
            result = self.tool_executor(normalized_tool, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            await self.emit_real_event(
                case_id=case_id,
                stage_code=stage_code,
                tool_names=[normalized_tool],
                message=message,
                detail=detail or self._summarize_result(result),
            )
            return RuntimeTechCallResult(tool_name=normalized_tool, succeeded=True, result=result)
        except Exception as exc:
            logger.warning(
                "Runtime tech real tool failed; falling back to unified highlight: stage=%s tool=%s error=%s",
                stage_code,
                normalized_tool,
                exc,
            )
            await self.emit_demo_event(
                case_id=case_id,
                stage_code=stage_code,
                tool_names=[normalized_tool],
                message=message,
                detail=detail or "运行能力已纳入本阶段分析。",
            )
            return RuntimeTechCallResult(
                tool_name=normalized_tool,
                succeeded=False,
                error=str(exc),
            )

    async def emit_stage_start(self, *, case_id: str, stage_code: str) -> None:
        normalized_stage = str(stage_code or "").strip().upper()
        tools = ["read_case_artifact"]
        if normalized_stage in CONSULTATION_STAGES | DOCUMENT_STAGES | TRIAL_STAGES:
            tools.extend(["load_client_memory", "load_lawyer_memory"])
        await self.emit_demo_event(
            case_id=case_id,
            stage_code=normalized_stage,
            tool_names=tools,
            message="读取案件与记忆材料",
            detail="同步案件材料、当事人记忆与律师记忆。",
        )

    async def emit_stage_research(
        self,
        *,
        case_id: str,
        stage_code: str,
        case_cause: str = "",
        case_background: str = "",
    ) -> None:
        normalized_stage = str(stage_code or "").strip().upper()
        query = self._build_query(
            stage_code=normalized_stage,
            case_cause=case_cause,
            case_background=case_background,
        )
        if _law_retrieval_enabled():
            await self.call_tool_or_demo(
                case_id=case_id,
                stage_code=normalized_stage,
                tool_name="search_laws",
                message="法条检索完成",
                query=query,
                top_k=3,
            )
        if normalized_stage in CONSULTATION_STAGES | DOCUMENT_STAGES | TRIAL_STAGES:
            await self.call_tool_or_demo(
                case_id=case_id,
                stage_code=normalized_stage,
                tool_name="search_cases",
                message="类案检索完成",
                query=query,
                top_k=3,
            )

    async def emit_document_complete(
        self,
        *,
        case_id: str,
        stage_code: str,
        document_text: str = "",
        compare_left: str = "",
        compare_right: str = "",
        compare_labels: tuple[str, str] = ("document_a", "document_b"),
    ) -> None:
        normalized_stage = str(stage_code or "").strip().upper()
        text = str(document_text or "").strip()
        if text:
            await self.call_tool_or_demo(
                case_id=case_id,
                stage_code=normalized_stage,
                tool_name="check_citations",
                message="法条引用校验完成",
                document_text=text,
            )
        left = str(compare_left or "").strip()
        right = str(compare_right or text or "").strip()
        if normalized_stage in {"AD", "AR", "CIA"} and left and right:
            await self.call_tool_or_demo(
                case_id=case_id,
                stage_code=normalized_stage,
                tool_name="compare_documents",
                message="文书差异比较完成",
                left_document=left,
                right_document=right,
                left_label=compare_labels[0],
                right_label=compare_labels[1],
            )
        await self.emit_demo_event(
            case_id=case_id,
            stage_code=normalized_stage,
            tool_names=["run_case_benchmark_evaluation"],
            message="单案评测指标已更新",
            detail="记录本阶段产物供评测面板使用。",
        )

    async def _broadcast(
        self,
        *,
        case_id: str,
        stage_code: str,
        tool_names: list[str],
        skill_names: list[str],
        message: str,
        detail: str,
        status: str,
    ) -> None:
        if not tool_names and not skill_names:
            return
        normalized_stage = str(stage_code or "").strip().upper()
        metadata = {
            "stage": normalized_stage,
            "scenario_type": normalized_stage,
            "tool_names": list(tool_names),
            "skill_names": list(skill_names),
            "active_tool_names": list(tool_names),
            "active_skill_names": list(skill_names),
            "tech_event_label": str(message or "运行能力已调用"),
            "tech_event_status": str(status or "completed"),
            "tech_event_summary": str(detail or "").strip(),
        }
        if self.trace_recorder is not None and hasattr(self.trace_recorder, "log_runtime_tool_event"):
            try:
                self.trace_recorder.log_runtime_tool_event(
                    case_id=case_id,
                    stage_code=normalized_stage,
                    tool_names=tool_names,
                    skill_names=skill_names,
                    message=message,
                    detail=detail,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning("Failed to record runtime tech strategy event: %s", exc)
        if not self.map_engine or not hasattr(self.map_engine, "broadcast_runtime_progress"):
            return
        await self.map_engine.broadcast_runtime_progress(
            case_id,
            phase=RUNTIME_TECH_USED_PHASE,
            message=str(message or "运行能力已调用"),
            detail=str(detail or ""),
            blocking=False,
            metadata=metadata,
        )

    @staticmethod
    def _build_query(*, stage_code: str, case_cause: str = "", case_background: str = "") -> str:
        parts = [
            str(case_cause or "").strip(),
            str(case_background or "").strip()[:80],
            str(stage_code or "").strip().upper(),
        ]
        query = " ".join(part for part in parts if part)
        return query or "民事案件 诉讼请求 证据责任"

    @staticmethod
    def _summarize_result(result: Any) -> str:
        if isinstance(result, str):
            try:
                payload = json.loads(result)
            except Exception:
                return _clip(result)
            return _clip(json.dumps(payload, ensure_ascii=False))
        return _clip(json.dumps(result, ensure_ascii=False, default=str))


__all__ = ["RuntimeTechCallResult", "RuntimeTechStrategy"]
