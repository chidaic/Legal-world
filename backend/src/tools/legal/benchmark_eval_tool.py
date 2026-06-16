"""Optional benchmark evaluation tool wrapping the existing EvalPipeline."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from camel.toolkits import FunctionTool

from src.utils.file_io import DEFAULT_DATASET_PATH, map_eval_stage_code, stage_label, write_json


BENCHMARK_EVAL_TOOL_NAME = "run_case_benchmark_evaluation"
_PARTY_ROLE_ALIASES = {
    "原告": "plaintiff",
    "plaintiff": "plaintiff",
    "被告": "defendant",
    "defendant": "defendant",
}


def _normalize_party_role(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = raw.lower()
    if normalized in _PARTY_ROLE_ALIASES:
        return _PARTY_ROLE_ALIASES[normalized]
    if raw in _PARTY_ROLE_ALIASES:
        return _PARTY_ROLE_ALIASES[raw]
    raise ValueError("party_role must be one of: 原告, 被告, plaintiff, defendant")


def _extract_summary_items(stage_code: str, stage_result: dict[str, Any]) -> dict[str, Any]:
    items: dict[str, Any] = {}

    if "qa_evals" in stage_result:
        for index, qa_eval in enumerate(stage_result.get("qa_evals", []), start=1):
            items[f"Q{index}"] = {
                "question": qa_eval.get("question"),
                "score": qa_eval.get("score_normalized"),
                "raw_score": qa_eval.get("score"),
                "reason": qa_eval.get("reason"),
            }
        return items

    metrics = stage_result.get("metrics")
    if isinstance(metrics, dict):
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, dict):
                items[metric_name] = {
                    "score": metric_value.get("score"),
                    "raw_score": metric_value.get("raw_score"),
                    "reason": metric_value.get("reason"),
                }
        return items

    doc_value = stage_result.get("DOC")
    if doc_value is not None:
        items["DOC"] = doc_value

    if stage_result.get("status") is not None:
        items["status"] = stage_result.get("status")
    if stage_result.get("error") is not None:
        items["error"] = stage_result.get("error")
    return items


def _build_case_eval_summary(
    pipeline_result: dict[str, Any],
    eval_result: dict[str, Any],
    *,
    party_role: str,
) -> dict[str, Any]:
    stage_overview: dict[str, Any] = {}
    for original_stage_code, stage_result in (eval_result.get("stage_eval_results") or {}).items():
        actual_stage_code = map_eval_stage_code(original_stage_code, pipeline_result)
        stage_overview[actual_stage_code] = {
            "stage_code": actual_stage_code,
            "stage_name": stage_label(actual_stage_code),
            "stage_score": stage_result.get("stage_score"),
            "items": _extract_summary_items(actual_stage_code, stage_result),
            "status": stage_result.get("status"),
            "error": stage_result.get("error"),
        }

    return {
        "case_id": eval_result.get("case_id"),
        "party_role": party_role,
        "judge_model": eval_result.get("judge_model"),
        "overall_score": eval_result.get("overall_score"),
        "stages_evaluated": list(stage_overview.keys()),
        "stage_overview": stage_overview,
    }


def _load_eval_runtime() -> tuple[Any, Any]:
    from src.data.data_loader import DataLoader
    from src.eval.eval_pipeline import EvalPipeline

    return DataLoader, EvalPipeline


class BenchmarkEvalTool:
    """Run one case benchmark evaluation without enabling it by default."""

    def run_case_benchmark_evaluation(
        self,
        case_dir: str,
        party_role: str,
        dataset_path: str = "",
        judge_model: str = "",
        start_stage: str = "",
        end_stage: str = "",
    ) -> str:
        resolved_case_dir = Path(str(case_dir or "")).expanduser().resolve()
        if not resolved_case_dir.exists():
            raise FileNotFoundError(f"case_dir not found: {resolved_case_dir}")

        pipeline_result_path = resolved_case_dir / "pipeline_result.json"
        if not pipeline_result_path.exists():
            raise FileNotFoundError(f"pipeline_result.json not found under: {resolved_case_dir}")

        normalized_party_role = _normalize_party_role(party_role)
        resolved_dataset_path = (
            Path(str(dataset_path).strip()).expanduser().resolve()
            if str(dataset_path or "").strip()
            else DEFAULT_DATASET_PATH.resolve()
        )
        if not resolved_dataset_path.exists():
            raise FileNotFoundError(f"dataset_path not found: {resolved_dataset_path}")

        DataLoader, EvalPipeline = _load_eval_runtime()
        original_pipeline_result = json.loads(pipeline_result_path.read_text(encoding="utf-8"))
        overridden_pipeline_result = copy.deepcopy(original_pipeline_result)
        overridden_pipeline_result["party_role"] = normalized_party_role
        stage_output = overridden_pipeline_result.get("stage_output")
        if isinstance(stage_output, dict):
            stage_output["party_role"] = normalized_party_role

        eval_dir = resolved_case_dir / "eval_result"
        eval_dir.mkdir(parents=True, exist_ok=True)
        full_result_path = eval_dir / f"eval_result_full_{normalized_party_role}.json"
        summary_path = eval_dir / f"summary_{normalized_party_role}.json"

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as temp_file:
            json.dump(overridden_pipeline_result, temp_file, ensure_ascii=False, indent=2)
            temp_pipeline_result_path = Path(temp_file.name)

        try:
            data_loader = DataLoader(str(resolved_dataset_path))
            eval_pipeline = EvalPipeline(
                pipeline_result_path=str(temp_pipeline_result_path),
                data_loader=data_loader,
                output_path=str(full_result_path),
                start_stage=str(start_stage or "").strip().upper() or None,
                end_stage=str(end_stage or "").strip().upper() or None,
                judge_model_type=str(judge_model or "").strip() or None,
            )
            final_eval_result = eval_pipeline.run()
        finally:
            temp_pipeline_result_path.unlink(missing_ok=True)

        summary = _build_case_eval_summary(
            overridden_pipeline_result,
            final_eval_result,
            party_role=normalized_party_role,
        )
        write_json(summary_path, summary)

        for original_stage_code, stage_result in (final_eval_result.get("stage_eval_results") or {}).items():
            actual_stage_code = map_eval_stage_code(original_stage_code, overridden_pipeline_result)
            write_json(
                eval_dir / f"{actual_stage_code}_{normalized_party_role}.json",
                {
                    "case_id": final_eval_result.get("case_id"),
                    "party_role": normalized_party_role,
                    "stage_code": actual_stage_code,
                    "stage_name": stage_label(actual_stage_code),
                    "detail": stage_result,
                },
            )

        payload = {
            "tool_name": BENCHMARK_EVAL_TOOL_NAME,
            "status": "ok",
            "case_dir": str(resolved_case_dir),
            "party_role": normalized_party_role,
            "dataset_path": str(resolved_dataset_path),
            "judge_model": final_eval_result.get("judge_model"),
            "overall_score": final_eval_result.get("overall_score"),
            "summary_path": str(summary_path),
            "full_result_path": str(full_result_path),
            "stages_evaluated": summary.get("stages_evaluated", []),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": BENCHMARK_EVAL_TOOL_NAME,
            "description": (
                "包装现有 Benchmark/EvalPipeline，对指定案件输出目录执行单案评估。"
                "输入案件输出目录和参与当事人身份，默认不启用。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "case_dir": {
                        "type": "string",
                        "description": "案件输出目录，目录下必须有 pipeline_result.json。",
                    },
                    "party_role": {
                        "type": "string",
                        "description": "参与当事人身份，只支持 原告/被告 或 plaintiff/defendant。",
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "可选数据集路径；默认使用项目默认评测数据集。",
                    },
                    "judge_model": {
                        "type": "string",
                        "description": "可选评测法官模型名；不传则使用运行时默认值。",
                    },
                    "start_stage": {
                        "type": "string",
                        "description": "可选起始评测阶段，如 LC、DRAFT、CI、APPEAL_DRAFT。",
                    },
                    "end_stage": {
                        "type": "string",
                        "description": "可选结束评测阶段，如 LC、DRAFT、CI、APPEAL_DRAFT、CIA。",
                    },
                },
                "required": ["case_dir", "party_role"],
                "additionalProperties": False,
            },
        },
    }


def create_benchmark_eval_tool(agent: Any | None = None) -> FunctionTool:
    del agent
    impl = BenchmarkEvalTool()
    return FunctionTool(
        impl.run_case_benchmark_evaluation,
        openai_tool_schema=_build_schema(),
    )


__all__ = [
    "BENCHMARK_EVAL_TOOL_NAME",
    "BenchmarkEvalTool",
    "create_benchmark_eval_tool",
]
