"""Evaluation pipeline for SimAilaw stage outputs."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from dotenv import load_dotenv

# Add backend src to path for camel imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
load_dotenv(env_path)

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from data.data_loader import DataLoader
from utils.model_config import (
    build_runtime_openai_chat_config,
    resolve_openai_chat_model,
)
from utils.drafted_document_sections import resolve_stage_document_text
from utils.prompt_profile import use_lightweight_eval_judge_prompt

logger = logging.getLogger(__name__)
QUESTION_DATASET_LOADER_CACHE: Dict[str, DataLoader] = {}

JUDGE_SCORE_BANDS: Dict[str, str] = {
    "9-10": "核心内容完整准确，论证充分，完整覆盖参考答案要点，但整体高度可信且有说服力。",
    "7-8": "大部分内容合理且较完整，与参考答案相比有少量遗漏或展开不足，但不影响主要判断。",
    "5-6": "部分内容成立，但存在较明显遗漏，或证据、理由运用不够充分。",
    "3-4": "只有少量相关内容，关键事实、证据或理由缺失较多，论证较弱。",
    "0-2": "与参考严重偏离、明显错误，或几乎未回应该评分维度。",
}

CURRENT_LAW_EVAL_RULE = (
    "法律适用口径：评估法律规则是否正确时，以评测时有效的现行法律、司法解释和通行裁判规则为基准；"
    "如果GT沿用历史旧规则或旧裁判口径，而候选答案采用现行有效规则且论证自洽，"
    "不得仅因法律口径不同于GT而降分。"
)

LLM_AS_JUDGE_PROMPT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "LC": {
        "system_prompt": "你是法律咨询阶段的评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考问题和参考答案，评估律师回答的质量。",
        "metrics": {
            "问答质量": {
                "focus": "重点看律师回答是否正面回应问题、事实法律是否基本正确、是否能帮助当事人理解问题；",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "CD": {
        "system_prompt": "你是起诉状评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选起诉状的内容质量。",
        "metrics": {
            "诉讼请求": {
                "focus": "重点看诉讼请求是否覆盖核心诉求、表述是否准确清楚，不要求与GT逐字复现。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "事实与理由": {
                "focus": "重点看事实脉络、争议焦点和支持理由是否合理完整，可以与GT表述不同，但应言之成理。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "证据": {
                "focus": "重点看是否覆盖关键证据并能对应诉求，不要求逐项照抄GT。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "DD": {
        "system_prompt": "你是答辩状评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选答辩状的内容质量。",
        "metrics": {
            "答辩意见": {
                "focus": "重点看是否清楚表达答辩立场、是否正面回应起诉主张，不要求与GT逐字一致。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "证据": {
                "focus": "重点看是否覆盖支撑答辩的关键证据，允许合理概括。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "AD": {
        "system_prompt": "你是上诉状评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选上诉状的内容质量。",
        "metrics": {
            "上诉请求": {
                "focus": "重点看是否准确表达上诉目标和请求事项，不要求逐字复现GT。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "事实与理由": {
                "focus": "重点看上诉理由是否成体系、是否能支撑上诉请求，允许不同表述方式。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "证据": {
                "focus": "重点看是否合理使用和概括二审新证据，不要求逐项照抄。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "AR": {
        "system_prompt": "你是上诉答辩状评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选上诉答辩状的内容质量。",
        "metrics": {
            "答辩意见": {
                "focus": "重点看是否正面回应上诉请求与理由、是否能形成维持原判或反驳上诉的完整立场。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "证据": {
                "focus": "重点看是否合理使用和概括被上诉人二审新证据，不要求逐项照抄。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "CI": {
        "system_prompt": "你是一审庭审评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选庭审发言质量，重点看发言是否言之成理、是否覆盖核心争点。",
        "metrics": {
            "诉讼与答辩一致性": {
                "focus": "重点看发言结论是否忠实、完整地表达本方诉讼主张或答辩意见，允许同义表达和不同组织方式。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "事实与证据运用完整性": {
                "focus": "重点看是否围绕整个庭审举证质证环节，充分处理全案证据、双方质证意见和法院查明事实；既能支持己方主张，也能回应对方证据与争点。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "法律说理充分性": {
                "focus": "重点看是否形成有根据的法律论证并与裁判说理相呼应，不要求复述GT原文。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
    "CIA": {
        "system_prompt": "你是二审庭审评测法官。给定参考答案，但不要机械做关键词匹配，只输出JSON。",
        "task_prompt": "请根据参考信息评估候选二审庭审发言质量，重点看发言是否言之成理、是否覆盖核心争点。",
        "metrics": {
            "上诉与答辩一致性": {
                "focus": "重点看发言结论是否忠实、完整地表达本方上诉主张或答辩意见，允许同义表达和不同组织方式。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "事实与证据运用完整性": {
                "focus": "重点看是否围绕整个二审举证质证环节，充分处理全案新证据、双方质证意见和法院查明事实；既能支持己方主张，也能回应对方证据与争点。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
            "法律说理充分性": {
                "focus": "重点看是否形成有根据的法律论证并与裁判说理相呼应，不要求复述GT原文。",
                "score_bands": JUDGE_SCORE_BANDS,
            },
        },
    },
}

LC_QA_EXTRACTION_SYSTEM_PROMPT = (
    "你是法律咨询阶段的评测预处理助手。"
    "请把完整咨询聊天中与给定标准问题相关的内容，抽取为结构化问答对。"
    "一个标准问题可能对应多轮追问、澄清和补充，你需要整合相关轮次，"
    "只保留与该标准问题直接相关的当事人提问摘要和律师回答。只输出JSON。"
)

EVIDENCE_SECTION_LABELS = [
    '证据和证据来源，证人姓名和住所：',
    '证据和证据来源，证人姓名和住所:',
    '证据和证据来源、证人姓名和住所：',
    '证据和证据来源、证人姓名和住所:',
    '证据和证据来源，证人证言等：',
    '证据和证据来源，证人证言等:',
    '证据：',
    '证据:',
]

BENCHMARK_STAGE_WEIGHTS = {
    "LC": 10,
    "CD": 15,
    "DD": 15,
    "CI": 30,
    "AD": 15,
    "AR": 15,
    "CIA": 30,
}


def _format_score_bands(score_bands: Dict[str, str]) -> str:
    ordered_bands = ["9-10", "7-8", "5-6", "3-4", "0-2"]
    return "\n".join(f"{band}分：{score_bands[band]}" for band in ordered_bands)


def _build_stage_metric_prompt(stage_code: str, metric_names: List[str]) -> str:
    stage_config = LLM_AS_JUDGE_PROMPT_LIBRARY[stage_code]
    metric_blocks = []
    for index, metric_name in enumerate(metric_names, start=1):
        metric_config = stage_config["metrics"][metric_name]
        score_bands = _format_score_bands(metric_config["score_bands"])
        metric_blocks.append(
            f"{index}. {metric_name}\n"
            f"评分关注点：{metric_config['focus']}\n"
            f"评分档位：\n{score_bands}"
        )
    metric_text = "\n\n".join(metric_blocks)
    brevity_rule = "每个评分维度的 reason 请尽量精炼，控制在 120 字以内，避免输出被截断。"
    document_context_rule = ""
    if stage_code in {"CD", "DD", "AD", "AR"}:
        document_context_rule = (
            "候选内容以完整文书形式提供。评分每个维度时，请在候选完整文书中查找对应内容，"
            "不要因为候选文书未使用固定标题、标题名称不同、段落顺序不同而直接判为空；"
            "只有完整文书确实没有该维度内容时，才按缺失处理。\n"
        )
    return (
        f"{stage_config['task_prompt']}\n{brevity_rule}\n"
        f"{CURRENT_LAW_EVAL_RULE}\n"
        f"{document_context_rule}"
        "评分时应以GT为重要参考，但不得机械按关键词命中或逐字复现程度打分。\n"
        "允许同义替换、表达顺序差异、合理概括与有根据的延展论证；只要核心主张、事实关系、证据运用或法律说理成立，就应给予相应分数。\n"
        "只有在关键遗漏、明显冲突、事实或法律错误、论证空泛时，再显著降分。\n"
        f"评分维度：\n{metric_text}\n"
        "只输出JSON。"
    )


def build_profiled_judge_system_prompt(stage_code: str, prod_prompt: str) -> str:
    if not use_lightweight_eval_judge_prompt():
        return prod_prompt
    return (
        f"你是法律评测助手，负责{stage_code}阶段快速评分。"
        "GT是重要参考，但不要机械做关键词匹配；应判断候选内容是否言之成理、是否覆盖核心要点。"
        f"{CURRENT_LAW_EVAL_RULE}"
        "只输出JSON。"
    )


def build_profiled_judge_eval_prompt(
    stage_code: str,
    prod_prompt: str,
    *,
    sections: List[tuple[str, str]],
    json_schema: str,
) -> str:
    rendered_sections = [f"{title}: {value or '无'}" for title, value in sections]
    compact_sections = "\n".join(rendered_sections)
    if not use_lightweight_eval_judge_prompt():
        return (
            f"{prod_prompt}\n\n"
            f"{CURRENT_LAW_EVAL_RULE}\n\n"
            f"参考信息：\n{compact_sections}\n\n"
            f"输出JSON格式：\n{json_schema}"
        )
    return (
        f"{stage_code}快速评分\n"
        "原则：GT是重要参考，不要求逐字复述；允许同义表述、顺序调整和合理概括，重点看是否言之有理、是否覆盖核心点。\n"
        f"{CURRENT_LAW_EVAL_RULE}\n"
        f"{prod_prompt}\n"
        f"{compact_sections}\n"
        f"JSON:{json_schema}"
    )


class EvalPipeline:
    VALID_STAGES: Set[str] = {
        "LC",
        "DRAFT",
        "CI",
        "SD",
        "APPEAL_DRAFT",
        "CIA",
    }
    STAGE_ORDER = ["LC", "DRAFT", "CI", "SD", "APPEAL_DRAFT", "CIA"]

    def __init__(
        self,
        pipeline_result_path: str,
        data_loader: DataLoader,
        case_index: Optional[int] = None,
        output_path: Optional[str] = None,
        start_stage: Optional[str] = None,
        end_stage: Optional[str] = None,
        judge_model_type: str | ModelType | None = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        with open(pipeline_result_path, "r", encoding="utf-8") as f:
            self.pipeline_result = json.load(f)

        self.data_loader = data_loader
        pipeline_case_id = self.pipeline_result.get("case_id")

        if case_index is not None:
            self.source_data = data_loader.get_case(case_index)
            if not self.source_data:
                raise ValueError(f"DataLoader 中找不到 index={case_index} 对应的案件")
        elif pipeline_case_id is not None:
            self.source_data = None
            for candidate in getattr(data_loader, "cases", []):
                if candidate and str(candidate.get("original_id", "")) == str(pipeline_case_id):
                    self.source_data = candidate
                    break
            if self.source_data is None:
                raise ValueError(f"DataLoader 中找不到 original_id={pipeline_case_id} 对应的案件")
        else:
            self.source_data = data_loader.get_case(0)
            if not self.source_data:
                raise ValueError("DataLoader 中没有任何案件")

        if output_path is None:
            result_dir = os.path.dirname(pipeline_result_path)
            output_path = os.path.join(result_dir, "eval_result.json")
        self.output_path = output_path

        self.start_stage = start_stage.upper() if start_stage else None
        self.end_stage = end_stage.upper() if end_stage else None
        if self.start_stage and self.start_stage not in self.VALID_STAGES:
            raise ValueError(f"非法 start_stage: {start_stage}")
        if self.end_stage and self.end_stage not in self.VALID_STAGES:
            raise ValueError(f"非法 end_stage: {end_stage}")

        self.judge_model_type = resolve_openai_chat_model(explicit_model=judge_model_type)
        self.progress_callback = progress_callback
        self.eval_results: Dict[str, Any] = {}
        self._reached_start = self.start_stage is None
        self._reached_end = False
        self.stages_completed = self.pipeline_result.get("stages_completed", [])

    def _should_eval_stage(self, stage: str) -> bool:
        if self._reached_end:
            return False
        if self.start_stage and stage == self.start_stage:
            self._reached_start = True
        if not self._reached_start:
            return False
        return stage in self.stages_completed

    def _mark_stage_done(self, stage: str) -> None:
        self._emit_progress("stage_completed", stage)
        if self.end_stage and stage == self.end_stage:
            self._reached_end = True

    def _notify_stage_started(self, stage: str) -> None:
        self._emit_progress("stage_started", stage)

    def _resolve_progress_stage(self, stage: str) -> str:
        if stage == "DRAFT":
            draft_result = self.pipeline_result.get("stage_results", {}).get("DRAFT", {})
            if isinstance(draft_result, dict):
                if "complaint_statement" in draft_result:
                    return "CD"
                if "defense_statement" in draft_result:
                    return "DD"
            party_role = (
                self.pipeline_result.get("stage_output", {}).get("party_role")
                or self.pipeline_result.get("party_role")
                or "plaintiff"
            )
            return "CD" if party_role == "plaintiff" else "DD"
        if stage == "APPEAL_DRAFT":
            appeal_result = self.pipeline_result.get("stage_results", {}).get("APPEAL_DRAFT", {})
            if isinstance(appeal_result, dict):
                if "appeal_statement" in appeal_result:
                    return "AD"
                if "appeal_response_statement" in appeal_result:
                    return "AR"
            sd_result = self.pipeline_result.get("stage_results", {}).get("SD", {})
            if isinstance(sd_result, dict):
                return "AD" if sd_result.get("is_appellant", True) else "AR"
        return stage

    def _emit_progress(self, event_type: str, stage: Optional[str] = None) -> None:
        if self.progress_callback is None:
            return
        stage_code = self._resolve_progress_stage(stage) if stage else None
        self.progress_callback(
            {
                "event": event_type,
                "case_id": self.pipeline_result.get("case_id"),
                "stage_code": stage_code,
            }
        )

    def _create_judge_agent(self, system_prompt: str) -> ChatAgent:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=self.judge_model_type,
            model_config_dict=build_runtime_openai_chat_config(
                model_name=self.judge_model_type
            ),
        )
        return ChatAgent(system_message=system_prompt, model=model)

    def _judge_call(self, agent: ChatAgent, prompt: str) -> str:
        user_message = BaseMessage.make_user_message(role_name="user", content=prompt)
        response = agent.step(user_message)
        return response.msgs[0].content

    @staticmethod
    def _is_structured_json_payload(payload: Any) -> bool:
        if isinstance(payload, dict):
            return True
        if isinstance(payload, list):
            return not payload or all(isinstance(item, dict) for item in payload)
        return False

    def _parse_json_payload(self, response: str) -> Any:
        text = str(response or "").strip()
        if not text:
            return None

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        for block in fenced_blocks:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            if self._is_structured_json_payload(payload):
                return payload

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\{\[]", text):
            snippet = text[match.start():]
            try:
                payload, _ = decoder.raw_decode(snippet)
            except json.JSONDecodeError:
                continue
            if self._is_structured_json_payload(payload):
                return payload

        return None

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        payload = self._parse_json_payload(response)
        if isinstance(payload, dict):
            return payload

        score_match = re.search(r"(\d+)", response)
        score = int(score_match.group(1)) if score_match else 0
        return {"score": score, "reason": response}

    def _salvage_metric_payload_from_response(
        self,
        response: str,
        metric_names: List[str],
    ) -> Dict[str, Any] | None:
        text = str(response or "").strip()
        if not text or not metric_names:
            return None

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        candidate_text = fenced_blocks[0] if fenced_blocks else text
        salvaged: Dict[str, Any] = {}

        for index, metric_name in enumerate(metric_names):
            current_match = re.search(rf'"{re.escape(metric_name)}"\s*:\s*\{{', candidate_text)
            if current_match is None:
                return None

            current_start = current_match.start()
            next_start = len(candidate_text)
            for next_metric_name in metric_names[index + 1 :]:
                next_match = re.search(
                    rf'\n\s*"{re.escape(next_metric_name)}"\s*:\s*\{{',
                    candidate_text[current_start + 1 :],
                )
                if next_match is not None:
                    next_start = current_start + 1 + next_match.start()
                    break

            metric_block = candidate_text[current_start:next_start]
            score_match = re.search(r'"score"\s*:\s*(\d+)', metric_block)
            reason_match = re.search(r'"reason"\s*:\s*"', metric_block)
            if score_match is None or reason_match is None:
                return None

            reason_start = reason_match.end()
            reason_text = metric_block[reason_start:]
            reason_text = re.sub(r'"\s*,?\s*\}?\s*$', "", reason_text, count=1, flags=re.DOTALL).strip()
            salvaged[metric_name] = {
                "score": int(score_match.group(1)),
                "reason": reason_text,
            }

        return salvaged if len(salvaged) == len(metric_names) else None

    def _truncate_text(self, text: str, limit: int = 300) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    @staticmethod
    def _normalize_metric_key(value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
        text = re.sub(r"\s+", "", text)
        return re.sub(r"[^\w\u4e00-\u9fff]", "", text, flags=re.UNICODE)

    def _resolve_metric_payload(
        self,
        judge_result: Dict[str, Any],
        metric_names: List[str],
    ) -> Dict[str, Any] | None:
        if not isinstance(judge_result, dict):
            return None

        available_items = [
            (str(key), value)
            for key, value in judge_result.items()
            if isinstance(value, dict) and "score" in value
        ]
        if not available_items:
            return None

        resolved: Dict[str, Any] = {}
        used_keys: Set[str] = set()

        for metric_name in metric_names:
            exact_value = judge_result.get(metric_name)
            if isinstance(exact_value, dict) and "score" in exact_value:
                resolved[metric_name] = exact_value
                used_keys.add(metric_name)

        for metric_name in metric_names:
            if metric_name in resolved:
                continue

            normalized_expected = self._normalize_metric_key(metric_name)
            if not normalized_expected:
                continue

            exact_matches = [
                (candidate_key, candidate_value)
                for candidate_key, candidate_value in available_items
                if candidate_key not in used_keys
                and self._normalize_metric_key(candidate_key) == normalized_expected
            ]
            if len(exact_matches) == 1:
                candidate_key, candidate_value = exact_matches[0]
                resolved[metric_name] = candidate_value
                used_keys.add(candidate_key)
                continue

            fuzzy_matches = [
                (candidate_key, candidate_value)
                for candidate_key, candidate_value in available_items
                if candidate_key not in used_keys
                and (
                    normalized_expected in self._normalize_metric_key(candidate_key)
                    or self._normalize_metric_key(candidate_key) in normalized_expected
                )
            ]
            if len(fuzzy_matches) == 1:
                candidate_key, candidate_value = fuzzy_matches[0]
                resolved[metric_name] = candidate_value
                used_keys.add(candidate_key)

        if len(resolved) != len(metric_names):
            return None
        return resolved

    def _has_expected_metric_payload(self, judge_result: Dict[str, Any], metric_names: List[str]) -> bool:
        return self._resolve_metric_payload(judge_result, metric_names) is not None

    def _describe_metric_payload_issue(
        self,
        response: str,
        judge_result: Dict[str, Any],
        metric_names: List[str],
    ) -> str:
        if not str(response or "").strip():
            return "Judge returned empty content"

        if not isinstance(judge_result, dict):
            return f"Judge returned non-dict payload: {self._truncate_text(response)}"

        available_metric_keys = [
            str(key)
            for key, value in judge_result.items()
            if isinstance(value, dict) and "score" in value
        ]
        missing_metrics = [
            metric_name
            for metric_name in metric_names
            if not isinstance(judge_result.get(metric_name), dict)
            or "score" not in judge_result.get(metric_name, {})
        ]
        if missing_metrics:
            return (
                f"Judge payload missing expected metrics {missing_metrics}: "
                f"available={available_metric_keys} "
                f"{self._truncate_text(response)}"
            )

        return f"Judge returned unrecognized payload: {self._truncate_text(response)}"

    def _normalize_document(self, document: str) -> str:
        return str(document or "").replace("\r\n", "\n").replace("\r", "\n")

    def _strip_draft_end_marker(self, text: str) -> str:
        return self._normalize_document(text).replace("【起草结束】", "").strip()

    def _average(self, values: List[Optional[float]]) -> float:
        filtered = [value for value in values if value is not None]
        return sum(filtered) / len(filtered) if filtered else 0.0

    def _empty_component_eval(self, reason: str = "GT missing") -> Dict[str, Any]:
        return {"score": None, "raw_score": None, "reason": reason}

    def _format_item_names(self, incorrect_names: List[str], correct_name: str, text: str) -> str:
        for name in incorrect_names:
            text = text.replace(name, correct_name)
        return text

    def _birth_date_variants(self, value: str) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []
        variants = [text]
        parts = re.split(r"[-/.年]\s*", text.replace("月", "-").replace("日", ""))
        parts = [part for part in parts if part]
        if len(parts) >= 3:
            year, month, day = parts[:3]
            try:
                month_num = int(month)
                day_num = int(day)
                variants.extend([f"{year}年{month}月{day}日", f"{year}年{month_num}月{day_num}日"])
            except ValueError:
                pass
        return list(dict.fromkeys(variants))

    def _eval_entity_em(self, profile: Dict[str, Any], text: str) -> Dict[str, Any]:
        party_type = profile.get("type", "personal")
        if party_type == "corporate" or profile.get("representative"):
            fields_to_check = {
                "name": profile.get("name", ""),
                "address": profile.get("address", ""),
                "representative": profile.get("representative", ""),
            }
        else:
            fields_to_check = {
                "name": profile.get("name", ""),
                "gender": profile.get("gender", ""),
                "birth_date": profile.get("birth_date", ""),
                "address": profile.get("address", ""),
            }

        matched = 0
        total = 0
        details: Dict[str, Any] = {}
        for field, value in fields_to_check.items():
            if not value:
                continue
            total += 1
            if field == "birth_date":
                hit = any(variant in text for variant in self._birth_date_variants(str(value)))
            else:
                hit = str(value) in text
            if hit:
                matched += 1
            details[field] = {"value": value, "matched": hit}

        return {
            "em_score": matched / total if total else 1.0,
            "matched": matched,
            "total": total,
            "details": details,
        }

    def _parse_comp_scores(self, result: dict, keys: List[str]) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        default_reason = ""
        if isinstance(result, dict):
            default_reason = str(result.get("__judge_error__", "") or "")
        for key in keys:
            comp_result = result.get(key, {}) if isinstance(result, dict) else {}
            raw_score = comp_result.get("score", 0) if isinstance(comp_result, dict) else comp_result
            try:
                score_int = int(raw_score)
            except (TypeError, ValueError):
                score_int = 0
            reason = default_reason or "Judge metric missing"
            if isinstance(comp_result, dict):
                reason = comp_result.get("reason", "") or default_reason or "Judge metric missing"
            parsed[key] = {
                "score": score_int / 10.0,
                "raw_score": score_int,
                "reason": reason,
            }
        return parsed

    def _collect_dialog_document(
        self,
        dialog_history: List[Dict[str, Any]],
        preferred_roles: List[str],
        fallback_roles: Optional[List[str]] = None,
    ) -> str:
        messages = [entry.get("content", "") for entry in dialog_history if entry.get("role") in preferred_roles]
        if not messages and fallback_roles:
            messages = [entry.get("content", "") for entry in dialog_history if entry.get("role") in fallback_roles]
        return "\n\n".join(message for message in messages if message)

    def _extract_document(self, stage_key: str) -> Optional[str]:
        stage_results = self.pipeline_result.get("stage_results", {}).get(stage_key, {})
        document = resolve_stage_document_text(
            stage_results,
            "complaint_statement",
            "defense_statement",
            "appeal_statement",
            "appeal_response_statement",
        )
        if document:
            return document
        dialog_history = stage_results.get("dialog_history", [])
        document = self._collect_dialog_document(dialog_history, ["lawyer"], ["client"])
        return document or None

    def _determine_draft_type(self) -> str:
        stage_results = self.pipeline_result.get("stage_results", {}).get("DRAFT", {})
        if "complaint_statement" in stage_results:
            return "CD"
        if "defense_statement" in stage_results:
            return "DD"
        party_role = self.pipeline_result.get("stage_output", {}).get("party_role", "") or self.pipeline_result.get("party_role", "plaintiff")
        return "CD" if party_role == "plaintiff" else "DD"

    def _determine_appeal_draft_type(self) -> str:
        stage_results = self.pipeline_result.get("stage_results", {}).get("APPEAL_DRAFT", {})
        if "appeal_statement" in stage_results:
            return "AD"
        if "appeal_response_statement" in stage_results:
            return "AR"
        sd_result = self.pipeline_result.get("stage_results", {}).get("SD", {})
        return "AD" if sd_result.get("is_appellant", True) else "AR"

    def _resolve_party_role(self) -> str:
        party_role = (
            self.pipeline_result.get("stage_output", {}).get("party_role")
            or self.pipeline_result.get("party_role")
            or "plaintiff"
        )
        return "defendant" if str(party_role).strip().lower() == "defendant" else "plaintiff"

    @staticmethod
    def _normalize_case_key(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").strip())

    def _extract_party_name_from_case(self, case: Dict[str, Any], party_role: str) -> str:
        party_info = case.get("extracted_info", {}).get("party_info", {})
        party_raw = party_info.get(party_role, {})

        if isinstance(party_raw, list):
            for item in party_raw:
                if isinstance(item, dict):
                    name = str(item.get("name", "") or "").strip()
                    if name:
                        return name
            return ""

        if isinstance(party_raw, dict):
            return str(party_raw.get("name", "") or "").strip()

        return ""

    def _extract_lc_reference_questions_from_case(
        self,
        case: Dict[str, Any],
        party_role: str,
    ) -> List[Dict[str, str]]:
        party_info = case.get("extracted_info", {}).get("party_info", {})
        party_raw = party_info.get(party_role, {})

        if isinstance(party_raw, list):
            party_data = next(
                (
                    item
                    for item in party_raw
                    if isinstance(item, dict) and item.get("questions")
                ),
                party_raw[0] if party_raw and isinstance(party_raw[0], dict) else {},
            )
        else:
            party_data = party_raw if isinstance(party_raw, dict) else {}

        reference_questions: List[Dict[str, str]] = []
        for raw_question in party_data.get("questions", []):
            if isinstance(raw_question, dict):
                question_text = str(raw_question.get("question", "") or "").strip()
                reference_answer = str(raw_question.get("reference_answer", "") or "").strip()
            else:
                question_text = str(raw_question or "").strip()
                reference_answer = ""

            if not question_text:
                continue

            reference_questions.append(
                {
                    "question_index": len(reference_questions),
                    "question": question_text,
                    "reference_answer": reference_answer,
                }
            )

        return reference_questions

    @staticmethod
    def _lc_question_dataset_candidates() -> List[Path]:
        project_root = Path(__file__).resolve().parents[3]
        ordered: List[Path] = []

        seed_dataset = project_root / "backend" / "sandbox_seed_data" / "case_data_extracted.json"
        if seed_dataset.exists():
            ordered.append(seed_dataset.resolve())

        data_dir = project_root / "data"
        if data_dir.exists():
            for candidate in sorted(data_dir.glob("*question*.json")):
                ordered.append(candidate.resolve())

        deduped: List[Path] = []
        seen: Set[str] = set()
        for candidate in ordered:
            resolved = str(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(candidate)
        return deduped

    @staticmethod
    def _get_cached_question_loader(dataset_path: Path) -> DataLoader:
        resolved_path = str(dataset_path.resolve())
        loader = QUESTION_DATASET_LOADER_CACHE.get(resolved_path)
        if loader is None:
            loader = DataLoader(resolved_path)
            QUESTION_DATASET_LOADER_CACHE[resolved_path] = loader
        return loader

    def _load_fallback_lc_reference_questions(self) -> List[Dict[str, str]]:
        party_role = self._resolve_party_role()
        target_name = self._extract_party_name_from_case(self.source_data, party_role)
        target_background = self._normalize_case_key(
            self.source_data.get("extracted_info", {}).get("case_background", "")
        )
        loose_matches: List[tuple[List[Dict[str, str]], Path]] = []

        for dataset_path in self._lc_question_dataset_candidates():
            loader = self._get_cached_question_loader(dataset_path)
            for candidate_case in loader.cases:
                candidate_questions = self._extract_lc_reference_questions_from_case(candidate_case, party_role)
                if not candidate_questions:
                    continue

                candidate_name = self._extract_party_name_from_case(candidate_case, party_role)
                if target_name and not DataLoader._name_matches(target_name, candidate_name):
                    continue

                candidate_background = self._normalize_case_key(
                    candidate_case.get("extracted_info", {}).get("case_background", "")
                )
                if target_background and candidate_background == target_background:
                    logger.info(
                        "Resolved LC reference questions from fallback dataset %s for %s",
                        dataset_path,
                        target_name or party_role,
                    )
                    return candidate_questions

                loose_matches.append((candidate_questions, dataset_path))

        if len(loose_matches) == 1:
            questions, dataset_path = loose_matches[0]
            logger.info(
                "Resolved LC reference questions by party name fallback from %s for %s",
                dataset_path,
                target_name or party_role,
            )
            return questions

        return []

    def _extract_lc_reference_questions(self) -> List[Dict[str, str]]:
        reference_questions = self._extract_lc_reference_questions_from_case(
            self.source_data,
            self._resolve_party_role(),
        )
        if reference_questions:
            return reference_questions
        return self._load_fallback_lc_reference_questions()

    def _format_lc_dialog_history(self, dialog_history: List[Dict[str, Any]]) -> str:
        rendered_entries: List[str] = []
        for entry in dialog_history:
            content = str(entry.get("content", "") or "").strip()
            if not content:
                continue
            turn = entry.get("turn", "?")
            role = entry.get("role", "unknown")
            rendered_entries.append(f"[turn={turn}][{role}] {content}")
        return "\n".join(rendered_entries)

    def _normalize_turn_numbers(self, raw_turns: Any) -> List[int]:
        if isinstance(raw_turns, list):
            candidates = raw_turns
        elif raw_turns in (None, ""):
            candidates = []
        else:
            candidates = [raw_turns]

        normalized: List[int] = []
        for item in candidates:
            try:
                normalized.append(int(item))
            except (TypeError, ValueError):
                continue
        return sorted(dict.fromkeys(normalized))

    @staticmethod
    def _normalize_text_key(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").strip())

    @staticmethod
    def _has_nonempty_lc_answers(qa_pairs: List[Dict[str, Any]]) -> bool:
        return any(
            str(item.get("lawyer_answer", "") or "").strip()
            for item in qa_pairs
            if isinstance(item, dict)
        )

    @staticmethod
    def _normalize_lc_matching_text(value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(value or "").strip().lower())

    @classmethod
    def _extract_lc_matching_keywords(cls, question: str) -> List[str]:
        normalized = cls._normalize_lc_matching_text(question)
        if not normalized:
            return []

        keywords: List[str] = []
        for match in re.findall(r"\d+(?:\.\d+)?", normalized):
            keywords.append(match)

        domain_terms = [
            '本金',
            '借款',
            '借贷',
            '投资',
            '拿回来',
            '还钱',
            '还款',
            '利息',
            '利率',
            '月息',
            '年利率',
            '时效',
            '诉讼时效',
            '转账',
            '分红',
            '合作开发',
            '名为投资实为借贷',
        ]
        for term in domain_terms:
            normalized_term = cls._normalize_lc_matching_text(term)
            if normalized_term and normalized_term in normalized:
                keywords.append(normalized_term)

        return list(dict.fromkeys(keywords))

    def _build_lc_exchange_pairs(self, dialog_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        exchanges: List[Dict[str, Any]] = []
        pending_client: Optional[Dict[str, Any]] = None

        for entry in dialog_history:
            role = str(entry.get("role", "") or "").strip()
            content = str(entry.get("content", "") or "").strip()
            if not content:
                continue

            if role == "client":
                pending_client = {
                    "content": content,
                    "turn": entry.get("turn"),
                }
                continue

            if role != "lawyer" or pending_client is None:
                continue

            exchanges.append(
                {
                    "client_asked": pending_client["content"],
                    "lawyer_answer": content,
                    "source_turns": self._normalize_turn_numbers(entry.get("turn", [])),
                }
            )
            pending_client = None

        return exchanges

    def _score_lc_exchange_for_question(
        self,
        question: str,
        exchange: Dict[str, Any],
    ) -> float:
        client_text = self._normalize_lc_matching_text(exchange.get("client_asked", ""))
        lawyer_text = self._normalize_lc_matching_text(exchange.get("lawyer_answer", ""))
        combined_text = f"{client_text} {lawyer_text}".strip()
        if not combined_text:
            return 0.0

        score = 0.0
        question_keywords = self._extract_lc_matching_keywords(question)
        for keyword in question_keywords:
            if not keyword:
                continue
            if keyword in client_text:
                score += 4.0 if any(ch.isdigit() for ch in keyword) else 2.5
                continue
            if keyword in lawyer_text:
                score += 2.0 if any(ch.isdigit() for ch in keyword) else 1.0

        normalized_question = self._normalize_lc_matching_text(question)
        if normalized_question:
            if normalized_question in client_text:
                score += 4.0
            elif normalized_question in combined_text:
                score += 2.0

        question_bigrams = {
            normalized_question[idx: idx + 2]
            for idx in range(max(len(normalized_question) - 1, 0))
            if len(normalized_question[idx: idx + 2]) == 2
        }
        client_bigram_hits = sum(1 for gram in question_bigrams if gram in client_text)
        lawyer_bigram_hits = sum(1 for gram in question_bigrams if gram in lawyer_text)
        score += min(client_bigram_hits, 8) * 0.35
        score += min(lawyer_bigram_hits, 8) * 0.15
        return score

    def _fallback_extract_lc_qa_pairs(
        self,
        dialog_history: List[Dict[str, Any]],
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        exchanges = self._build_lc_exchange_pairs(dialog_history)
        assigned: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(reference_questions))}
        leftovers: List[Dict[str, Any]] = []

        for exchange in exchanges:
            scored = [
                (self._score_lc_exchange_for_question(item["question"], exchange), item["question_index"])
                for item in reference_questions
            ]
            best_score, best_index = max(scored, key=lambda item: item[0], default=(0.0, None))
            if best_index is None or best_score <= 0:
                leftovers.append(exchange)
                continue
            assigned[best_index].append(exchange)

        missing_indexes = [
            item["question_index"]
            for item in reference_questions
            if not assigned.get(item["question_index"])
        ]
        for question_index, exchange in zip(missing_indexes, leftovers):
            assigned[question_index].append(exchange)

        fallback_pairs: List[Dict[str, Any]] = []
        for reference_question in reference_questions:
            question_index = reference_question["question_index"]
            exchanges_for_question = assigned.get(question_index, [])
            if not exchanges_for_question:
                fallback_pairs.append(
                    {
                        "question_index": question_index,
                        "question": reference_question["question"],
                        "reference_answer": reference_question["reference_answer"],
                        "client_asked": "",
                        "lawyer_answer": "",
                        "source_turns": [],
                    }
                )
                continue

            client_asked = "\n\n".join(
                item["client_asked"]
                for item in exchanges_for_question
                if str(item.get("client_asked", "") or "").strip()
            )
            lawyer_answer = "\n\n".join(
                item["lawyer_answer"]
                for item in exchanges_for_question
                if str(item.get("lawyer_answer", "") or "").strip()
            )
            source_turns: List[int] = []
            for item in exchanges_for_question:
                source_turns.extend(item.get("source_turns", []))

            fallback_pairs.append(
                {
                    "question_index": question_index,
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "client_asked": client_asked.strip(),
                    "lawyer_answer": lawyer_answer.strip(),
                    "source_turns": sorted(dict.fromkeys(source_turns)),
                }
            )

        return fallback_pairs

    def _merge_lc_qa_pairs(
        self,
        primary_pairs: List[Dict[str, Any]],
        fallback_pairs: List[Dict[str, Any]],
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        merged_pairs: List[Dict[str, Any]] = []
        fallback_by_index = {
            item.get("question_index"): item
            for item in fallback_pairs
            if isinstance(item, dict)
        }

        for reference_question in reference_questions:
            question_index = reference_question["question_index"]
            primary = next(
                (
                    item for item in primary_pairs
                    if isinstance(item, dict) and item.get("question_index") == question_index
                ),
                None,
            ) or {
                "question_index": question_index,
                "question": reference_question["question"],
                "reference_answer": reference_question["reference_answer"],
                "client_asked": "",
                "lawyer_answer": "",
                "source_turns": [],
            }
            fallback = fallback_by_index.get(question_index, {})

            client_asked = str(primary.get("client_asked", "") or "").strip()
            lawyer_answer = str(primary.get("lawyer_answer", "") or "").strip()
            source_turns = self._normalize_turn_numbers(primary.get("source_turns", []))

            if not client_asked:
                client_asked = str(fallback.get("client_asked", "") or "").strip()
            if not lawyer_answer:
                lawyer_answer = str(fallback.get("lawyer_answer", "") or "").strip()
            if not source_turns:
                source_turns = self._normalize_turn_numbers(fallback.get("source_turns", []))

            merged_pairs.append(
                {
                    "question_index": question_index,
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "client_asked": client_asked,
                    "lawyer_answer": lawyer_answer,
                    "source_turns": source_turns,
                }
            )

        return merged_pairs

    def _normalize_lc_qa_pairs(
        self,
        payload: Any,
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        raw_pairs: List[Any] = []
        if isinstance(payload, dict):
            candidate_pairs = payload.get("qa_pairs")
            if isinstance(candidate_pairs, list):
                raw_pairs = candidate_pairs
        elif isinstance(payload, list):
            raw_pairs = payload

        question_index_by_text = {
            self._normalize_text_key(item["question"]): item["question_index"]
            for item in reference_questions
            if item.get("question")
        }
        normalized_pairs: Dict[int, Dict[str, Any]] = {}

        for raw_pair in raw_pairs:
            if not isinstance(raw_pair, dict):
                continue

            question_index: Optional[int] = None
            try:
                question_index = int(raw_pair.get("question_index"))
            except (TypeError, ValueError):
                pass

            if question_index is None or question_index not in range(len(reference_questions)):
                question_text_key = self._normalize_text_key(
                    str(
                        raw_pair.get("question")
                        or raw_pair.get("question_text")
                        or ""
                    )
                )
                question_index = question_index_by_text.get(question_text_key)

            if question_index is None or question_index not in range(len(reference_questions)):
                continue

            reference_question = reference_questions[question_index]
            client_asked = str(
                raw_pair.get("client_asked")
                or raw_pair.get("client_question")
                or raw_pair.get("client_question_summary")
                or ""
            ).strip()
            lawyer_answer = str(
                raw_pair.get("lawyer_answer")
                or raw_pair.get("answer")
                or ""
            ).strip()
            source_turns = self._normalize_turn_numbers(raw_pair.get("source_turns", []))

            existing = normalized_pairs.get(question_index)
            if existing is None:
                normalized_pairs[question_index] = {
                    "question_index": question_index,
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "client_asked": client_asked,
                    "lawyer_answer": lawyer_answer,
                    "source_turns": source_turns,
                }
                continue

            if client_asked and client_asked not in existing["client_asked"]:
                existing["client_asked"] = "\n\n".join(
                    item for item in [existing["client_asked"], client_asked] if item
                )
            if lawyer_answer and lawyer_answer not in existing["lawyer_answer"]:
                existing["lawyer_answer"] = "\n\n".join(
                    item for item in [existing["lawyer_answer"], lawyer_answer] if item
                )
            existing["source_turns"] = sorted(
                dict.fromkeys(existing["source_turns"] + source_turns)
            )

        return [
            normalized_pairs.get(
                reference_question["question_index"],
                {
                    "question_index": reference_question["question_index"],
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "client_asked": "",
                    "lawyer_answer": "",
                    "source_turns": [],
                },
            )
            for reference_question in reference_questions
        ]

    def _extract_lc_qa_pairs_with_llm(
        self,
        dialog_history: List[Dict[str, Any]],
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        transcript = self._format_lc_dialog_history(dialog_history)
        if not transcript:
            return []

        standard_questions = [
            {
                "question_index": item["question_index"],
                "question": item["question"],
            }
            for item in reference_questions
        ]
        output_schema = {
            "qa_pairs": [
                {
                    "question_index": item["question_index"],
                    "question": item["question"],
                    "client_asked": "",
                    "lawyer_answer": "",
                    "source_turns": [],
                }
                for item in reference_questions
            ]
        }
        extraction_prompt = (
            "请阅读下面的法律咨询完整聊天记录，并按给定的标准问题抽取结构化问答对。\n"
            "要求：\n"
            "1. 以标准问题为准归类，不要新增、删减或合并标准问题。\n"
            "2. 一个标准问题可能对应多轮追问、澄清和补充；请整合所有相关律师回答。\n"
            "3. `client_asked` 用当事人的实际提问原句或忠实概括。\n"
            "4. `lawyer_answer` 只保留律师对该标准问题的回答，可整合多轮，但不要补写聊天中不存在的新信息。\n"
            "5. `source_turns` 只记录支撑该答案的律师 turn 编号列表。\n"
            "6. 必须输出与标准问题数量一致的 `qa_pairs`。如果某个标准问题没有明确回答，保留空字符串和空列表。\n"
            "7. 只输出JSON。\n\n"
            f"标准问题：\n{json.dumps(standard_questions, ensure_ascii=False, indent=2)}\n\n"
            f"聊天记录：\n{transcript}\n\n"
            f"输出JSON格式：\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
        )

        extractor_agent = self._create_judge_agent(LC_QA_EXTRACTION_SYSTEM_PROMPT)
        try:
            response = self._judge_call(extractor_agent, extraction_prompt)
        finally:
            del extractor_agent

        payload = self._parse_json_payload(response)
        if payload is None:
            logger.warning("LC extraction returned invalid JSON: %s", self._truncate_text(response, 500))

        primary_pairs = self._normalize_lc_qa_pairs(payload, reference_questions)
        fallback_pairs = self._fallback_extract_lc_qa_pairs(dialog_history, reference_questions)
        merged_pairs = self._merge_lc_qa_pairs(primary_pairs, fallback_pairs, reference_questions)

        if self._has_nonempty_lc_answers(fallback_pairs) and not self._has_nonempty_lc_answers(primary_pairs):
            logger.warning(
                "LC extraction fallback used for case_id=%s because LLM extraction returned empty answers.",
                self.pipeline_result.get("case_id"),
            )

        return merged_pairs

    def _normalize_lc_qa_evals(
        self,
        payload: Any,
        reference_questions: List[Dict[str, str]],
        qa_pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        raw_results: List[Any] = []
        if isinstance(payload, dict):
            for key in ("qa_evals", "results", "evaluations"):
                candidate_results = payload.get(key)
                if isinstance(candidate_results, list):
                    raw_results = candidate_results
                    break
        elif isinstance(payload, list):
            raw_results = payload

        normalized_results: Dict[int, Dict[str, Any]] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue

            try:
                question_index = int(raw_result.get("question_index"))
            except (TypeError, ValueError):
                continue

            if question_index not in range(len(reference_questions)):
                continue

            try:
                raw_score = int(raw_result.get("score", 0))
            except (TypeError, ValueError):
                raw_score = 0
            raw_score = max(0, min(10, raw_score))
            normalized_results[question_index] = {
                "score": raw_score,
                "reason": str(raw_result.get("reason", "") or "").strip(),
            }

        qa_pair_by_index = {item["question_index"]: item for item in qa_pairs}
        qa_evals: List[Dict[str, Any]] = []
        for reference_question in reference_questions:
            question_index = reference_question["question_index"]
            qa_pair = qa_pair_by_index.get(question_index, {})
            result = normalized_results.get(question_index, {})
            raw_score = int(result.get("score", 0) or 0)
            reason = str(result.get("reason", "") or "").strip()
            if not reason:
                reason = "Judge result missing"

            qa_evals.append(
                {
                    "question_index": question_index,
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "lawyer_answer": qa_pair.get("lawyer_answer", ""),
                    "client_asked": qa_pair.get("client_asked", ""),
                    "source_turns": qa_pair.get("source_turns", []),
                    "score": raw_score,
                    "score_normalized": raw_score / 10.0,
                    "reason": reason,
                }
            )

        return qa_evals

    def _salvage_lc_qa_evals_payload(
        self,
        response: str,
        reference_questions: List[Dict[str, str]],
    ) -> Dict[str, Any] | None:
        text = str(response or "").strip()
        if not text:
            return None

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        candidate_text = fenced_blocks[0] if fenced_blocks else text
        qa_evals: List[Dict[str, Any]] = []

        for index, reference_question in enumerate(reference_questions):
            question_index = reference_question["question_index"]
            current_match = re.search(
                rf'"question_index"\s*:\s*{question_index}\b',
                candidate_text,
            )
            if current_match is None:
                return None

            current_start = current_match.start()
            next_start = len(candidate_text)
            for next_question in reference_questions[index + 1 :]:
                next_match = re.search(
                    rf'\n\s*\{{[^{{}}]*"question_index"\s*:\s*{next_question["question_index"]}\b',
                    candidate_text[current_start + 1 :],
                    re.DOTALL,
                )
                if next_match is not None:
                    next_start = current_start + 1 + next_match.start()
                    break

            item_block = candidate_text[current_start:next_start]
            score_match = re.search(r'"score"\s*:\s*(\d+)', item_block)
            reason_match = re.search(r'"reason"\s*:\s*"', item_block)
            if score_match is None or reason_match is None:
                return None

            reason_start = reason_match.end()
            reason_text = item_block[reason_start:]
            reason_text = re.sub(r'"\s*,?\s*\}?\s*$', "", reason_text, count=1, flags=re.DOTALL).strip()
            qa_evals.append(
                {
                    "question_index": question_index,
                    "score": int(score_match.group(1)),
                    "reason": reason_text,
                }
            )

        return {"qa_evals": qa_evals} if qa_evals else None

    def _salvage_single_lc_qa_eval_payload(self, response: str) -> Dict[str, Any] | None:
        text = str(response or "").strip()
        if not text:
            return None

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        candidate_text = fenced_blocks[0] if fenced_blocks else text

        score_match = re.search(r'"score"\s*:\s*(\d+)', candidate_text)
        reason_match = re.search(r'"reason"\s*:\s*"', candidate_text)
        if score_match is None or reason_match is None:
            return None

        reason_start = reason_match.end()
        reason_text = candidate_text[reason_start:]
        reason_text = re.sub(r'"\s*,?\s*\}?\s*$', "", reason_text, count=1, flags=re.DOTALL).strip()
        return {
            "score": int(score_match.group(1)),
            "reason": reason_text,
        }

    def _normalize_single_lc_qa_eval(self, payload: Any) -> Dict[str, Any] | None:
        candidate = payload
        if isinstance(payload, dict):
            for key in ("qa_eval", "result", "evaluation"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    candidate = nested
                    break

        if not isinstance(candidate, dict):
            return None

        try:
            raw_score = int(candidate.get("score", 0))
        except (TypeError, ValueError):
            raw_score = 0

        raw_score = max(0, min(10, raw_score))
        reason = str(candidate.get("reason", "") or "").strip()
        if not reason:
            return None

        return {
            "score": raw_score,
            "reason": reason,
        }

    def _evaluate_missing_lc_questions_individually(
        self,
        reference_questions: List[Dict[str, str]],
        qa_pairs: List[Dict[str, Any]],
        qa_evals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        missing_question_indices = [
            item["question_index"]
            for item in qa_evals
            if item.get("reason") == "Judge result missing"
        ]
        if not missing_question_indices:
            return qa_evals

        qa_pair_by_index = {item["question_index"]: item for item in qa_pairs}
        reference_by_index = {item["question_index"]: item for item in reference_questions}
        json_schema = json.dumps({"score": 0, "reason": ""}, ensure_ascii=False, indent=2)

        judge_agent = self._create_judge_agent(
            build_profiled_judge_system_prompt(
                "LC",
                LLM_AS_JUDGE_PROMPT_LIBRARY["LC"]["system_prompt"],
            )
        )
        try:
            recovered_results: Dict[int, Dict[str, Any]] = {}
            for question_index in missing_question_indices:
                reference_question = reference_by_index.get(question_index)
                qa_pair = qa_pair_by_index.get(question_index)
                if reference_question is None or qa_pair is None:
                    continue

                single_pair_json = json.dumps(
                    {
                        "question_index": question_index,
                        "reference_question": reference_question["question"],
                        "reference_answer": reference_question["reference_answer"],
                        "client_asked": qa_pair.get("client_asked", ""),
                        "lawyer_answer": qa_pair.get("lawyer_answer", ""),
                        "source_turns": qa_pair.get("source_turns", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                prompt = (
                    "请作为法律咨询阶段的评测法官，仅针对下面这一条结构化问答对打 0-10 分。\n"
                    "评分要求：\n"
                    f"{CURRENT_LAW_EVAL_RULE}\n"
                    "1. GT 是重要参考，但不要机械做关键词命中或逐字匹配。\n"
                    "2. 允许同义表达、合理概括、信息重组和言之成理的分析。\n"
                    "3. 重点看律师是否正面回答问题、法律分析是否正确、是否抓住关键事实和法律关系、是否对当事人有帮助。\n"
                    "4. 如果 `lawyer_answer` 为空、明显答非所问，或只停留在追问未形成实质回答，应给低分。\n"
                    "5. 只输出 JSON。\n\n"
                    f"评分档位：\n{_format_score_bands(JUDGE_SCORE_BANDS)}\n\n"
                    "结构化问答对：\n"
                    f"{single_pair_json}\n\n"
                    f"输出JSON格式：\n{json_schema}"
                )

                for attempt in range(2):
                    response = self._judge_call(judge_agent, prompt)
                    payload = self._parse_json_payload(response)
                    if payload is None:
                        payload = self._salvage_single_lc_qa_eval_payload(response)
                    normalized = self._normalize_single_lc_qa_eval(payload)
                    if normalized is not None:
                        recovered_results[question_index] = normalized
                        break
                    logger.info(
                        "LC single-question judge payload incomplete for question_index=%s attempt=%s: %s",
                        question_index,
                        attempt + 1,
                        self._truncate_text(response, 500),
                    )
        finally:
            del judge_agent

        recovered_evals: List[Dict[str, Any]] = []
        for item in qa_evals:
            recovered = recovered_results.get(item["question_index"])
            if recovered is None:
                recovered_evals.append(item)
                continue
            updated = dict(item)
            updated["score"] = recovered["score"]
            updated["score_normalized"] = recovered["score"] / 10.0
            updated["reason"] = recovered["reason"]
            recovered_evals.append(updated)

        return recovered_evals

    def _evaluate_lc_qa_pairs_with_llm(
        self,
        reference_questions: List[Dict[str, str]],
        qa_pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        structured_pairs = []
        for reference_question, qa_pair in zip(reference_questions, qa_pairs):
            structured_pairs.append(
                {
                    "question_index": reference_question["question_index"],
                    "reference_question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "client_asked": qa_pair.get("client_asked", ""),
                    "lawyer_answer": qa_pair.get("lawyer_answer", ""),
                    "source_turns": qa_pair.get("source_turns", []),
                }
            )

        output_schema = {
            "qa_evals": [
                {
                    "question_index": item["question_index"],
                    "score": 0,
                    "reason": "",
                }
                for item in reference_questions
            ]
        }
        eval_prompt = (
            "请作为法律咨询阶段的评测法官，基于下面已经抽取好的结构化问答对，对每个标准问题分别打 0-10 分。\n"
            "评分要求：\n"
            f"{CURRENT_LAW_EVAL_RULE}\n"
            "1. GT 是重要参考，但不要机械做关键词命中或逐字匹配。\n"
            "2. 允许同义表达、合理概括、信息重组和言之成理的分析。\n"
            "3. 重点看律师是否正面回答问题、法律分析是否正确、是否抓住关键事实和法律关系、是否对当事人有帮助。\n"
            "4. 如果 `lawyer_answer` 为空、明显答非所问，或只停留在追问未形成实质回答，应给低分。\n"
            "5. 请一次性输出全部问题的评分结果，只输出JSON。\n\n"
            f"评分档位：\n{_format_score_bands(JUDGE_SCORE_BANDS)}\n\n"
            f"结构化问答对：\n{json.dumps(structured_pairs, ensure_ascii=False, indent=2)}\n\n"
            f"输出JSON格式：\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
        )

        judge_agent = self._create_judge_agent(
            build_profiled_judge_system_prompt(
                "LC",
                LLM_AS_JUDGE_PROMPT_LIBRARY["LC"]["system_prompt"],
            )
        )
        last_response = ""
        payload: Any = None
        try:
            for attempt in range(2):
                response = self._judge_call(judge_agent, eval_prompt)
                last_response = response
                payload = self._parse_json_payload(response)
                if payload is None:
                    payload = self._salvage_lc_qa_evals_payload(response, reference_questions)
                normalized = self._normalize_lc_qa_evals(payload, reference_questions, qa_pairs)
                if all(item.get("reason") != "Judge result missing" for item in normalized):
                    return normalized
                logger.info(
                    "LC judge payload incomplete on attempt=%s: %s",
                    attempt + 1,
                    self._truncate_text(response, 500),
                )
        finally:
            del judge_agent

        if payload is None:
            logger.warning("LC judge returned invalid JSON: %s", self._truncate_text(last_response, 500))
        normalized = self._normalize_lc_qa_evals(payload, reference_questions, qa_pairs)
        return self._evaluate_missing_lc_questions_individually(
            reference_questions,
            qa_pairs,
            normalized,
        )

    def _normalize_lc_full_dialog_evals(
        self,
        payload: Any,
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        raw_results: List[Any] = []
        if isinstance(payload, dict):
            for key in ("qa_evals", "results", "evaluations"):
                candidate_results = payload.get(key)
                if isinstance(candidate_results, list):
                    raw_results = candidate_results
                    break
        elif isinstance(payload, list):
            raw_results = payload

        normalized_results: Dict[int, Dict[str, Any]] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue

            try:
                question_index = int(raw_result.get("question_index"))
            except (TypeError, ValueError):
                continue

            if question_index not in range(len(reference_questions)):
                continue

            try:
                raw_score = int(raw_result.get("score", 0))
            except (TypeError, ValueError):
                raw_score = 0
            raw_score = max(0, min(10, raw_score))
            normalized_results[question_index] = {
                "score": raw_score,
                "reason": str(raw_result.get("reason", "") or "").strip(),
                "evidence": raw_result.get("evidence") or raw_result.get("dialog_evidence") or "",
                "source_turns": self._normalize_turn_numbers(raw_result.get("source_turns", [])),
            }

        qa_evals: List[Dict[str, Any]] = []
        for reference_question in reference_questions:
            question_index = reference_question["question_index"]
            result = normalized_results.get(question_index, {})
            raw_score = int(result.get("score", 0) or 0)
            reason = str(result.get("reason", "") or "").strip()
            if not reason:
                reason = "Judge result missing"

            qa_evals.append(
                {
                    "question_index": question_index,
                    "question": reference_question["question"],
                    "reference_answer": reference_question["reference_answer"],
                    "score": raw_score,
                    "score_normalized": raw_score / 10.0,
                    "reason": reason,
                    "evidence": result.get("evidence", ""),
                    "source_turns": result.get("source_turns", []),
                }
            )

        return qa_evals

    def _evaluate_missing_lc_questions_from_full_dialog(
        self,
        dialog_history: List[Dict[str, Any]],
        reference_questions: List[Dict[str, str]],
        qa_evals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        missing_question_indices = [
            item["question_index"]
            for item in qa_evals
            if item.get("reason") == "Judge result missing"
        ]
        if not missing_question_indices:
            return qa_evals

        transcript = self._format_lc_dialog_history(dialog_history)
        reference_by_index = {item["question_index"]: item for item in reference_questions}
        json_schema = json.dumps(
            {"score": 0, "reason": "", "evidence": "", "source_turns": []},
            ensure_ascii=False,
            indent=2,
        )

        judge_agent = self._create_judge_agent(
            build_profiled_judge_system_prompt(
                "LC",
                LLM_AS_JUDGE_PROMPT_LIBRARY["LC"]["system_prompt"],
            )
        )
        try:
            recovered_results: Dict[int, Dict[str, Any]] = {}
            for question_index in missing_question_indices:
                reference_question = reference_by_index.get(question_index)
                if reference_question is None:
                    continue

                single_question_payload = json.dumps(
                    {
                        "question_index": question_index,
                        "question": reference_question["question"],
                        "reference_answer": reference_question["reference_answer"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                prompt = (
                    "请作为法律咨询阶段评测法官，直接阅读完整聊天记录，并只针对下面这一条标准问题打 0-10 分。\n"
                    "不要先抽取结构化问答；请直接判断完整聊天记录中律师是否实质回答了该标准问题。\n"
                    "评分要求：\n"
                    f"{CURRENT_LAW_EVAL_RULE}\n"
                    "1. 参考答案是重要参照，但不要机械关键词匹配。\n"
                    "2. 如果律师在多轮对话中分散回答了问题，应合并理解后评分。\n"
                    "3. 如果律师只追问、答非所问、或没有形成实质法律分析，应给低分。\n"
                    "4. `evidence` 简要摘录或概括你依据的聊天片段，`source_turns` 填相关 turn 编号。\n"
                    "5. 只输出 JSON。\n\n"
                    f"评分档位：\n{_format_score_bands(JUDGE_SCORE_BANDS)}\n\n"
                    f"标准问题与参考答案：\n{single_question_payload}\n\n"
                    f"完整咨询聊天记录：\n{transcript}\n\n"
                    f"输出JSON格式：\n{json_schema}"
                )

                for attempt in range(2):
                    response = self._judge_call(judge_agent, prompt)
                    payload = self._parse_json_payload(response)
                    if payload is None:
                        payload = self._salvage_single_lc_qa_eval_payload(response)
                    normalized = self._normalize_single_lc_qa_eval(payload)
                    if normalized is not None:
                        normalized["evidence"] = (
                            payload.get("evidence", "")
                            if isinstance(payload, dict)
                            else ""
                        )
                        normalized["source_turns"] = self._normalize_turn_numbers(
                            payload.get("source_turns", []) if isinstance(payload, dict) else []
                        )
                        recovered_results[question_index] = normalized
                        break
                    logger.info(
                        "LC full-dialog single-question judge payload incomplete for question_index=%s attempt=%s: %s",
                        question_index,
                        attempt + 1,
                        self._truncate_text(response, 500),
                    )
        finally:
            del judge_agent

        recovered_evals: List[Dict[str, Any]] = []
        for item in qa_evals:
            recovered = recovered_results.get(item["question_index"])
            if recovered is None:
                recovered_evals.append(item)
                continue
            updated = dict(item)
            updated["score"] = recovered["score"]
            updated["score_normalized"] = recovered["score"] / 10.0
            updated["reason"] = recovered["reason"]
            updated["evidence"] = recovered.get("evidence", "")
            updated["source_turns"] = recovered.get("source_turns", [])
            recovered_evals.append(updated)

        return recovered_evals

    def _evaluate_lc_full_dialog_with_llm(
        self,
        dialog_history: List[Dict[str, Any]],
        reference_questions: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        transcript = self._format_lc_dialog_history(dialog_history)
        reference_payload = [
            {
                "question_index": item["question_index"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
            }
            for item in reference_questions
        ]
        output_schema = {
            "qa_evals": [
                {
                    "question_index": item["question_index"],
                    "score": 0,
                    "reason": "",
                    "evidence": "",
                    "source_turns": [],
                }
                for item in reference_questions
            ]
        }
        eval_prompt = (
            "请作为法律咨询阶段评测法官，直接阅读完整咨询聊天记录，并根据给定标准问题与参考答案逐项打 0-10 分。\n"
            "不要先抽取结构化问答；不要要求标准问题必须由当事人逐字问出。真实咨询中问题可能分散在多轮事实陈述、追问和补充中。\n"
            "评分要求：\n"
            f"{CURRENT_LAW_EVAL_RULE}\n"
            "1. 对每个标准问题，综合完整聊天记录判断律师是否实质回答了该问题。\n"
            "2. 参考答案是重要参照，但不要机械关键词匹配；允许同义表达、合理概括和顺序重组。\n"
            "3. 重点看律师是否正面回应问题、法律分析是否正确、是否抓住关键事实和法律关系、是否对当事人有帮助。\n"
            "4. 如果律师只追问、答非所问、或没有形成实质法律分析，应给低分。\n"
            "5. `evidence` 简要摘录或概括你依据的聊天片段，`source_turns` 填相关 turn 编号。\n"
            "6. 必须一次性输出所有标准问题的评分结果，只输出 JSON。\n\n"
            f"评分档位：\n{_format_score_bands(JUDGE_SCORE_BANDS)}\n\n"
            f"标准问题与参考答案：\n{json.dumps(reference_payload, ensure_ascii=False, indent=2)}\n\n"
            f"完整咨询聊天记录：\n{transcript}\n\n"
            f"输出JSON格式：\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
        )

        judge_agent = self._create_judge_agent(
            build_profiled_judge_system_prompt(
                "LC",
                LLM_AS_JUDGE_PROMPT_LIBRARY["LC"]["system_prompt"],
            )
        )
        last_response = ""
        payload: Any = None
        try:
            for attempt in range(2):
                response = self._judge_call(judge_agent, eval_prompt)
                last_response = response
                payload = self._parse_json_payload(response)
                if payload is None:
                    payload = self._salvage_lc_qa_evals_payload(response, reference_questions)
                normalized = self._normalize_lc_full_dialog_evals(payload, reference_questions)
                if all(item.get("reason") != "Judge result missing" for item in normalized):
                    return normalized
                logger.info(
                    "LC full-dialog judge payload incomplete on attempt=%s: %s",
                    attempt + 1,
                    self._truncate_text(response, 500),
                )
        finally:
            del judge_agent

        if payload is None:
            logger.warning("LC full-dialog judge returned invalid JSON: %s", self._truncate_text(last_response, 500))
        normalized = self._normalize_lc_full_dialog_evals(payload, reference_questions)
        return self._evaluate_missing_lc_questions_from_full_dialog(
            dialog_history,
            reference_questions,
            normalized,
        )

    def _evaluate_components_with_judge(
        self,
        stage_code: str,
        sections: List[tuple[str, str]],
        metric_names: List[str],
    ) -> Dict[str, Any]:
        json_schema = json.dumps(
            {name: {"score": 0, "reason": ""} for name in metric_names},
            ensure_ascii=False,
            indent=2,
        )
        prod_eval_prompt = _build_stage_metric_prompt(stage_code, metric_names)
        eval_prompt = build_profiled_judge_eval_prompt(
            stage_code,
            prod_eval_prompt,
            sections=sections,
            json_schema=json_schema,
        )
        system_prompt = build_profiled_judge_system_prompt(
            stage_code,
            LLM_AS_JUDGE_PROMPT_LIBRARY[stage_code]["system_prompt"],
        )

        last_issue = ""
        for attempt in range(2):
            judge_agent = self._create_judge_agent(system_prompt)
            try:
                response = self._judge_call(judge_agent, eval_prompt)
            finally:
                del judge_agent

            judge_result = self._parse_judge_response(response)
            salvaged_metric_payload = self._salvage_metric_payload_from_response(response, metric_names)
            if salvaged_metric_payload is not None:
                judge_result = salvaged_metric_payload
            resolved_metric_payload = self._resolve_metric_payload(judge_result, metric_names)
            if resolved_metric_payload is not None:
                if attempt > 0:
                    logger.info(
                        "Judge retry succeeded for stage=%s metrics=%s",
                        stage_code,
                        metric_names,
                    )
                return self._parse_comp_scores(resolved_metric_payload, metric_names)

            last_issue = self._describe_metric_payload_issue(response, judge_result, metric_names)
            log_fn = logger.warning if attempt == 1 else logger.info
            log_fn(
                "Invalid judge payload for stage=%s attempt=%s metrics=%s: %s",
                stage_code,
                attempt + 1,
                metric_names,
                last_issue,
            )

        individual_metric_payload: Dict[str, Any] = {}
        for metric_name in metric_names:
            single_schema = json.dumps(
                {metric_name: {"score": 0, "reason": ""}},
                ensure_ascii=False,
                indent=2,
            )
            single_prompt = build_profiled_judge_eval_prompt(
                stage_code,
                _build_stage_metric_prompt(stage_code, [metric_name]),
                sections=sections,
                json_schema=single_schema,
            )

            recovered_metric: Dict[str, Any] | None = None
            single_issue = ""
            for attempt in range(2):
                judge_agent = self._create_judge_agent(system_prompt)
                try:
                    response = self._judge_call(judge_agent, single_prompt)
                finally:
                    del judge_agent

                judge_result = self._parse_judge_response(response)
                salvaged_metric_payload = self._salvage_metric_payload_from_response(response, [metric_name])
                if salvaged_metric_payload is not None:
                    judge_result = salvaged_metric_payload
                resolved_metric_payload = self._resolve_metric_payload(judge_result, [metric_name])
                if resolved_metric_payload is not None:
                    recovered_metric = resolved_metric_payload.get(metric_name)
                    break

                single_issue = self._describe_metric_payload_issue(response, judge_result, [metric_name])
                logger.info(
                    "Single-metric judge payload incomplete for stage=%s metric=%s attempt=%s: %s",
                    stage_code,
                    metric_name,
                    attempt + 1,
                    single_issue,
                )

            if isinstance(recovered_metric, dict) and "score" in recovered_metric:
                individual_metric_payload[metric_name] = recovered_metric
            else:
                individual_metric_payload[metric_name] = {
                    "score": 0,
                    "reason": single_issue or last_issue or "Judge result missing",
                }

        return self._parse_comp_scores(individual_metric_payload, metric_names)

    def _eval_draft_cd(self, document: str) -> Dict[str, Any]:
        plaintiff_profile = self.data_loader.extract_plaintiff_profile(self.source_data)
        defendant_profile = self.data_loader.extract_defendant_profile(self.source_data)
        plaintiff_em = self._eval_entity_em(plaintiff_profile, document)
        defendant_em = self._eval_entity_em(defendant_profile, document)

        candidate_document = self._strip_draft_end_marker(document)

        gt_claims = self.data_loader.extract_claims(self.source_data)
        gt_facts = self.data_loader.extract_facts_and_reasons(self.source_data)
        gt_evidence = self.data_loader.extract_plaintiff_evidence(self.source_data)

        metrics: List[str] = []
        sections: List[tuple[str, str]] = []
        if gt_claims:
            metrics.append("诉讼请求")
            sections.append(("参考诉讼请求", gt_claims))
        if gt_facts:
            metrics.append("事实与理由")
            sections.append(("参考事实与理由", gt_facts))
        if gt_evidence:
            metrics.append("证据")
            sections.append(("参考证据", gt_evidence))
        if metrics:
            sections.append(("候选完整起诉状", candidate_document))

        claims_eval = self._empty_component_eval()
        facts_eval = self._empty_component_eval()
        evidence_eval = self._empty_component_eval()
        if metrics:
            metric_result = self._evaluate_components_with_judge("CD", sections, metrics)
            claims_eval = metric_result.get("诉讼请求", claims_eval)
            facts_eval = metric_result.get("事实与理由", facts_eval)
            evidence_eval = metric_result.get("证据", evidence_eval)

        doc_score = self._average([
            plaintiff_em["em_score"],
            defendant_em["em_score"],
            claims_eval.get("score"),
            facts_eval.get("score"),
            evidence_eval.get("score"),
        ])
        return {
            "draft_type": "CD",
            "stage_score": doc_score,
            "DOC": {
                "doc_score": doc_score,
                "plaintiff_em": plaintiff_em,
                "defendant_em": defendant_em,
                "claims_eval": claims_eval,
                "facts_eval": facts_eval,
                "evidence_eval": evidence_eval,
            },
        }

    def _eval_draft_dd(self, document: str) -> Dict[str, Any]:
        defendant_profile = self.data_loader.extract_defendant_profile(self.source_data)
        defendant_em = self._eval_entity_em(defendant_profile, document)

        candidate_document = self._strip_draft_end_marker(document)
        gt_plea = self.data_loader.extract_defendant_defense(self.source_data)
        gt_evidence = self.data_loader.extract_defendant_evidence(self.source_data)

        metrics: List[str] = []
        sections: List[tuple[str, str]] = []
        if gt_plea:
            metrics.append("答辩意见")
            sections.append(("参考答辩意见", gt_plea))
        if gt_evidence:
            metrics.append("证据")
            sections.append(("参考证据", gt_evidence))
        if metrics:
            sections.append(("候选完整答辩状", candidate_document))

        plea_eval = self._empty_component_eval()
        evidence_eval = self._empty_component_eval()
        if metrics:
            metric_result = self._evaluate_components_with_judge("DD", sections, metrics)
            plea_eval = metric_result.get("答辩意见", plea_eval)
            evidence_eval = metric_result.get("证据", evidence_eval)

        doc_score = self._average([
            defendant_em["em_score"],
            plea_eval.get("score"),
            evidence_eval.get("score"),
        ])
        return {
            "draft_type": "DD",
            "stage_score": doc_score,
            "DOC": {
                "doc_score": doc_score,
                "defendant_em": defendant_em,
                "plea_eval": plea_eval,
                "evidence_eval": evidence_eval,
            },
        }

    def _eval_draft_ad(self, document: str) -> Dict[str, Any]:
        extracted_info = self.source_data.get("extracted_info", {})
        appellant_role = extracted_info.get("appellant", "")
        if appellant_role == "原告":
            appellant_profile = self.data_loader.extract_plaintiff_profile(self.source_data)
            appellee_profile = self.data_loader.extract_defendant_profile(self.source_data)
        else:
            appellant_profile = self.data_loader.extract_defendant_profile(self.source_data)
            appellee_profile = self.data_loader.extract_plaintiff_profile(self.source_data)

        appellant_em = self._eval_entity_em(appellant_profile, document)
        appellee_em = self._eval_entity_em(appellee_profile, document)

        candidate_document = self._strip_draft_end_marker(document)

        appellant_appeal = self.data_loader.extract_appellant_appeal(self.source_data)
        gt_claims_raw = appellant_appeal.get("claim", [])
        gt_claims = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(gt_claims_raw)) if isinstance(gt_claims_raw, list) else str(gt_claims_raw or "")
        gt_reasons = appellant_appeal.get("reasons", "")
        gt_evidence = self.data_loader.extract_second_instance_evidence(self.source_data, side="appellant")

        metrics: List[str] = []
        sections: List[tuple[str, str]] = []
        if gt_claims:
            metrics.append("上诉请求")
            sections.append(("参考上诉请求", gt_claims))
        if gt_reasons:
            metrics.append("事实与理由")
            sections.append(("参考上诉理由", gt_reasons))
        if gt_evidence:
            metrics.append("证据")
            sections.append(("参考新证据", gt_evidence))
        if metrics:
            sections.append(("候选完整上诉状", candidate_document))

        claims_eval = self._empty_component_eval()
        reasons_eval = self._empty_component_eval()
        evidence_eval = self._empty_component_eval()
        if metrics:
            metric_result = self._evaluate_components_with_judge("AD", sections, metrics)
            claims_eval = metric_result.get("上诉请求", claims_eval)
            reasons_eval = metric_result.get("事实与理由", reasons_eval)
            evidence_eval = metric_result.get("证据", evidence_eval)

        doc_score = self._average([
            appellant_em["em_score"],
            appellee_em["em_score"],
            claims_eval.get("score"),
            reasons_eval.get("score"),
            evidence_eval.get("score"),
        ])
        return {
            "draft_type": "AD",
            "stage_score": doc_score,
            "DOC": {
                "doc_score": doc_score,
                "appellant_em": appellant_em,
                "appellee_em": appellee_em,
                "claims_eval": claims_eval,
                "reasons_eval": reasons_eval,
                "evidence_eval": evidence_eval,
            },
        }

    def _eval_draft_ar(self, document: str) -> Dict[str, Any]:
        extracted_info = self.source_data.get("extracted_info", {})
        appellant_role = extracted_info.get("appellant", "")
        if appellant_role == "原告":
            appellee_profile = self.data_loader.extract_defendant_profile(self.source_data)
        else:
            appellee_profile = self.data_loader.extract_plaintiff_profile(self.source_data)
        appellee_em = self._eval_entity_em(appellee_profile, document)

        candidate_document = self._strip_draft_end_marker(document)
        gt_plea = self.data_loader.extract_second_instance_appellee_defense(self.source_data)
        gt_evidence = self.data_loader.extract_second_instance_evidence(self.source_data, side="appellee")

        metrics: List[str] = []
        sections: List[tuple[str, str]] = []
        if gt_plea:
            metrics.append("答辩意见")
            sections.append(("参考答辩意见", gt_plea))
        if gt_evidence:
            metrics.append("证据")
            sections.append(("参考新证据", gt_evidence))
        if metrics:
            sections.append(("候选完整上诉答辩状", candidate_document))

        plea_eval = self._empty_component_eval()
        evidence_eval = self._empty_component_eval()
        if metrics:
            metric_result = self._evaluate_components_with_judge("AR", sections, metrics)
            plea_eval = metric_result.get("答辩意见", plea_eval)
            evidence_eval = metric_result.get("证据", evidence_eval)

        doc_score = self._average([
            appellee_em["em_score"],
            plea_eval.get("score"),
            evidence_eval.get("score"),
        ])
        return {
            "draft_type": "AR",
            "stage_score": doc_score,
            "DOC": {
                "doc_score": doc_score,
                "appellee_em": appellee_em,
                "plea_eval": plea_eval,
                "evidence_eval": evidence_eval,
            },
        }

    def _eval_ci(self) -> Dict[str, Any]:
        return self._eval_ci_aligned()

    def _eval_cia(self) -> Dict[str, Any]:
        return self._eval_cia_aligned()

    def _eval_ci_aligned(self) -> Dict[str, Any]:
        party_role = self.pipeline_result.get("stage_output", {}).get("party_role", "") or self.pipeline_result.get("party_role", "plaintiff")
        is_plaintiff = party_role == "plaintiff"
        dialog_history = self.pipeline_result.get("stage_results", {}).get("CI", {}).get("dialog_history", [])
        document = self._collect_dialog_document(
            dialog_history,
            ["plaintiff_lawyer"] if is_plaintiff else ["defendant_lawyer"],
            ["plaintiff"] if is_plaintiff else ["defendant"],
        )
        if not document.strip():
            return {"stage_score": 0, "error": "CI dialog history is empty"}

        if is_plaintiff:
            gt_claim_or_plea = self.data_loader.extract_claims(self.source_data)
            gt_label = "诉讼请求"
            evidence_text = self.data_loader.extract_evidence(self.source_data)
            dispute_text = self.data_loader.extract_all_evidence_disputes(self.source_data)
            role_name = "plaintiff"
        else:
            gt_claim_or_plea = self.data_loader.extract_defendant_defense(self.source_data)
            gt_label = "答辩意见"
            evidence_text = self.data_loader.extract_evidence(self.source_data)
            dispute_text = self.data_loader.extract_all_evidence_disputes(self.source_data)
            role_name = "defendant"

        finding_text = self.data_loader.extract_first_instance_info(self.source_data).get("court_finding", "")
        opinion_text = self.data_loader.extract_court_opinion(self.source_data)
        metrics = [
            "诉讼与答辩一致性",
            "事实与证据运用完整性",
            "法律说理充分性",
        ]
        sections = [
            (f"参考{gt_label}", gt_claim_or_plea or ""),
            ("参考全案证据", evidence_text or ""),
            ("参考全案质证意见", dispute_text or ""),
            ("参考法院查明", finding_text or ""),
            ("参考法院意见", opinion_text or ""),
            ("候选庭审发言（含举证质证）", document),
        ]
        metric_result = self._evaluate_components_with_judge("CI", sections, metrics)
        stage_score = self._average([metric_result[name]["score"] for name in metrics])
        return {"stage_score": stage_score, "metrics": metric_result, "role": role_name}

    def _eval_cia_aligned(self) -> Dict[str, Any]:
        is_appellant = self.pipeline_result.get("stage_results", {}).get("SD", {}).get("is_appellant", True)
        dialog_history = self.pipeline_result.get("stage_results", {}).get("CIA", {}).get("dialog_history", [])
        document = self._collect_dialog_document(
            dialog_history,
            ["appellant_lawyer"] if is_appellant else ["appellee_lawyer"],
            ["appellant"] if is_appellant else ["appellee"],
        )
        if not document.strip():
            return {"stage_score": 0, "error": "CIA dialog history is empty"}

        if is_appellant:
            appellant_appeal = self.data_loader.extract_appellant_appeal(self.source_data)
            gt_claims_raw = appellant_appeal.get("claim", [])
            gt_claim_or_plea = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(gt_claims_raw)) if isinstance(gt_claims_raw, list) else str(gt_claims_raw or "")
            gt_reasons = appellant_appeal.get("reasons", "")
            if gt_reasons:
                gt_claim_or_plea = f"{gt_claim_or_plea}\n\n上诉理由：\n{gt_reasons}" if gt_claim_or_plea else f"上诉理由：\n{gt_reasons}"
            gt_label = "上诉请求与理由"
            role_name = "appellant"
        else:
            gt_claim_or_plea = self.data_loader.extract_second_instance_appellee_defense(self.source_data)
            gt_label = "答辩意见"
            role_name = "appellee"

        evidence_text = self.data_loader.extract_all_second_instance_evidence(self.source_data)
        dispute_text = self.data_loader.extract_all_second_instance_evidence_disputes(self.source_data)
        second_info = self.data_loader.extract_second_instance_info(self.source_data)
        finding_text = second_info.get("court_finding", "")
        opinion_text = self.data_loader.extract_second_instance_court_opinion(self.source_data)
        metrics = [
            "上诉与答辩一致性",
            "事实与证据运用完整性",
            "法律说理充分性",
        ]
        sections = [
            (f"参考{gt_label}", gt_claim_or_plea or ""),
            ("参考全案新证据", evidence_text or ""),
            ("参考全案质证意见", dispute_text or ""),
            ("参考法院查明", finding_text or ""),
            ("参考法院意见", opinion_text or ""),
            ("候选庭审发言（含举证质证）", document),
        ]
        metric_result = self._evaluate_components_with_judge("CIA", sections, metrics)
        stage_score = self._average([metric_result[name]["score"] for name in metrics])
        return {"stage_score": stage_score, "metrics": metric_result, "role": role_name}

    def _compute_weighted_overall_summary(
        self,
        scored_stages: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        weighted_stage_scores: Dict[str, Any] = {}
        overall_score = 0.0
        overall_possible_score = 0

        for original_stage_code, stage_result in scored_stages.items():
            actual_stage_code = self._resolve_progress_stage(original_stage_code)
            weight = BENCHMARK_STAGE_WEIGHTS.get(actual_stage_code, 0)
            stage_score = stage_result.get("stage_score")
            if stage_score is None or weight <= 0:
                continue

            weighted_score = float(stage_score) * weight
            overall_score += weighted_score
            overall_possible_score += weight
            weighted_stage_scores[actual_stage_code] = {
                "original_stage_code": original_stage_code,
                "stage_score": float(stage_score),
                "weight": weight,
                "weighted_score": weighted_score,
            }

        return {
            "overall_score": overall_score if overall_possible_score else None,
            "overall_possible_score": overall_possible_score,
            "overall_score_normalized": (
                overall_score / overall_possible_score if overall_possible_score else None
            ),
            "weighted_stage_scores": weighted_stage_scores,
        }

    def _eval_lc(self) -> Dict[str, Any]:
        lc_result = self.pipeline_result.get("stage_results", {}).get("LC", {})
        dialog_history = lc_result.get("dialog_history", [])
        if not dialog_history:
            return {"stage_score": 0, "error": "LC dialog history missing", "qa_evals": []}

        reference_questions = self._extract_lc_reference_questions()
        if not reference_questions:
            return {"stage_score": 0, "error": "LC questions missing", "qa_evals": []}

        qa_evals = self._evaluate_lc_full_dialog_with_llm(dialog_history, reference_questions)
        total_score = sum(item.get("score", 0) for item in qa_evals)

        avg_score = total_score / len(qa_evals) if qa_evals else 0
        return {
            "stage_score": avg_score / 10.0,
            "avg_raw_score": avg_score,
            "total_questions": len(qa_evals),
            "party_role": self._resolve_party_role(),
            "evaluation_mode": "full_dialog_direct_judge",
            "qa_evals": qa_evals,
        }

    def run(self) -> Dict[str, Any]:
        start_time = datetime.now()
        case_id = (
            self.pipeline_result.get("case_id")
            or self.source_data.get("id")
            or self.source_data.get("original_id")
        )

        if self._should_eval_stage("LC"):
            self._notify_stage_started("LC")
            self.eval_results["LC"] = self._eval_lc()
            self._mark_stage_done("LC")

        if self._should_eval_stage("DRAFT"):
            self._notify_stage_started("DRAFT")
            document = self._extract_document("DRAFT")
            if document:
                draft_type = self._determine_draft_type()
                self.eval_results["DRAFT"] = self._eval_draft_cd(document) if draft_type == "CD" else self._eval_draft_dd(document)
            else:
                self.eval_results["DRAFT"] = {"stage_score": 0, "error": "draft document missing"}
            self._mark_stage_done("DRAFT")

        if self._should_eval_stage("CI"):
            self._notify_stage_started("CI")
            self.eval_results["CI"] = self._eval_ci()
            self._mark_stage_done("CI")

        if self._should_eval_stage("SD"):
            self._notify_stage_started("SD")
            self.eval_results["SD"] = {"stage_score": None, "status": "not_implemented"}
            self._mark_stage_done("SD")

        if self._should_eval_stage("APPEAL_DRAFT"):
            self._notify_stage_started("APPEAL_DRAFT")
            document = self._extract_document("APPEAL_DRAFT")
            if document:
                draft_type = self._determine_appeal_draft_type()
                self.eval_results["APPEAL_DRAFT"] = self._eval_draft_ad(document) if draft_type == "AD" else self._eval_draft_ar(document)
            else:
                self.eval_results["APPEAL_DRAFT"] = {"stage_score": 0, "error": "appeal draft document missing"}
            self._mark_stage_done("APPEAL_DRAFT")

        if self._should_eval_stage("CIA"):
            self._notify_stage_started("CIA")
            self.eval_results["CIA"] = self._eval_cia()
            self._mark_stage_done("CIA")

        duration = (datetime.now() - start_time).total_seconds()
        scored_stages = {
            stage: result
            for stage, result in self.eval_results.items()
            if result.get("stage_score") is not None
        }
        weighted_summary = self._compute_weighted_overall_summary(scored_stages)
        overall_score = weighted_summary.get("overall_score")
        overall_possible_score = weighted_summary.get("overall_possible_score")
        overall_score_normalized = weighted_summary.get("overall_score_normalized")
        overall_score_unweighted = (
            self._average([result["stage_score"] for result in scored_stages.values()])
            if scored_stages
            else None
        )

        final_result = {
            "case_id": case_id,
            "case_cause": self.pipeline_result.get("case_cause", ""),
            "eval_time": datetime.now().isoformat(),
            "eval_duration_seconds": duration,
            "judge_model": str(self.judge_model_type),
            "stages_evaluated": list(self.eval_results.keys()),
            "overall_score": overall_score,
            "overall_possible_score": overall_possible_score,
            "overall_score_normalized": overall_score_normalized,
            "overall_score_unweighted": overall_score_unweighted,
            "weighted_stage_scores": weighted_summary.get("weighted_stage_scores", {}),
            "stage_eval_results": self.eval_results,
        }

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        return final_result
