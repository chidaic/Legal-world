"""Closing summary and AI evaluation for player-lawyer runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .run_ledger import PlayerRunLedger


EVALUATION_DIMENSIONS = (
    ("事实把握", 25),
    ("法律论证", 25),
    ("程序/任务完成", 25),
    ("表达与职业沟通", 25),
)


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw[3:-3].strip()
        lowered = inner.lower()
        if lowered.startswith("json"):
            inner = inner[4:].strip()
        return inner.strip()
    return raw


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_generator(prompt: str) -> str:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    from ..utils.model_config import build_runtime_openai_chat_config, resolve_openai_chat_model

    model_name = resolve_openai_chat_model()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_name,
        model_config_dict=build_runtime_openai_chat_config(
            model_name=model_name,
            temperature=0.2,
            max_tokens=1000,
        ),
    )
    agent = ChatAgent(
        system_message="你是法律实训系统的玩家表现评价教练，只输出 JSON，不输出解释。",
        model=model,
    )
    response = agent.step(BaseMessage.make_user_message(role_name="user", content=prompt))
    return response.msgs[0].content


class PlayerClosingSummaryService:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        generator: Callable[[str], str] | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.generator = generator or _default_generator
        self.ledger = PlayerRunLedger(storage_root=self.storage_root)

    def build_summary(
        self,
        *,
        case_id: str,
        case_entry: dict[str, Any] | None = None,
        documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_case_id = str(case_id or "").strip()
        player_turns = self._load_player_turns(normalized_case_id)
        available_documents = [
            dict(item)
            for item in list(documents or [])
            if bool(item.get("available"))
        ]
        evaluation = self._load_cached_evaluation(normalized_case_id)
        return {
            "case_id": normalized_case_id,
            "case": self._normalize_case_entry(case_entry or {}),
            "documents": list(documents or []),
            "document_count": len(available_documents),
            "player_turns": player_turns,
            "player_turn_count": len(player_turns),
            "evaluation": evaluation,
        }

    def generate_evaluation(
        self,
        *,
        case_id: str,
        case_entry: dict[str, Any] | None = None,
        documents: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summary = self.build_summary(
            case_id=case_id,
            case_entry=case_entry or {},
            documents=documents or [],
        )
        prompt = self._build_evaluation_prompt(summary)
        raw = _strip_code_fences(self.generator(prompt))
        evaluation = self._normalize_evaluation_payload(self._parse_json_object(raw))
        evaluation["generated_at"] = datetime.utcnow().isoformat()
        self._save_cached_evaluation(str(case_id or "").strip(), evaluation)
        self.ledger.record_evaluation(case_id=str(case_id or "").strip(), evaluation=evaluation)
        return evaluation

    def _case_output_dir(self, case_id: str) -> Path:
        return self.storage_root / "output" / str(case_id or "").strip()

    def _player_dir(self, case_id: str) -> Path:
        return self._case_output_dir(case_id) / "_player_lawyer"

    def _evaluation_path(self, case_id: str) -> Path:
        return self._player_dir(case_id) / "closing_evaluation.json"

    def _load_cached_evaluation(self, case_id: str) -> dict[str, Any] | None:
        path = self._evaluation_path(case_id)
        if not path.exists():
            return None
        payload = _read_json(path)
        return payload or None

    def _save_cached_evaluation(self, case_id: str, evaluation: dict[str, Any]) -> Path:
        path = self._evaluation_path(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _load_player_turns(self, case_id: str) -> list[dict[str, Any]]:
        return self.ledger.load_player_turns(case_id)

    def _normalize_case_entry(self, case_entry: dict[str, Any]) -> dict[str, str]:
        return {
            "title": str(case_entry.get("title") or "").strip(),
            "plaintiff_name": str(case_entry.get("plaintiff_name") or case_entry.get("plaintiffName") or "").strip(),
            "defendant_name": str(case_entry.get("defendant_name") or case_entry.get("defendantName") or "").strip(),
            "training_category": str(case_entry.get("training_category") or case_entry.get("trainingCategory") or "").strip(),
            "difficulty": str(case_entry.get("difficulty") or "").strip(),
        }

    def _build_evaluation_prompt(self, summary: dict[str, Any]) -> str:
        case_info = summary.get("case") if isinstance(summary.get("case"), dict) else {}
        player_turns = summary.get("player_turns") if isinstance(summary.get("player_turns"), list) else []
        turn_lines = []
        for index, turn in enumerate(player_turns, start=1):
            if not isinstance(turn, dict):
                continue
            stage = str(turn.get("stage") or "").strip()
            prompt = str(turn.get("prompt") or "").strip()
            final_message = str(turn.get("final_message") or "").strip()
            turn_lines.append(f"{index}. [{stage}] 任务：{prompt}\n提交：{final_message}")
        player_text = "\n\n".join(turn_lines) or "本轮没有记录到玩家提交。"
        document_titles = [
            str(item.get("title") or item.get("document_key") or "").strip()
            for item in list(summary.get("documents") or [])
            if isinstance(item, dict) and item.get("available")
        ]
        documents_text = "、".join([item for item in document_titles if item]) or "无可用文书。"
        return (
            "请评价玩家在法律实训系统中作为当前玩家方律师的最终提交质量和全流程表现。\n"
            "评分主体包括玩家最终提交的普通发言、确认的文书、追问当事人的问题、阶段职责完成情况。\n"
            "完整对话只作为上下文，不要把对手方或法官 AI 表现计入玩家得分。\n"
            "只评价玩家提交文本与任务完成质量，不评价也不扣减临时 AI 代答或润色使用情况。\n"
            "请输出严格 JSON，不要 Markdown、解释或代码块。JSON 格式：\n"
            "{\n"
            '  "overall_score": 0-100整数,\n'
            '  "summary": "80字以内总评",\n'
            '  "dimensions": [\n'
            '    {"label":"事实把握","score":0-25整数,"max_score":25},\n'
            '    {"label":"法律论证","score":0-25整数,"max_score":25},\n'
            '    {"label":"程序/任务完成","score":0-25整数,"max_score":25},\n'
            '    {"label":"表达与职业沟通","score":0-25整数,"max_score":25}\n'
            "  ],\n"
            '  "strengths": ["优点1","优点2"],\n'
            '  "improvements": ["建议1","建议2"]\n'
            "}\n\n"
            f"【案件】{case_info.get('title') or summary.get('case_id')}\n"
            f"【当事人】原告：{case_info.get('plaintiff_name') or '未知'}；被告：{case_info.get('defendant_name') or '未知'}\n"
            f"【生成文书】{documents_text}\n\n"
            f"【玩家提交记录】\n{player_text}\n"
        )

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise ValueError("Closing evaluation did not return JSON.")
            payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError("Closing evaluation JSON must be an object.")
        return payload

    def _normalize_evaluation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        dimensions_by_label = {
            str(item.get("label") or "").strip(): item
            for item in list(payload.get("dimensions") or [])
            if isinstance(item, dict)
        }
        dimensions = []
        total = 0
        for label, max_score in EVALUATION_DIMENSIONS:
            item = dimensions_by_label.get(label, {})
            score = self._clamp_int(item.get("score"), 0, max_score)
            total += score
            dimensions.append({
                "label": label,
                "score": score,
                "max_score": max_score,
            })
        overall_score = self._clamp_int(payload.get("overall_score"), 0, 100)
        if overall_score == 0 and total > 0:
            overall_score = total
        return {
            "overall_score": overall_score,
            "summary": str(payload.get("summary") or "").strip(),
            "dimensions": dimensions,
            "strengths": self._string_list(payload.get("strengths")),
            "improvements": self._string_list(payload.get("improvements")),
        }

    def _clamp_int(self, value: Any, minimum: int, maximum: int) -> int:
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            number = minimum
        return max(minimum, min(maximum, number))

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item or "").strip() for item in value if str(item or "").strip()]


__all__ = ["EVALUATION_DIMENSIONS", "PlayerClosingSummaryService"]
