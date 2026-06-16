"""AI polish and persistence for player-lawyer text replies."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import ResponseAssist


HINT_LABELS = {
    "liability_scope": "解释责任和赔偿范围",
    "evidence_support": "说明法院会不会支持证据",
    "claim_items": "估算可主张项目",
    "missing_info": "追问缺少的关键信息",
}


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw[3:-3].strip()
        lowered = inner.lower()
        if lowered.startswith("markdown"):
            inner = inner[8:].strip()
        elif lowered.startswith("text"):
            inner = inner[4:].strip()
        return inner.strip()
    return raw


def _normalize_hint_ids(hint_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    for item in list(hint_ids or []):
        value = str(item or "").strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def build_response_polish_prompt(
    *,
    stage: str,
    role: str,
    speaker_label: str,
    prompt: str,
    context_summary: str,
    original_message: str,
    hint_ids: list[str],
) -> str:
    hint_lines = [
        f"- {HINT_LABELS.get(hint_id, hint_id)}"
        for hint_id in hint_ids
        if str(hint_id or "").strip()
    ]
    hints = "\n".join(hint_lines) if hint_lines else "用户未选择提示方向。"
    return (
        "你是法律训练系统中的回复润色助手。请把用户写的当前角色回复润色成自然、专业、口语化的律师表达。\n\n"
        "规则：\n"
        "1. 只润色表达，不新增用户原文和案件上下文都没有的事实、金额、证据或承诺。\n"
        "2. 输出 100-220 字的纯文本回复，不要输出标题、列表、Markdown、解释或工具调用说明。\n"
        "3. 保留用户原意；如果用户表达很短，也只能做合理展开，不能替用户虚构法律结论。\n\n"
        f"【阶段】{stage}\n"
        f"【当前角色】{speaker_label or role}\n"
        f"【提示方向】\n{hints}\n\n"
        f"【上一轮对话/任务】\n{str(prompt or '').strip()}\n\n"
        f"【案件上下文】\n{str(context_summary or '').strip() or '暂无额外上下文。'}\n\n"
        f"【用户原始回复】\n{str(original_message or '').strip()}\n"
    )


def build_response_draft_prompt(
    *,
    stage: str,
    role: str,
    speaker_label: str,
    prompt: str,
    context_summary: str,
    hint_ids: list[str],
) -> str:
    hint_lines = [
        f"- {HINT_LABELS.get(hint_id, hint_id)}"
        for hint_id in hint_ids
        if str(hint_id or "").strip()
    ]
    hints = "\n".join(hint_lines) if hint_lines else "用户未选择提示方向。"
    return (
        "你是法律训练系统中的 AI 代答助手。请直接生成一版可提交的当前角色回复，用于快速跑通流程。\n\n"
        "规则：\n"
        "1. 只根据上一轮任务和案件上下文回答，不新增上下文没有的金额、证据或承诺。\n"
        "2. 输出 100-220 字的纯文本回复，不要输出标题、列表、Markdown、解释或工具调用说明。\n"
        "3. 语气要像当前角色正在自然回复当事人或流程任务，结论清楚，方便直接提交。\n\n"
        f"【阶段】{stage}\n"
        f"【当前角色】{speaker_label or role}\n"
        f"【提示方向】\n{hints}\n\n"
        f"【上一轮对话/任务】\n{str(prompt or '').strip()}\n\n"
        f"【案件上下文】\n{str(context_summary or '').strip() or '暂无额外上下文。'}\n"
    )


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
            max_tokens=800,
        ),
    )
    agent = ChatAgent(
        system_message="你只输出润色后的当前角色回复，不输出解释或工具调用。",
        model=model,
    )
    response = agent.step(BaseMessage.make_user_message(role_name="user", content=prompt))
    return response.msgs[0].content


class PlayerResponseAssistService:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        generator: Callable[[str], str] | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.generator = generator or _default_generator

    def polish_response(
        self,
        *,
        request_id: str,
        sandbox_id: int = 0,
        case_id: str,
        stage: str,
        role: str,
        speaker_label: str = "",
        prompt: str = "",
        context_summary: str = "",
        original_message: str,
        hint_ids: list[str] | None = None,
    ) -> ResponseAssist:
        original = str(original_message or "").strip()
        if not original:
            raise ValueError("original_message is required.")
        normalized_hints = _normalize_hint_ids(hint_ids)
        polish_prompt = build_response_polish_prompt(
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
            original_message=original,
            hint_ids=normalized_hints,
        )
        polished = _strip_code_fences(self.generator(polish_prompt)).strip()
        if not polished:
            raise RuntimeError("Response polish returned empty text.")
        assist = self._load_or_create(
            request_id=request_id,
            sandbox_id=sandbox_id,
            case_id=case_id,
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
        )
        assist.hint_ids = normalized_hints
        assist.user_original_message = original
        assist.ai_polished_message = polished
        assist.used_ai_polish = False
        assist.updated_at = datetime.utcnow().isoformat()
        self._save(assist)
        return assist

    def draft_response(
        self,
        *,
        request_id: str,
        sandbox_id: int = 0,
        case_id: str,
        stage: str,
        role: str,
        speaker_label: str = "",
        prompt: str = "",
        context_summary: str = "",
        hint_ids: list[str] | None = None,
    ) -> ResponseAssist:
        normalized_hints = _normalize_hint_ids(hint_ids)
        draft_prompt = build_response_draft_prompt(
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
            hint_ids=normalized_hints,
        )
        drafted = _strip_code_fences(self.generator(draft_prompt)).strip()
        if not drafted:
            raise RuntimeError("Response draft returned empty text.")
        assist = self._load_or_create(
            request_id=request_id,
            sandbox_id=sandbox_id,
            case_id=case_id,
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
        )
        assist.hint_ids = normalized_hints
        assist.user_original_message = ""
        assist.ai_polished_message = drafted
        assist.used_ai_polish = False
        assist.updated_at = datetime.utcnow().isoformat()
        self._save(assist)
        return assist

    def record_submission(
        self,
        *,
        request_id: str,
        sandbox_id: int = 0,
        case_id: str,
        stage: str,
        role: str,
        speaker_label: str = "",
        prompt: str = "",
        context_summary: str = "",
        original_message: str = "",
        polished_message: str = "",
        final_message: str,
        hint_ids: list[str] | None = None,
        used_ai_polish: bool = False,
    ) -> ResponseAssist:
        final_text = str(final_message or "").strip()
        if not final_text:
            raise ValueError("final_message is required.")
        assist = self._load_or_create(
            request_id=request_id,
            sandbox_id=sandbox_id,
            case_id=case_id,
            stage=stage,
            role=role,
            speaker_label=speaker_label,
            prompt=prompt,
            context_summary=context_summary,
        )
        normalized_hints = _normalize_hint_ids(hint_ids)
        if normalized_hints:
            assist.hint_ids = normalized_hints
        if str(original_message or "").strip():
            assist.user_original_message = str(original_message or "").strip()
        elif not assist.user_original_message:
            assist.user_original_message = final_text
        if str(polished_message or "").strip():
            assist.ai_polished_message = str(polished_message or "").strip()
        assist.final_submitted_message = final_text
        assist.used_ai_polish = bool(used_ai_polish)
        assist.updated_at = datetime.utcnow().isoformat()
        self._save(assist)
        return assist

    def _assist_path(self, case_id: str, request_id: str) -> Path:
        return (
            self.storage_root
            / "output"
            / str(case_id or "").strip()
            / "_player_lawyer"
            / "response_assists"
            / f"{str(request_id or '').strip()}.json"
        )

    def _load_or_create(
        self,
        *,
        request_id: str,
        sandbox_id: int,
        case_id: str,
        stage: str,
        role: str,
        speaker_label: str,
        prompt: str,
        context_summary: str,
    ) -> ResponseAssist:
        normalized_case_id = str(case_id or "").strip()
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("request_id is required.")
        if not normalized_case_id:
            raise ValueError("case_id is required.")
        path = self._assist_path(normalized_case_id, normalized_request_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return response_assist_from_dict(payload)
        return ResponseAssist(
            request_id=normalized_request_id,
            sandbox_id=int(sandbox_id or 0),
            case_id=normalized_case_id,
            stage=str(stage or "").strip(),
            role=str(role or "").strip(),
            speaker_label=str(speaker_label or "").strip(),
            prompt=str(prompt or ""),
            context_summary=str(context_summary or ""),
        )

    def _save(self, assist: ResponseAssist) -> Path:
        path = self._assist_path(assist.case_id, assist.request_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(assist.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def response_assist_from_dict(payload: dict[str, Any]) -> ResponseAssist:
    return ResponseAssist(
        request_id=str(payload.get("request_id", "") or ""),
        sandbox_id=int(payload.get("sandbox_id", 0) or 0),
        case_id=str(payload.get("case_id", "") or ""),
        stage=str(payload.get("stage", "") or ""),
        role=str(payload.get("role", "") or ""),
        speaker_label=str(payload.get("speaker_label", "") or ""),
        prompt=str(payload.get("prompt", "") or ""),
        context_summary=str(payload.get("context_summary", "") or ""),
        hint_ids=_normalize_hint_ids(payload.get("hint_ids") or []),
        user_original_message=str(payload.get("user_original_message", "") or ""),
        ai_polished_message=str(payload.get("ai_polished_message", "") or ""),
        final_submitted_message=str(payload.get("final_submitted_message", "") or ""),
        used_ai_polish=bool(payload.get("used_ai_polish")),
        created_at=str(payload.get("created_at", "") or datetime.utcnow().isoformat()),
        updated_at=payload.get("updated_at"),
    )


__all__ = [
    "HINT_LABELS",
    "PlayerResponseAssistService",
    "build_response_draft_prompt",
    "build_response_polish_prompt",
    "response_assist_from_dict",
]
