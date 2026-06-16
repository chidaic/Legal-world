"""AI-assisted document drafting for the player-lawyer flow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .models import DocumentDraft
from .storage import PlayerLawyerStorage


COMPLAINT_SKILL_ID = "lawyer-complaint-drafting"
DEFENSE_SKILL_ID = "lawyer-defense-drafting"
APPEAL_SKILL_ID = "lawyer-appeal-drafting"
APPEAL_RESPONSE_SKILL_ID = "lawyer-appeal-response-drafting"

DOCUMENT_TYPE_ALIASES = {
    "complaint": "complaint",
    "CD": "complaint",
    "defense": "defense",
    "DD": "defense",
    "appeal": "appeal",
    "AD": "appeal",
    "appeal_response": "appeal_response",
    "AR": "appeal_response",
}

DOCUMENT_SKILL_SPECS = {
    "complaint": {
        "skill_id": COMPLAINT_SKILL_ID,
        "name": "民事起诉状",
        "description": "根据玩家提示、草稿和案件上下文辅助生成《民事起诉状》草稿。",
        "title": "《民事起诉状》",
    },
    "defense": {
        "skill_id": DEFENSE_SKILL_ID,
        "name": "民事答辩状",
        "description": "根据玩家提示、草稿和案件上下文辅助生成《民事答辩状》草稿。",
        "title": "《民事答辩状》",
    },
    "appeal": {
        "skill_id": APPEAL_SKILL_ID,
        "name": "民事上诉状",
        "description": "根据玩家提示、草稿和案件上下文辅助生成《民事上诉状》草稿。",
        "title": "《民事上诉状》",
    },
    "appeal_response": {
        "skill_id": APPEAL_RESPONSE_SKILL_ID,
        "name": "民事上诉答辩状",
        "description": "根据玩家提示、草稿和案件上下文辅助生成《民事上诉答辩状》草稿。",
        "title": "《民事上诉答辩状》",
    },
}


@dataclass(frozen=True)
class DocumentSkill:
    skill_id: str
    document_type: str
    name: str
    description: str
    path: str
    template_title: str = ""
    template_text: str = ""
    quality_check: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "document_type": self.document_type,
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "template_title": self.template_title,
            "template_text": self.template_text,
            "quality_check": list(self.quality_check or []),
        }


def default_skill_root() -> Path:
    return Path(__file__).resolve().parents[2] / "legal-skillhub" / "public" / "legal" / "lawyer" / "document-drafting"


def normalize_phase1_document_type(document_type: str) -> str:
    raw = str(document_type or "").strip()
    normalized = DOCUMENT_TYPE_ALIASES.get(raw) or DOCUMENT_TYPE_ALIASES.get(raw.upper())
    if not normalized:
        raise ValueError("Player document assist only supports complaint/CD, defense/DD, appeal/AD, and appeal_response/AR.")
    return normalized


def _strip_code_fences(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw[3:-3].strip()
        lowered = inner.lower()
        if lowered.startswith("markdown"):
            inner = inner[8:].strip()
        elif lowered.startswith("text"):
            inner = inner[4:].strip()
        elif lowered.startswith("json"):
            inner = inner[4:].strip()
        return inner.strip()
    return raw


def _extract_markdown_heading_section(markdown_text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^#\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^#\s+|\Z)",
        re.MULTILINE,
    )
    match = pattern.search(str(markdown_text or ""))
    return match.group(1).strip() if match else ""


def _extract_first_code_block(markdown_text: str) -> str:
    match = re.search(r"```(?:text|markdown)?\s*\n([\s\S]*?)\n```", str(markdown_text or ""), re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_quality_check(markdown_text: str) -> list[str]:
    section = _extract_markdown_heading_section(markdown_text, "质量检查")
    if not section:
        return []
    forbidden_terms = ("PDF", "工具调用", "文件路径", "后台导出", "起草结束")
    items: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        item = line.lstrip("-").strip()
        if not item or any(term in item for term in forbidden_terms):
            continue
        items.append(item)
    return items


def extract_frontend_document_template(skill_guidance: str) -> tuple[str, list[str]]:
    template_section = _extract_markdown_heading_section(skill_guidance, "文书模板")
    template_text = _extract_first_code_block(template_section)
    return template_text, _extract_quality_check(skill_guidance)


def list_phase1_document_skills(skill_root: str | Path | None = None) -> list[DocumentSkill]:
    root = Path(skill_root) if skill_root is not None else default_skill_root()
    skills: list[DocumentSkill] = []
    for document_type, spec in DOCUMENT_SKILL_SPECS.items():
        skill_path = root / spec["skill_id"] / "SKILL.md"
        if not skill_path.exists():
            continue
        skill_guidance = skill_path.read_text(encoding="utf-8-sig")
        template_text, quality_check = extract_frontend_document_template(skill_guidance)
        skills.append(
            DocumentSkill(
                skill_id=spec["skill_id"],
                document_type=document_type,
                name=spec["name"],
                description=spec["description"],
                path=str(skill_path.resolve()),
                template_title=spec["title"],
                template_text=template_text,
                quality_check=quality_check,
            )
        )
    return skills


def load_skill_guidance(skill_id: str, skill_root: str | Path | None = None) -> str:
    normalized = str(skill_id or "").strip()
    supported_skill_ids = {spec["skill_id"] for spec in DOCUMENT_SKILL_SPECS.values()}
    if normalized not in supported_skill_ids:
        raise ValueError(f"Unsupported document skill: {skill_id}")
    root = Path(skill_root) if skill_root is not None else default_skill_root()
    skill_path = root / normalized / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill not found: {normalized}")
    return skill_path.read_text(encoding="utf-8-sig")


def build_complaint_assist_prompt(
    *,
    document_type: str = "complaint",
    skill_guidance: str,
    case_context: dict[str, Any] | None,
    player_prompt: str,
    player_draft: str = "",
) -> str:
    context = case_context or {}
    context_lines = []
    for key in (
        "case_id",
        "case_background",
        "claims",
        "evidence",
        "court_name",
        "case_cause",
        "consultation_history",
    ):
        value = context.get(key)
        if value:
            context_lines.append(f"{key}: {value}")
    context_text = "\n".join(context_lines) if context_lines else "暂无额外案件上下文。"
    draft_text = str(player_draft or "").strip() or "玩家未提供正文草稿。"

    document_title = DOCUMENT_SKILL_SPECS[normalize_phase1_document_type(document_type)]["title"]

    return (
        "你是法律训练系统中的文书辅助写作模块。请基于玩家律师的提示和案件上下文，"
        f"生成可供玩家继续编辑确认的{document_title}正文。\n\n"
        "规则：\n"
        f"1. 只输出完整{document_title}正文，不要输出解释、总结、Markdown 标题、代码块、PDF 路径或工具调用说明。\n"
        "2. 不要编造缺失的姓名、地址、身份证号、金额、日期、证据编号；缺失信息可写“待补充”。\n"
        "3. Skill 中若出现调用工具或结束标记要求，只把它理解为格式参考，不要在正文中写工具调用或结束标记。\n\n"
        f"【Skill 指南】\n{skill_guidance}\n\n"
        f"【案件上下文】\n{context_text}\n\n"
        f"【玩家提示】\n{str(player_prompt or '').strip()}\n\n"
        f"【玩家草稿】\n{draft_text}\n"
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
            max_tokens=4096,
        ),
    )
    agent = ChatAgent(
        system_message="你只输出可编辑的法律文书正文，不输出解释或工具调用。",
        model=model,
    )
    response = agent.step(BaseMessage.make_user_message(role_name="user", content=prompt))
    return response.msgs[0].content


def _default_renderer(
    *,
    document_type: str,
    document_text: str,
    case_output_dir: Path,
) -> dict[str, str]:
    from ..tools.legal.document_drafting_registry import render_document_drafting_payload_for_output_dir

    return render_document_drafting_payload_for_output_dir(
        document_type=document_type,
        document_text=document_text,
        case_output_dir=case_output_dir,
    )


class PlayerDocumentAssistService:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        skill_root: str | Path | None = None,
        generator: Callable[[str], str] | None = None,
        renderer: Callable[..., dict[str, str]] | None = None,
    ) -> None:
        self.storage = PlayerLawyerStorage(storage_root)
        self.skill_root = Path(skill_root) if skill_root is not None else default_skill_root()
        self.generator = generator or _default_generator
        self.renderer = renderer or _default_renderer

    def list_skills(self) -> list[dict[str, str]]:
        return [skill.to_dict() for skill in list_phase1_document_skills(self.skill_root)]

    def create_draft(
        self,
        *,
        sandbox_id: int = 0,
        request_id: str = "",
        case_id: str,
        document_type: str,
        skill_id: str,
        player_prompt: str,
        player_draft: str = "",
        case_context: dict[str, Any] | None = None,
    ) -> DocumentDraft:
        normalized_document_type = normalize_phase1_document_type(document_type)
        expected_skill_id = DOCUMENT_SKILL_SPECS[normalized_document_type]["skill_id"]
        if str(skill_id or "").strip() != expected_skill_id:
            raise ValueError(f"Skill {skill_id} does not match document type {normalized_document_type}.")
        prompt_text = str(player_prompt or "").strip()
        draft_text = str(player_draft or "").strip()
        if not prompt_text and not draft_text:
            raise ValueError("player_prompt or player_draft is required.")

        skill_guidance = load_skill_guidance(skill_id, self.skill_root)
        prompt = build_complaint_assist_prompt(
            document_type=normalized_document_type,
            skill_guidance=skill_guidance,
            case_context={**(case_context or {}), "case_id": case_id},
            player_prompt=prompt_text,
            player_draft=draft_text,
        )
        document_text = _strip_code_fences(self.generator(prompt)).strip()
        if not document_text:
            raise RuntimeError("Document assist returned empty draft.")

        draft = DocumentDraft(
            request_id=str(request_id or ""),
            sandbox_id=int(sandbox_id or 0),
            case_id=str(case_id or "").strip(),
            document_type=normalized_document_type,
            skill_id=skill_id,
            player_prompt=prompt_text,
            player_draft=draft_text,
            document_text=document_text,
        )
        self.storage.save_draft(draft)
        return draft

    def confirm_draft(self, *, draft_id: str, document_text: str) -> tuple[DocumentDraft, dict[str, str]]:
        draft = self.storage.find_draft(draft_id)
        final_text = str(document_text or "").strip()
        if not final_text:
            raise ValueError("document_text is required.")

        payload = self.renderer(
            document_type=draft.document_type,
            document_text=final_text,
            case_output_dir=self.storage.case_output_dir(draft.case_id),
        )
        draft.document_text = final_text
        draft.confirmed = True
        draft.finish_reason = "player_confirmed"
        draft.pdf_path = str(payload.get("pdf_path", "") or "")
        draft.confirmed_at = datetime.utcnow().isoformat()
        self.storage.save_draft(draft)
        self.storage.save_document_result(
            case_id=draft.case_id,
            document_type=draft.document_type,
            document_text=final_text,
            drafted_document_payload=payload,
        )
        return draft, payload

    def confirm_manual_document(
        self,
        *,
        sandbox_id: int = 0,
        request_id: str = "",
        case_id: str,
        document_type: str,
        document_text: str,
    ) -> tuple[DocumentDraft, dict[str, str]]:
        normalized_document_type = normalize_phase1_document_type(document_type)
        final_text = str(document_text or "").strip()
        if not final_text:
            raise ValueError("document_text is required.")

        payload = self.renderer(
            document_type=normalized_document_type,
            document_text=final_text,
            case_output_dir=self.storage.case_output_dir(case_id),
        )
        draft = DocumentDraft(
            request_id=str(request_id or ""),
            sandbox_id=int(sandbox_id or 0),
            case_id=str(case_id or "").strip(),
            document_type=normalized_document_type,
            skill_id=DOCUMENT_SKILL_SPECS[normalized_document_type]["skill_id"],
            player_prompt="",
            player_draft=final_text,
            document_text=final_text,
            confirmed=True,
            finish_reason="player_confirmed",
            pdf_path=str(payload.get("pdf_path", "") or ""),
            confirmed_at=datetime.utcnow().isoformat(),
        )
        self.storage.save_draft(draft)
        self.storage.save_document_result(
            case_id=draft.case_id,
            document_type=draft.document_type,
            document_text=final_text,
            drafted_document_payload=payload,
        )
        return draft, payload


__all__ = [
    "APPEAL_RESPONSE_SKILL_ID",
    "APPEAL_SKILL_ID",
    "COMPLAINT_SKILL_ID",
    "DEFENSE_SKILL_ID",
    "PlayerDocumentAssistService",
    "build_complaint_assist_prompt",
    "extract_frontend_document_template",
    "list_phase1_document_skills",
    "load_skill_guidance",
    "normalize_phase1_document_type",
]
