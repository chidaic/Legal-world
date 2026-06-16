"""Prompt assembly helpers for pipeline and sandbox agents."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from .legal_persona import build_legal_persona_prompt
from ..utils.live_card_memory import build_memory_prompt_block

logger = logging.getLogger(__name__)


class PromptAssembler:
    """Assemble profile and scenario prompt into one system prompt."""

    @staticmethod
    def _normalize_party_type(profile: Dict[str, Any]) -> str:
        raw_value = str(
            profile.get("party_type", "")
            or profile.get("type", "")
            or ""
        ).strip().lower()
        if raw_value in {"法人", "企业", "公司", "corporate", "company", "legal_person"}:
            return "corporate"
        if raw_value in {"自然人", "个人", "personal", "natural_person", "person"}:
            return "personal"
        return ""

    @staticmethod
    def build(
        profile: Dict[str, Any],
        long_term_memory: Optional[Any] = None,
        memory_owner: Optional[str] = None,
        scenario_prompt: str = "",
        **_ignored: Any,
    ) -> str:
        parts: list[str] = []

        profile_section = PromptAssembler._build_profile_section(profile)
        if profile_section:
            parts.append(profile_section)

        memory_section = ""
        if long_term_memory is not None:
            try:
                memory_section = build_memory_prompt_block(str(memory_owner or "").strip(), long_term_memory)
            except Exception:
                memory_section = ""
        if memory_section:
            parts.append(memory_section)

        if scenario_prompt:
            parts.append(scenario_prompt)

        return "\n\n".join(parts)

    @staticmethod
    def _build_profile_section(profile: Dict[str, Any]) -> str:
        if not profile:
            return ""

        name = str(profile.get("name", "未知") or "未知").strip()
        party_type = PromptAssembler._normalize_party_type(profile)
        representative = str(profile.get("representative", "") or "").strip()
        address = str(profile.get("address", "") or "").strip()

        if party_type == "corporate":
            if representative:
                lines = [f"你是{name}的代表人{representative}。"]
            else:
                lines = [f"你代表{name}参与当前案件。"]
            if address:
                lines.append(f"地址：{address}")
        else:
            lines = [f"你是{name}。"]

        if party_type == "corporate":
            field_mappings = [
                (("phone", "phone_number", "contact_phone"), "联系电话"),
                (("personality",), "性格特点"),
                (("speaking_style",), "说话风格"),
                (("background",), "背景"),
                (("seniority",), "资历"),
                (("specialty", "specialty_areas"), "擅长领域"),
                (("court", "court_name"), "所属法院"),
                (("court_level",), "法院层级"),
                (("years_of_experience",), "审判年限"),
                (("law_firm",), "所属律所"),
            ]
        else:
            field_mappings = [
                (("gender",), "性别"),
                (("birth_date",), "出生日期"),
                (("ethnicity",), "民族"),
                (("address",), "住址"),
                (("phone", "phone_number", "contact_phone"), "联系电话"),
                (("id_number", "id_card_number"), "身份证号"),
                (("age",), "年龄"),
                (("occupation",), "职业"),
                (("personality",), "性格特点"),
                (("speaking_style",), "说话风格"),
                (("background",), "背景"),
                (("seniority",), "资历"),
                (("specialty", "specialty_areas"), "擅长领域"),
                (("court", "court_name"), "所属法院"),
                (("court_level",), "法院层级"),
                (("years_of_experience",), "审判年限"),
                (("law_firm",), "所属律所"),
            ]

        for keys, label in field_mappings:
            value = next((profile.get(key) for key in keys if profile.get(key)), None)
            if not value:
                continue
            if isinstance(value, list):
                value = "、".join(str(item) for item in value)
            lines.append(f"{label}：{value}")

        guidelines = str(profile.get("interaction_guidelines", "") or "").strip()
        if guidelines:
            lines.append("")
            lines.append(guidelines)

        legal_persona_prompt = build_legal_persona_prompt(profile.get("legal_persona_profile"))
        if legal_persona_prompt:
            lines.append("")
            lines.append("【法律人格画像】")
            lines.append(legal_persona_prompt)

        return "\n".join(lines)

    @staticmethod
    def render_template(template: str, data: Dict[str, Any]) -> str:
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            value = data.get(key, "")
            if isinstance(value, list):
                return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(value))
            return str(value) if value else ""

        return re.sub(r"\$\{(\w+)\}", _replace, template)

    @staticmethod
    def build_scenario_prompt(
        agent_type: str,
        scenario_type: str,
        scenario_data: Dict[str, Any],
        court_role: Optional[str] = None,
        template_key: Optional[str] = None,
    ) -> str:
        from .scenario_templates import SCENARIO_TEMPLATES

        templates = SCENARIO_TEMPLATES.get(agent_type, {})

        resolved_template_key = str(template_key or "").strip()
        if not resolved_template_key:
            resolved_template_key = scenario_type
            if court_role and scenario_type in ("CI", "CIA"):
                resolved_template_key = f"{scenario_type}-{court_role}"

        template = templates.get(resolved_template_key, "")
        if not template:
            logger.warning("No template: agent_type=%s, key=%s", agent_type, resolved_template_key)
            return ""

        return PromptAssembler.render_template(template, scenario_data)
