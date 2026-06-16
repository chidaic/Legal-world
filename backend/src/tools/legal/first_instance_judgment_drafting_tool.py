"""First-instance judgment PDF rendering tool."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from xml.sax.saxutils import escape

from camel.toolkits import FunctionTool
from .document_drafting_support import resolve_case_output_dir

try:
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    REPORTLAB_AVAILABLE = False


logger = logging.getLogger(__name__)

FIRST_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME = "draft_first_instance_judgment_document"
FIRST_INSTANCE_JUDGMENT_DOCUMENT_TYPE = "first_instance_judgment"
FIRST_INSTANCE_JUDGMENT_RESULT_FIELD = "final_judgment"
FIRST_INSTANCE_JUDGMENT_PDF_FILENAME = "CI_document.pdf"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_case_output_dir(agent: Any) -> Path:
    return resolve_case_output_dir(agent)


def _register_pdf_font() -> None:
    if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def _render_pdf(document_text: str, output_path: Path) -> None:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed.")

    _register_pdf_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    title_style = ParagraphStyle(
        "FirstInstanceJudgmentTitle",
        fontName="STSong-Light",
        fontSize=16,
        leading=22,
        alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        "FirstInstanceJudgmentBody",
        fontName="STSong-Light",
        fontSize=11,
        leading=18,
        alignment=TA_LEFT,
    )

    story = []
    for index, raw_line in enumerate(document_text.replace("\r\n", "\n").split("\n")):
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        style = title_style if index == 0 else body_style
        story.append(Paragraph(escape(line), style))
        story.append(Spacer(1, 2 if index else 10))

    doc.build(story)


def _build_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FIRST_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME,
            "description": (
                "接收审判长已经写好的《民事判决书》全文，生成一审判决书 PDF 文件。"
                "工具本身不负责起草正文，只返回 document_type 和 pdf_path。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "审判长已经写好的完整《民事判决书》正文。",
                    }
                },
                "required": ["document_text"],
                "additionalProperties": False,
            },
        },
    }


class FirstInstanceJudgmentDraftingTool:
    """Render one first-instance judgment PDF from judge-authored text."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def draft_first_instance_judgment_document(self, document_text: str) -> str:
        normalized_text = _normalize_text(document_text)
        if not normalized_text:
            raise ValueError("document_text is required.")

        pdf_path = ""
        try:
            resolved_pdf_path = (
                _resolve_case_output_dir(self.agent) / FIRST_INSTANCE_JUDGMENT_PDF_FILENAME
            )
            _render_pdf(normalized_text, resolved_pdf_path)
            pdf_path = str(resolved_pdf_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to render first-instance judgment PDF: %s", exc)

        payload = {
            "document_type": FIRST_INSTANCE_JUDGMENT_DOCUMENT_TYPE,
            "pdf_path": pdf_path,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def create_first_instance_judgment_drafting_tool(agent: Any) -> FunctionTool:
    impl = FirstInstanceJudgmentDraftingTool(agent)
    return FunctionTool(
        impl.draft_first_instance_judgment_document,
        openai_tool_schema=_build_schema(),
    )


__all__ = [
    "FIRST_INSTANCE_JUDGMENT_DOCUMENT_TYPE",
    "FIRST_INSTANCE_JUDGMENT_DRAFT_TOOL_NAME",
    "FIRST_INSTANCE_JUDGMENT_PDF_FILENAME",
    "FIRST_INSTANCE_JUDGMENT_RESULT_FIELD",
    "FirstInstanceJudgmentDraftingTool",
    "REPORTLAB_AVAILABLE",
    "create_first_instance_judgment_drafting_tool",
]
