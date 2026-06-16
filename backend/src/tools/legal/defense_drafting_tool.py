"""Defense PDF rendering tool."""

from __future__ import annotations

import json
import logging
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

DEFENSE_DRAFT_TOOL_NAME = "draft_defense_document"
DEFENSE_DOCUMENT_TYPE = "defense"
DEFENSE_RESULT_FIELD = "defense_statement"
DEFENSE_PDF_FILENAME = "DD_document.pdf"


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_case_output_dir(agent: Any):
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
        "DefenseTitle",
        fontName="STSong-Light",
        fontSize=16,
        leading=22,
        alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        "DefenseBody",
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
            "name": DEFENSE_DRAFT_TOOL_NAME,
            "description": (
                "接收律师已经写好的《民事答辩状》全文，生成 PDF 文件。"
                "工具本身不负责起草正文，只返回 document_type 和 pdf_path。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "律师已经写好的完整《民事答辩状》正文。",
                    }
                },
                "required": ["document_text"],
                "additionalProperties": False,
            },
        },
    }


class DefenseDraftingTool:
    """Render one defense PDF from lawyer-authored text."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def draft_defense_document(self, document_text: str) -> str:
        normalized_text = _normalize_text(document_text)
        if not normalized_text:
            raise ValueError("document_text is required.")

        pdf_path = ""
        try:
            resolved_pdf_path = _resolve_case_output_dir(self.agent) / DEFENSE_PDF_FILENAME
            _render_pdf(normalized_text, resolved_pdf_path)
            pdf_path = str(resolved_pdf_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to render defense PDF: %s", exc)

        payload = {
            "document_type": DEFENSE_DOCUMENT_TYPE,
            "document_text": normalized_text,
            "pdf_path": pdf_path,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def create_defense_drafting_tool(agent: Any) -> FunctionTool:
    impl = DefenseDraftingTool(agent)
    return FunctionTool(
        impl.draft_defense_document,
        openai_tool_schema=_build_schema(),
    )


__all__ = [
    "DEFENSE_DOCUMENT_TYPE",
    "DEFENSE_DRAFT_TOOL_NAME",
    "DEFENSE_PDF_FILENAME",
    "DEFENSE_RESULT_FIELD",
    "DefenseDraftingTool",
    "REPORTLAB_AVAILABLE",
    "create_defense_drafting_tool",
]
