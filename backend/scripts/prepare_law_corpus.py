"""Prepare raw law sources into structured JSONL documents."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from html import unescape
from pathlib import Path
from typing import Any


HTML_CONTAINER_PATTERNS = (
    re.compile(
        r'<div[^>]+class="pages_content"[^>]*id="UCAP-CONTENT"[^>]*>(?P<content>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'<div[^>]+class="txt_txt"[^>]*id="zoom"[^>]*>(?P<content>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'<div[^>]+id="zoom"[^>]*class="txt_txt"[^>]*>(?P<content>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'<div[^>]+class="txt_txt"[^>]*>(?P<content>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r'<div[^>]+class="news_content_style"[^>]*>(?P<content>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    ),
)

ARTICLE_SPLIT_PATTERN = re.compile(
    r"(?=^第[一二三四五六七八九十百千万零〇两\d]+条)",
    re.MULTILINE,
)
ARTICLE_HEADER_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万零〇两\d]+条)"
    r"(?:\s*[【\[](.*?)[】\]])?"
    r"\s*(.*)$"
)
INLINE_ARTICLE_HEADER_PATTERN = re.compile(
    r"([。！？；:：])\s+(第[一二三四五六七八九十百千万零〇两\d]+条)"
)
PAGE_NUMBER_LINE_PATTERN = re.compile(r"^\d+$")
STRUCTURE_HEADING_PATTERN = re.compile(
    r"^第[一二三四五六七八九十百千万零〇两\d]+(?:编|章|节)\s+.+$"
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
SUPPORTED_RAW_SUFFIXES = (".txt", ".text", ".html", ".docx")


def _normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\ufeff", "")
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    normalized = "\n".join(line for line in lines if line)
    return MULTI_NEWLINE_PATTERN.sub("\n\n", normalized).strip()


def _build_markdown_text(source_entry: dict[str, Any], text: str) -> str:
    normalized = _normalize_text(text)
    return "\n".join(
        [
            f"# {source_entry['title']}",
            "",
            "## 元数据",
            "",
            f"- 文书标识：`{source_entry['document_id']}`",
            f"- 分类：{source_entry['category']}",
            f"- 生效日期：{source_entry['effective_date']}",
            f"- 来源：{source_entry['source_url']}",
            "",
            "## 正文",
            "",
            normalized,
            "",
        ]
    )


def extract_html_main_text(html: str) -> str:
    for pattern in HTML_CONTAINER_PATTERNS:
        match = pattern.search(html)
        if match:
            content = match.group("content")
            break
    else:
        raise ValueError("unable to locate supported HTML content container")

    content = re.sub(r"<script.*?</script>", "", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"<style.*?</style>", "", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</p\s*>", "\n", content, flags=re.IGNORECASE)
    content = re.sub(r"</div\s*>", "\n", content, flags=re.IGNORECASE)
    content = TAG_PATTERN.sub("", content)
    content = unescape(content)

    text = _normalize_text(content)
    text = re.sub(r"\n?责任编辑[:：].*$", "", text, flags=re.MULTILINE)
    return text.strip()


def extract_docx_text(path: str | Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    document_xml = re.sub(r"</w:p>", "\n", document_xml)
    document_xml = re.sub(r"</w:tr>", "\n", document_xml)
    document_xml = re.sub(r"<w:tab[^>]*/>", "\t", document_xml)
    document_xml = TAG_PATTERN.sub("", document_xml)
    return _normalize_text(unescape(document_xml))


def build_documents_from_plaintext(
    source_document_id: str,
    source_title: str,
    category: str,
    effective_date: str,
    source_url: str,
    text: str,
) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    normalized = INLINE_ARTICLE_HEADER_PATTERN.sub(r"\1\n\2", normalized)
    normalized_lines = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if PAGE_NUMBER_LINE_PATTERN.fullmatch(stripped):
            continue
        if STRUCTURE_HEADING_PATTERN.fullmatch(stripped):
            continue
        normalized_lines.append(stripped)
    normalized = "\n".join(normalized_lines)
    parts = [part.strip() for part in ARTICLE_SPLIT_PATTERN.split(normalized) if part.strip()]

    documents: list[dict[str, Any]] = []
    for part in parts:
        part_lines = [line.strip() for line in part.splitlines() if line.strip()]
        if not part_lines:
            continue

        header_match = ARTICLE_HEADER_PATTERN.match(part_lines[0])
        if not header_match:
            continue

        article_ref, bracket_title, first_line_rest = header_match.groups()
        content_lines = []
        if first_line_rest.strip():
            content_lines.append(first_line_rest.strip())
        content_lines.extend(part_lines[1:])
        content = _normalize_text("\n".join(content_lines))
        if not content:
            continue

        title = bracket_title.strip() if bracket_title else article_ref
        documents.append(
            {
                "document_id": f"{source_document_id}:{article_ref}",
                "source_title": source_title,
                "category": category,
                "article_ref": article_ref,
                "title": title,
                "content": content,
                "effective_date": effective_date,
                "source_url": source_url,
            }
        )

    return documents


def load_raw_text(path: str | Path) -> str:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return _normalize_text(source_path.read_text(encoding="utf-8"))
    if suffix == ".html":
        return extract_html_main_text(source_path.read_text(encoding="utf-8"))
    if suffix == ".docx":
        return extract_docx_text(source_path)
    raise ValueError(f"unsupported raw source format: {source_path.name}")


def load_markdown_text(path: str | Path) -> str:
    return _normalize_text(Path(path).read_text(encoding="utf-8"))


def write_markdown_source(
    source_entry: dict[str, Any],
    text: str,
    output_path: str | Path,
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        _build_markdown_text(source_entry, text),
        encoding="utf-8",
    )


def write_processed_documents(documents: list[dict[str, Any]], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def load_source_catalog(path: str | Path) -> list[dict[str, Any]]:
    catalog_path = Path(path)
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def resolve_raw_source_path(raw_dir: str | Path, document_id: str) -> Path:
    raw_path = Path(raw_dir)
    for suffix in SUPPORTED_RAW_SUFFIXES:
        candidate = raw_path / f"{document_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"raw source not found for document_id={document_id}")


def prepare_documents_for_source(
    source_entry: dict[str, Any],
    raw_dir: str | Path,
    markdown_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    source_path = resolve_raw_source_path(raw_dir, str(source_entry["document_id"]))
    text = load_raw_text(source_path)

    if markdown_dir is not None:
        markdown_path = Path(markdown_dir) / f"{source_entry['document_id']}.md"
        write_markdown_source(source_entry, text, markdown_path)
        text = load_markdown_text(markdown_path)

    return build_documents_from_plaintext(
        source_document_id=str(source_entry["document_id"]),
        source_title=str(source_entry["title"]),
        category=str(source_entry["category"]),
        effective_date=str(source_entry["effective_date"]),
        source_url=str(source_entry["source_url"]),
        text=text,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw law sources")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write processed JSONL documents",
    )
    parser.add_argument(
        "--markdown-dir",
        default=None,
        help="Directory to write normalized Markdown law sources before processing",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional source catalog path; defaults to <raw-dir>/source_catalog.json",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Process only the specified document_id values; repeatable",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    markdown_dir = Path(args.markdown_dir) if args.markdown_dir else raw_dir.parent / "markdown"
    catalog_path = Path(args.catalog) if args.catalog else raw_dir / "source_catalog.json"

    catalog = load_source_catalog(catalog_path)
    requested_ids = set(args.source_id or [])
    selected_entries = [
        entry
        for entry in catalog
        if entry.get("status") == "selected"
        and (not requested_ids or entry.get("document_id") in requested_ids)
    ]

    if not selected_entries:
        raise ValueError("no selected source entries matched the current filters")

    summary: list[dict[str, Any]] = []
    for entry in selected_entries:
        markdown_path = markdown_dir / f"{entry['document_id']}.md"
        documents = prepare_documents_for_source(entry, raw_dir, markdown_dir)
        output_path = output_dir / f"{entry['document_id']}.jsonl"
        write_processed_documents(documents, output_path)
        summary.append(
            {
                "document_id": entry["document_id"],
                "article_count": len(documents),
                "markdown_path": str(markdown_path),
                "output_path": str(output_path),
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "build_documents_from_plaintext",
    "extract_docx_text",
    "extract_html_main_text",
    "load_markdown_text",
    "load_raw_text",
    "load_source_catalog",
    "main",
    "prepare_documents_for_source",
    "resolve_raw_source_path",
    "write_markdown_source",
    "write_processed_documents",
]


if __name__ == "__main__":
    raise SystemExit(main())
