"""Optional legal citation validation tool."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from camel.toolkits import FunctionTool


CITATION_CHECK_TOOL_NAME = "check_citations"
DEFAULT_LEGAL_CORPUS_DIR = Path(__file__).resolve().parents[3] / "legal_corpus" / "processed"
ARTICLE_REF_RE = (
    r"第[一二三四五六七八九十百千万零〇\d]+条"
    r"(?:第[一二三四五六七八九十百千万零〇\d]+款)?"
    r"(?:第[一二三四五六七八九十百千万零〇\d]+项)?"
)
EXPLICIT_CITATION_RE = re.compile(
    rf"《(?P<title>[^》\n]{{2,80}})》\s*(?P<article>{ARTICLE_REF_RE})"
)


def _normalize_text(value: Any) -> str:
    return (
        str(value or "")
        .replace("（", "(")
        .replace("）", ")")
        .replace("《", "")
        .replace("》", "")
        .replace(" ", "")
        .replace("\u3000", "")
        .strip()
    )


def _normalize_title(value: Any) -> str:
    return _normalize_text(value)


def _normalize_article_ref(value: Any) -> str:
    return _normalize_text(value)


def _build_title_aliases(title: str) -> set[str]:
    raw = str(title or "").strip()
    normalized = _normalize_title(raw)
    aliases = {raw, normalized}

    if normalized.startswith("中华人民共和国") and len(normalized) > len("中华人民共和国"):
        aliases.add(normalized[len("中华人民共和国") :])

    if normalized.startswith("最高人民法院关于适用"):
        aliases.add(normalized[len("最高人民法院关于适用") :])
    if normalized.startswith("最高人民法院关于审理"):
        aliases.add(normalized[len("最高人民法院关于审理") :])

    quote_match = re.search(r"《(?P<quoted>[^》]+)》(?P<suffix>.*)", raw)
    if quote_match:
        quoted = _normalize_title(quote_match.group("quoted"))
        suffix = _normalize_title(quote_match.group("suffix"))
        if quoted:
            aliases.add(quoted)
            if quoted.startswith("中华人民共和国") and len(quoted) > len("中华人民共和国"):
                aliases.add(quoted[len("中华人民共和国") :])
        if quoted and suffix:
            aliases.add(f"{quoted}{suffix}")
            if quoted.startswith("中华人民共和国") and len(quoted) > len("中华人民共和国"):
                aliases.add(f"{quoted[len('中华人民共和国'):]}{suffix}")
            shortened_suffix = suffix.replace("的解释", "解释").replace("若干问题的规定", "规定")
            aliases.add(f"{quoted}{shortened_suffix}")
            if quoted.startswith("中华人民共和国") and len(quoted) > len("中华人民共和国"):
                aliases.add(f"{quoted[len('中华人民共和国'):]}{shortened_suffix}")
            if "编" in suffix:
                aliases.add(suffix)

    compact = normalized
    compact = compact.replace("若干问题的解释", "解释")
    compact = compact.replace("若干问题的规定", "规定")
    compact = compact.replace("适用法律", "")
    compact = compact.replace("关于适用", "")
    compact = compact.replace("关于审理", "")
    aliases.add(compact)

    filtered: set[str] = set()
    for alias in aliases:
        normalized_alias = _normalize_title(alias)
        if len(normalized_alias) >= 3:
            filtered.add(normalized_alias)
    return filtered


@lru_cache(maxsize=1)
def _load_citation_index() -> dict[str, Any]:
    alias_to_title: dict[str, str] = {}
    title_index: dict[str, dict[str, Any]] = {}

    for path in sorted(DEFAULT_LEGAL_CORPUS_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                record = json.loads(text)
                source_title = str(record.get("source_title") or "").strip()
                article_ref = _normalize_article_ref(record.get("article_ref"))
                if not source_title or not article_ref:
                    continue

                title_bucket = title_index.setdefault(
                    source_title,
                    {
                        "source_title": source_title,
                        "normalized_title": _normalize_title(source_title),
                        "articles": {},
                    },
                )
                title_bucket["articles"][article_ref] = {
                    "article_ref": article_ref,
                    "content": str(record.get("content") or "").strip(),
                    "source_url": str(record.get("source_url") or "").strip(),
                    "document_id": str(record.get("document_id") or "").strip(),
                }

    for source_title in title_index:
        for alias in _build_title_aliases(source_title):
            alias_to_title.setdefault(alias, source_title)

    return {
        "alias_to_title": alias_to_title,
        "title_index": title_index,
    }


def _resolve_title(raw_title: str) -> tuple[str | None, str]:
    index = _load_citation_index()
    normalized = _normalize_title(raw_title)
    if not normalized:
        return None, "missing"

    alias_to_title = index["alias_to_title"]
    if normalized in alias_to_title:
        return alias_to_title[normalized], "exact"

    candidates = [
        title
        for alias, title in alias_to_title.items()
        if normalized in alias or alias in normalized
    ]
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0], "fuzzy"
    return None, "unresolved"


def _extract_explicit_citations(document_text: str) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for match in EXPLICIT_CITATION_RE.finditer(document_text):
        citations.append(
            {
                "citation_text": match.group(0).strip(),
                "raw_title": match.group("title").strip(),
                "article_ref": match.group("article").strip(),
                "extraction_mode": "explicit",
            }
        )
    return citations


def _extract_alias_citations(document_text: str) -> list[dict[str, str]]:
    normalized_text = _normalize_text(document_text)
    if not normalized_text:
        return []

    alias_to_title = _load_citation_index()["alias_to_title"]
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias in sorted(alias_to_title.keys(), key=len, reverse=True):
        if len(alias) < 4 or alias.startswith("最高人民法院"):
            continue
        pattern = re.compile(rf"(?P<title>{re.escape(alias)})(?P<article>{ARTICLE_REF_RE})")
        for match in pattern.finditer(normalized_text):
            key = (match.group("title"), match.group("article"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "citation_text": f"{match.group('title')}{match.group('article')}",
                    "raw_title": match.group("title"),
                    "article_ref": match.group("article"),
                    "extraction_mode": "alias",
                }
            )
    return citations


def _deduplicate_citations(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (
            _normalize_title(item.get("raw_title")),
            _normalize_article_ref(item.get("article_ref")),
            str(item.get("extraction_mode") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


class CitationCheckTool:
    """Validate cited statutes against the local law corpus."""

    def check_citations(self, document_text: str) -> str:
        content = str(document_text or "").strip()
        if not content:
            raise ValueError("document_text is required.")

        title_index = _load_citation_index()["title_index"]
        extracted = _deduplicate_citations(
            _extract_explicit_citations(content) + _extract_alias_citations(content)
        )

        results: list[dict[str, Any]] = []
        valid_count = 0
        for item in extracted:
            resolved_title, match_mode = _resolve_title(item["raw_title"])
            if not resolved_title:
                results.append(
                    {
                        **item,
                        "status": "invalid_title",
                        "resolved_title": "",
                        "title_match_mode": match_mode,
                        "article_exists": False,
                    }
                )
                continue

            article_ref = _normalize_article_ref(item["article_ref"])
            article_payload = title_index[resolved_title]["articles"].get(article_ref)
            if article_payload is None:
                results.append(
                    {
                        **item,
                        "status": "invalid_article",
                        "resolved_title": resolved_title,
                        "title_match_mode": match_mode,
                        "article_exists": False,
                    }
                )
                continue

            valid_count += 1
            results.append(
                {
                    **item,
                    "status": "valid",
                    "resolved_title": resolved_title,
                    "title_match_mode": match_mode,
                    "article_exists": True,
                    "source_url": article_payload.get("source_url", ""),
                    "content_preview": str(article_payload.get("content") or "")[:120],
                }
            )

        payload = {
            "tool_name": CITATION_CHECK_TOOL_NAME,
            "status": "ok",
            "total_citations": len(results),
            "valid_count": valid_count,
            "invalid_count": len(results) - valid_count,
            "citations": results,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _build_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": CITATION_CHECK_TOOL_NAME,
            "description": (
                "校验文书中引用的法条是否存在于本地法律语料，并检查法条号是否写错。"
                "默认不启用，适合在文书完成后做专业校验。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "需要校验引用法条的完整文书正文。",
                    }
                },
                "required": ["document_text"],
                "additionalProperties": False,
            },
        },
    }


def create_citation_check_tool(agent: Any | None = None) -> FunctionTool:
    del agent
    impl = CitationCheckTool()
    return FunctionTool(
        impl.check_citations,
        openai_tool_schema=_build_schema(),
    )


__all__ = [
    "CITATION_CHECK_TOOL_NAME",
    "CitationCheckTool",
    "create_citation_check_tool",
]
