"""File-based similar-case retrieval tool backed by local JSONL documents."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable

from camel.toolkits import FunctionTool


CASE_RETRIEVAL_TOOL_NAME = "search_cases"
DEFAULT_TOP_K = 5
DEFAULT_CASE_DOCS_ENV = "SIMLAW_CASE_RETRIEVAL_DOCS_PATH"
DEFAULT_CASE_DOCS_PATH = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "case_retrieval"
    / "cases_retrieval_docs.jsonl"
)

FIELD_WEIGHTS: dict[str, float] = {
    "title_clean": 3.0,
    "case_cause": 4.0,
    "legal_basis": 2.0,
    "first_instance_text": 1.6,
    "second_instance_text": 1.6,
}

K1 = 1.5
B = 0.75

_CASE_TOOL_CACHE_LOCK = Lock()
_CASE_TOOL_CACHE: dict[tuple[str], "LocalCaseRetrievalEngine"] = {}

TITLE_SUFFIX_RE = re.compile(
    r"(?:一审|二审|再审|民事一审|民事二审|民事再审)?(?:民事)?(?:判决书|裁定书|调解书|决定书)$"
)
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_SPLIT_RE = re.compile(r"[，。；：！？、（）【】《》“”\"'\[\]\{\}\n\r\t ]+")
ARTICLE_RE = re.compile(
    r"第[一二三四五六七八九十百千万零〇\d]+条"
    r"(?:第[一二三四五六七八九十百千万零〇\d]+款)?"
    r"(?:第[一二三四五六七八九十百千万零〇\d]+项)?"
)
CAUSE_RE = re.compile(r"[\u4e00-\u9fff]{2,20}纠纷")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")

GENERIC_TOKENS = {
    "人民法院",
    "民事判决书",
    "民事裁定书",
    "民事调解书",
    "原告",
    "被告",
    "上诉人",
    "被上诉人",
    "本院认为",
    "判决如下",
    "经审理查明",
    "本院查明",
    "审理终结",
    "案件受理费",
}

SECTION_MARKERS_FIRST = (
    "诉讼请求",
    "事实和理由",
    "事实与理由",
    "辩称",
    "答辩",
    "经审理查明",
    "本院查明",
    "本院认为",
    "判决如下",
)

SECTION_MARKERS_SECOND = (
    "上诉请求",
    "上诉理由",
    "答辩",
    "本院查明",
    "本院认为",
    "判决如下",
)


def resolve_case_docs_path(storage_path: str | None = None) -> Path:
    """Resolve the case retrieval corpus path."""
    configured = storage_path or os.getenv(DEFAULT_CASE_DOCS_ENV) or str(DEFAULT_CASE_DOCS_PATH)
    return Path(configured).expanduser().resolve()


def normalize_text(text: Any) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def clean_title(title: Any) -> str:
    cleaned = normalize_text(title)
    cleaned = TITLE_SUFFIX_RE.sub("", cleaned).strip(" -")
    return cleaned


def unique_non_empty(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def _clip(text: Any, limit: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _extract_sections(text: Any, markers: tuple[str, ...]) -> list[str]:
    content = normalize_text(text)
    if not content:
        return []

    positions: list[tuple[int, str]] = []
    for marker in markers:
        index = content.find(marker)
        if index >= 0:
            positions.append((index, marker))

    if not positions:
        return []

    positions.sort()
    sections: list[str] = []
    for idx, (start, _) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(content)
        section = content[start:end].strip()
        if section:
            sections.append(section)
    return sections


def compress_judgment_text(text: Any, stage: str) -> str:
    markers = SECTION_MARKERS_SECOND if stage == "second" else SECTION_MARKERS_FIRST
    sections = _extract_sections(text, markers)
    if sections:
        combined = "\n".join(unique_non_empty(_clip(section, 1200) for section in sections))
        if combined:
            return combined
    return _clip(text, 4000)


def join_retrieval_fields(doc: dict[str, Any]) -> str:
    return "\n".join(
        [
            normalize_text(doc.get("title_clean", "")),
            normalize_text(doc.get("case_cause", "")),
            normalize_text(doc.get("legal_basis", "")),
            normalize_text(doc.get("first_instance_text", "")),
            normalize_text(doc.get("second_instance_text", "")),
        ]
    ).strip()


def build_retrieval_doc(raw_case: dict[str, Any]) -> dict[str, Any]:
    first_instance = raw_case.get("first_instance", {}) or {}
    second_instance = raw_case.get("second_instance", {}) or {}

    first_title = clean_title(first_instance.get("标题", ""))
    second_title = clean_title(second_instance.get("标题", ""))
    title_clean = " | ".join(unique_non_empty([first_title, second_title]))

    case_cause = normalize_text(
        first_instance.get("案由") or second_instance.get("案由") or ""
    )
    legal_basis = "；".join(
        unique_non_empty(
            [
                first_instance.get("法律依据", ""),
                second_instance.get("法律依据", ""),
            ]
        )
    )

    return {
        "case_id": int(raw_case.get("id") or 0),
        "title_clean": title_clean,
        "case_cause": case_cause,
        "legal_basis": legal_basis,
        "first_instance_text": compress_judgment_text(
            first_instance.get("文书内容", ""),
            stage="first",
        ),
        "second_instance_text": compress_judgment_text(
            second_instance.get("文书内容", ""),
            stage="second",
        ),
    }


def tokenize(text: Any) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens: list[str] = []
    tokens.extend(ARTICLE_RE.findall(normalized))
    tokens.extend(CAUSE_RE.findall(normalized))

    for token in ASCII_TOKEN_RE.findall(normalized):
        tokens.append(token.lower())

    for match in CHINESE_RE.finditer(normalized):
        chunk = match.group(0)
        if len(chunk) <= 1:
            continue

        if 2 <= len(chunk) <= 8 and chunk not in GENERIC_TOKENS:
            tokens.append(chunk)

        if len(chunk) >= 3:
            for start in range(0, len(chunk) - 2, 2):
                piece = chunk[start : start + 3]
                if piece not in GENERIC_TOKENS:
                    tokens.append(piece)

        if len(chunk) >= 4:
            for start in range(0, len(chunk) - 3, 3):
                piece = chunk[start : start + 4]
                if piece not in GENERIC_TOKENS:
                    tokens.append(piece)

    return [token.strip() for token in tokens if token.strip() and token.strip() not in GENERIC_TOKENS]


def _bm25_score(
    tf: int,
    doc_len: int,
    avg_doc_len: float,
    doc_freq: int,
    doc_count: int,
) -> float:
    if tf <= 0 or doc_len <= 0 or avg_doc_len <= 0.0 or doc_freq <= 0 or doc_count <= 0:
        return 0.0
    idf = math.log(1.0 + ((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))
    denom = tf + K1 * (1.0 - B + B * (doc_len / avg_doc_len))
    if denom <= 0.0:
        return 0.0
    return idf * (tf * (K1 + 1.0) / denom)


def _best_snippet(text: Any, query_tokens: list[str], max_chars: int = 180) -> str:
    content = normalize_text(text)
    if not content:
        return ""

    segments = [segment.strip() for segment in PUNCT_SPLIT_RE.split(content) if segment.strip()]
    if not segments:
        return content[:max_chars]

    best_segment = segments[0]
    best_score = -1
    for segment in segments:
        score = sum(1 for token in query_tokens if token and token in segment)
        if score > best_score:
            best_score = score
            best_segment = segment
    return best_segment[:max_chars]


class LocalCaseRetrievalEngine:
    """Simple in-memory BM25-style similar-case retriever."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.doc_count = len(docs)
        self._field_doc_lengths: dict[str, list[int]] = {}
        self._field_doc_freqs: dict[str, Counter[str]] = {}
        self._field_avg_lengths: dict[str, float] = {}
        self._field_postings: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self._retrieval_text_lower = [join_retrieval_fields(doc).lower() for doc in docs]

        for field in FIELD_WEIGHTS:
            doc_lengths: list[int] = []
            doc_freqs: Counter[str] = Counter()
            postings: dict[str, list[tuple[int, int]]] = {}

            for doc_index, doc in enumerate(docs):
                counter = Counter(tokenize(doc.get(field, "")))
                doc_lengths.append(sum(counter.values()))
                for token, tf in counter.items():
                    doc_freqs[token] += 1
                    postings.setdefault(token, []).append((doc_index, tf))

            self._field_doc_lengths[field] = doc_lengths
            self._field_doc_freqs[field] = doc_freqs
            self._field_avg_lengths[field] = (
                sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
            )
            self._field_postings[field] = postings

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LocalCaseRetrievalEngine":
        docs: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    docs.append(json.loads(text))
        return cls(docs)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        include_full_texts: bool = False,
    ) -> list[dict[str, Any]]:
        query_text = normalize_text(query)
        if not query_text:
            raise ValueError("query must not be empty.")

        query_tokens = tokenize(query_text)
        if not query_tokens:
            raise ValueError("query contains no searchable tokens.")

        scores: dict[int, float] = {}
        candidate_doc_ids: set[int] = set()
        for field, weight in FIELD_WEIGHTS.items():
            doc_lengths = self._field_doc_lengths[field]
            doc_freqs = self._field_doc_freqs[field]
            avg_len = self._field_avg_lengths[field]
            postings = self._field_postings[field]

            for token in query_tokens:
                token_postings = postings.get(token, [])
                if not token_postings:
                    continue
                token_df = doc_freqs.get(token, 0)
                for doc_index, tf in token_postings:
                    candidate_doc_ids.add(doc_index)
                    scores[doc_index] = scores.get(doc_index, 0.0) + weight * _bm25_score(
                        tf=tf,
                        doc_len=doc_lengths[doc_index],
                        avg_doc_len=avg_len,
                        doc_freq=token_df,
                        doc_count=self.doc_count,
                    )

        lower_query = query_text.lower()
        if not candidate_doc_ids:
            candidate_doc_ids = set(range(self.doc_count))

        for doc_index in candidate_doc_ids:
            doc = self.docs[doc_index]
            title_clean = str(doc.get("title_clean", "") or "")
            case_cause = str(doc.get("case_cause", "") or "")
            legal_basis = str(doc.get("legal_basis", "") or "")

            if title_clean and title_clean in query_text:
                scores[doc_index] = scores.get(doc_index, 0.0) + 6.0
            if case_cause and case_cause in query_text:
                scores[doc_index] = scores.get(doc_index, 0.0) + 8.0
            article_hits = sum(1 for article in ARTICLE_RE.findall(legal_basis) if article in query_text)
            if article_hits:
                scores[doc_index] = scores.get(doc_index, 0.0) + 1.5 * article_hits
            if lower_query and lower_query in self._retrieval_text_lower[doc_index]:
                scores[doc_index] = scores.get(doc_index, 0.0) + 2.0

        ranked_indices = sorted(
            candidate_doc_ids,
            key=lambda idx: scores.get(idx, 0.0),
            reverse=True,
        )[: max(1, int(top_k))]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            source_doc = dict(self.docs[idx])
            payload = {
                "case_id": source_doc["case_id"],
                "title_clean": source_doc["title_clean"],
                "case_cause": source_doc["case_cause"],
                "legal_basis": source_doc["legal_basis"],
                "score": round(float(scores.get(idx, 0.0)), 6),
                "rank": rank,
                "matched_snippet_first": _best_snippet(
                    source_doc.get("first_instance_text", ""),
                    query_tokens,
                ),
                "matched_snippet_second": _best_snippet(
                    source_doc.get("second_instance_text", ""),
                    query_tokens,
                ),
            }
            if include_full_texts:
                payload["first_instance_text"] = source_doc.get("first_instance_text", "")
                payload["second_instance_text"] = source_doc.get("second_instance_text", "")
            results.append(payload)
        return results


def _get_shared_case_tool(storage_path: str | Path) -> LocalCaseRetrievalEngine:
    resolved_path = str(Path(storage_path).resolve())
    cache_key = (resolved_path,)
    with _CASE_TOOL_CACHE_LOCK:
        tool = _CASE_TOOL_CACHE.get(cache_key)
        if tool is None:
            tool = LocalCaseRetrievalEngine.from_jsonl(resolved_path)
            _CASE_TOOL_CACHE[cache_key] = tool
        return tool


def create_case_search_function(
    storage_path: str | None = None,
    top_k: int = DEFAULT_TOP_K,
):
    """Create the raw callable used by the CAMEL FunctionTool wrapper."""

    resolved_path = resolve_case_docs_path(storage_path)

    def search_cases(
        query: str,
        top_k: int = top_k,
        include_full_texts: bool = False,
    ) -> str:
        results = _get_shared_case_tool(resolved_path).search(
            query=query,
            top_k=top_k,
            include_full_texts=include_full_texts,
        )
        return json.dumps(results, ensure_ascii=False, separators=(",", ":"))

    return search_cases


def _build_search_cases_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": CASE_RETRIEVAL_TOOL_NAME,
            "description": (
                "检索与当前案情最相关的类案。优先使用简短、聚焦的 query，"
                "例如“买卖合同纠纷 口头协议 居间人 职务行为”。"
                "默认返回案由、法律依据和命中片段，不返回全文。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "简短、聚焦的类案检索关键词。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关案件数量；不传时默认 5。",
                    },
                    "include_full_texts": {
                        "type": "boolean",
                        "description": "是否附带一审、二审核心文本；默认 false。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def create_case_retrieval_tool(
    agent: Any | None = None,
    storage_path: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> FunctionTool:
    """Create the CAMEL FunctionTool wrapper for local similar-case retrieval."""
    del agent
    return FunctionTool(
        create_case_search_function(storage_path=storage_path, top_k=top_k),
        openai_tool_schema=_build_search_cases_schema(),
    )


def benchmark_queries(
    engine: LocalCaseRetrievalEngine,
    queries: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    top1_hits = 0
    top5_hits = 0
    samples: list[dict[str, Any]] = []

    for index, item in enumerate(queries):
        started = time.perf_counter()
        results = engine.search(str(item["query"]), top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        target_case_id = int(item["target_case_id"])
        result_case_ids = [int(result["case_id"]) for result in results]
        if result_case_ids and result_case_ids[0] == target_case_id:
            top1_hits += 1
        if target_case_id in result_case_ids[:5]:
            top5_hits += 1

        if index < 5:
            samples.append(
                {
                    "query": item["query"],
                    "target_case_id": target_case_id,
                    "results": [
                        {
                            "rank": result["rank"],
                            "case_id": result["case_id"],
                            "title_clean": result["title_clean"],
                            "case_cause": result["case_cause"],
                            "score": result["score"],
                        }
                        for result in results[:3]
                    ],
                }
            )

    total = len(queries)
    avg_latency = sum(latencies_ms) / total if total else 0.0
    p95_latency = (
        sorted(latencies_ms)[max(0, min(total - 1, math.ceil(total * 0.95) - 1))]
        if total
        else 0.0
    )
    return {
        "query_count": total,
        "top1_recall": round(top1_hits / total, 4) if total else 0.0,
        "top5_recall": round(top5_hits / total, 4) if total else 0.0,
        "avg_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "samples": samples,
    }


__all__ = [
    "CASE_RETRIEVAL_TOOL_NAME",
    "DEFAULT_CASE_DOCS_ENV",
    "DEFAULT_CASE_DOCS_PATH",
    "DEFAULT_TOP_K",
    "LocalCaseRetrievalEngine",
    "benchmark_queries",
    "build_retrieval_doc",
    "clean_title",
    "compress_judgment_text",
    "create_case_retrieval_tool",
    "create_case_search_function",
    "join_retrieval_fields",
    "normalize_text",
    "resolve_case_docs_path",
    "tokenize",
]
