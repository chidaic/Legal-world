"""Local semantic law retrieval backed by file-based vectors."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from camel.embeddings import BaseEmbedding
from camel.toolkits import FunctionTool


logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "cn_law_articles"
DEFAULT_INDEX_MANIFEST_FILENAME = "law_vector_index_manifest.json"
DEFAULT_VECTOR_FILENAME = "law_embeddings.float16.npy"
DEFAULT_METADATA_FILENAME = "law_metadata.jsonl"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QUERY_CACHE_SIZE = 256
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60
DEFAULT_EMBEDDING_MAX_ATTEMPTS = 3
DEFAULT_EMBEDDING_LOG_BODY_LIMIT = 600

_LAW_TOOL_CACHE_LOCK = Lock()
_LAW_TOOL_CACHE: dict[tuple[str, str, int], "LawRetrievalTool"] = {}


class DashScopeMultiModalEmbedding(BaseEmbedding[str]):
    """DashScope native multimodal embedding client for text query vectors."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        output_dim: int,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = self._normalize_base_url(base_url)
        self.output_dim = output_dim

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = str(base_url or "").rstrip("/")
        if normalized.endswith("/compatible-mode/v1"):
            return normalized[: -len("/compatible-mode/v1")] + "/api/v1"
        return normalized

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            url=(
                f"{self.base_url}/services/embeddings/"
                "multimodal-embedding/multimodal-embedding"
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": {"contents": [{"text": text} for text in texts]},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise ValueError("DashScope multimodal embedding response missing embeddings.")

        vectors: list[list[float]] = []
        for item in embeddings:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError(
                    "DashScope multimodal embedding response missing embedding vector."
                )
            vectors.append(vector)
        return vectors

    def embed(self, obj: str) -> list[float]:
        return self.embed_list([obj])[0]

    def embed_list(self, objs: list[str]) -> list[list[float]]:
        texts = [str(obj) for obj in objs if str(obj).strip()]
        if not texts:
            raise ValueError("Embedding input must not be empty.")
        return self._request_embeddings(texts)

    def get_output_dim(self) -> int:
        return self.output_dim


class RetriableOpenAICompatibleEmbedding(BaseEmbedding[str]):
    """OpenAI-compatible embedding client with explicit response diagnostics."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        output_dim: int,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = str(base_url or "").rstrip("/")
        self.output_dim = output_dim

    def _request_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            url=f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": texts,
            },
            timeout=DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        )

        if response.status_code == 503:
            logger.warning(
                "Law embedding API returned 503 [model=%s base_url=%s body=%s]",
                self.model_name,
                self.base_url,
                _truncate_for_log(response.text),
            )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning(
                "Law embedding API returned non-JSON response [model=%s base_url=%s body=%s]",
                self.model_name,
                self.base_url,
                _truncate_for_log(response.text),
            )
            raise ValueError("Embedding API response is not valid JSON.") from exc

        if not isinstance(payload, dict):
            logger.warning(
                "Law embedding API returned non-object JSON [model=%s base_url=%s payload=%s]",
                self.model_name,
                self.base_url,
                _truncate_for_log(json.dumps(payload, ensure_ascii=False)),
            )
            raise ValueError("Embedding API response must be a JSON object.")

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            logger.warning(
                "Law embedding API returned empty data [model=%s base_url=%s payload=%s]",
                self.model_name,
                self.base_url,
                _truncate_for_log(json.dumps(payload, ensure_ascii=False)),
            )
            raise ValueError("No embedding data received")

        vectors: list[list[float]] = []
        for item in data:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                logger.warning(
                    "Law embedding API returned empty embedding vector [model=%s base_url=%s payload=%s]",
                    self.model_name,
                    self.base_url,
                    _truncate_for_log(json.dumps(payload, ensure_ascii=False)),
                )
                raise ValueError("No embedding data received")
            vectors.append(vector)

        return vectors

    def embed(self, obj: str) -> list[float]:
        return self.embed_list([obj])[0]

    def embed_list(self, objs: list[str]) -> list[list[float]]:
        texts = [str(obj) for obj in objs if str(obj).strip()]
        if not texts:
            raise ValueError("Embedding input must not be empty.")
        vectors = self._request_embeddings(texts)
        if vectors and self.output_dim is None:
            self.output_dim = len(vectors[0])
        return vectors

    def get_output_dim(self) -> int:
        return self.output_dim


def _truncate_for_log(value: Any, limit: int = DEFAULT_EMBEDDING_LOG_BODY_LIMIT) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _is_retryable_embedding_error(error: Exception) -> bool:
    if isinstance(error, requests.RequestException):
        return True
    message = str(error or "")
    return any(
        marker in message
        for marker in (
            "No embedding data received",
            "shell_api_error",
            "503",
            "temporarily unavailable",
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON object at {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                items.append(payload)
    return items


def _resolve_embedding_api_key() -> str:
    api_key = (
        os.environ.get("LAW_EMBEDDING_API_KEY")
        or os.environ.get("LAW_RETRIEVAL_EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_COMPATIBILITY_API_KEY")
    )
    if not api_key:
        raise ValueError("Missing embedding API key for law retrieval embeddings.")
    return api_key


def _resolve_embedding_api_url() -> str:
    base_url = (
        os.environ.get("LAW_EMBEDDING_API_BASE_URL")
        or os.environ.get("LAW_RETRIEVAL_EMBEDDING_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL")
    )
    if not base_url:
        raise ValueError("Missing embedding API base URL for law retrieval embeddings.")
    return base_url


def _resolve_embedding_model_name(manifest: dict[str, Any]) -> str:
    return (
        os.environ.get("LAW_EMBEDDING_MODEL")
        or os.environ.get("LAW_RETRIEVAL_EMBEDDING_MODEL")
        or os.environ.get("OPENAI_EMBEDDING_MODEL")
        or str(manifest.get("query_embedding_model_hint", "") or "").strip()
        or DEFAULT_EMBEDDING_MODEL
    )


def _build_default_embedding_model(
    manifest: dict[str, Any],
    output_dim: int,
) -> BaseEmbedding[str]:
    model_name = _resolve_embedding_model_name(manifest)
    api_key = _resolve_embedding_api_key()
    base_url = _resolve_embedding_api_url()

    if model_name == "multimodal-embedding-v1":
        return DashScopeMultiModalEmbedding(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            output_dim=output_dim,
        )

    return RetriableOpenAICompatibleEmbedding(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        output_dim=output_dim,
    )


class LawRetrievalTool:
    """Semantic law retrieval over exported local vector files."""

    def __init__(
        self,
        storage_path: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = 5,
        embedding_model: Optional[BaseEmbedding[str]] = None,
    ) -> None:
        self.index_dir = Path(storage_path).resolve()
        self.collection_name = collection_name
        self.top_k = top_k

        self.manifest_path = self.index_dir / DEFAULT_INDEX_MANIFEST_FILENAME
        self.manifest = _load_json(self.manifest_path)
        self.vector_path = self.index_dir / str(
            self.manifest.get("vector_file", DEFAULT_VECTOR_FILENAME)
        )
        self.metadata_path = self.index_dir / str(
            self.manifest.get("metadata_file", DEFAULT_METADATA_FILENAME)
        )

        self.vectors = np.load(self.vector_path, mmap_mode="r")
        self.records = _load_jsonl(self.metadata_path)

        if self.vectors.ndim != 2 or self.vectors.shape[0] == 0:
            raise ValueError(f"Invalid law vector matrix: {self.vector_path}")
        if len(self.records) != int(self.vectors.shape[0]):
            raise ValueError(
                "Law metadata count does not match vector count: "
                f"{len(self.records)} != {self.vectors.shape[0]}"
            )

        manifest_dim = int(self.manifest.get("vector_dim") or self.vectors.shape[1])
        if int(self.vectors.shape[1]) != manifest_dim:
            raise ValueError(
                f"Law vector dim mismatch: matrix={self.vectors.shape[1]}, manifest={manifest_dim}"
            )

        self.vector_dim = manifest_dim
        self.embedding_model = embedding_model or _build_default_embedding_model(
            self.manifest,
            output_dim=self.vector_dim,
        )
        self._query_cache_lock = Lock()
        self._query_cache: OrderedDict[
            tuple[str, int],
            list[dict[str, Any]],
        ] = OrderedDict()

    def _embed_query(self, query: str) -> np.ndarray:
        model_name = _resolve_embedding_model_name(self.manifest)
        base_url = _resolve_embedding_api_url()
        vector: np.ndarray | None = None
        last_error: Exception | None = None

        for attempt in range(1, DEFAULT_EMBEDDING_MAX_ATTEMPTS + 1):
            try:
                vector = np.asarray(
                    self.embedding_model.embed(query),
                    dtype=np.float32,
                )
                if attempt > 1:
                    logger.info(
                        "Law embedding recovered after retry [attempt=%s model=%s base_url=%s]",
                        attempt,
                        model_name,
                        base_url,
                    )
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Law embedding attempt failed [attempt=%s/%s model=%s base_url=%s error=%s]",
                    attempt,
                    DEFAULT_EMBEDDING_MAX_ATTEMPTS,
                    model_name,
                    base_url,
                    exc,
                )
                if (
                    attempt >= DEFAULT_EMBEDDING_MAX_ATTEMPTS
                    or not _is_retryable_embedding_error(exc)
                ):
                    raise
                time.sleep(float(attempt))

        if vector is None:
            raise RuntimeError(
                "Law embedding query failed without returning a vector."
            ) from last_error

        if vector.ndim != 1 or vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Query embedding dim mismatch: expected {self.vector_dim}, got {vector.shape}"
            )

        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Query embedding norm is zero.")
        return vector / norm

    def _cache_get(self, query: str, top_k: int) -> Optional[list[dict[str, Any]]]:
        cache_key = (query, top_k)
        with self._query_cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached is None:
                return None
            self._query_cache.move_to_end(cache_key)
            return [dict(item) for item in cached]

    def _cache_set(self, query: str, top_k: int, results: list[dict[str, Any]]) -> None:
        cache_key = (query, top_k)
        with self._query_cache_lock:
            self._query_cache[cache_key] = [dict(item) for item in results]
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > DEFAULT_QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)

    def search_laws(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            raise ValueError("query must not be empty.")

        k = int(top_k or self.top_k)
        if k <= 0:
            raise ValueError("top_k must be greater than 0.")
        k = min(k, len(self.records))

        cached = self._cache_get(query_text, k)
        if cached is not None:
            return cached

        query_vector = self._embed_query(query_text)
        scores = np.asarray(self.vectors @ query_vector, dtype=np.float32)

        if k >= scores.shape[0]:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[dict[str, Any]] = []
        for index in top_indices.tolist():
            payload = dict(self.records[index])
            payload["similarity"] = float(scores[index])
            payload["retrieval_backend"] = "semantic_vector"
            results.append(payload)

        self._cache_set(query_text, k, results)
        return results

    def get_database_info(self) -> Dict[str, Any]:
        return {
            "storage_path": str(self.index_dir),
            "collection_name": self.collection_name,
            "status": "ready",
            "point_count": int(self.vectors.shape[0]),
            "vector_dim": self.vector_dim,
            "vector_file": str(self.vector_path),
            "metadata_file": str(self.metadata_path),
            "manifest_file": str(self.manifest_path),
            "embedding_model": _resolve_embedding_model_name(self.manifest),
            "retrieval_backend": "semantic_vector",
        }


def _get_shared_law_tool(
    storage_path: str,
    collection_name: str,
    top_k: int,
) -> LawRetrievalTool:
    resolved_path = str(Path(storage_path).resolve())
    cache_key = (resolved_path, collection_name, top_k)

    with _LAW_TOOL_CACHE_LOCK:
        tool = _LAW_TOOL_CACHE.get(cache_key)
        if tool is None:
            tool = LawRetrievalTool(
                storage_path=resolved_path,
                collection_name=collection_name,
                top_k=top_k,
            )
            _LAW_TOOL_CACHE[cache_key] = tool
        return tool


def create_law_search_function(
    storage_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    agent: Any = None,
):
    """Create the law search function used by agent tools."""

    def _get_tool() -> LawRetrievalTool:
        return _get_shared_law_tool(
            storage_path=storage_path,
            collection_name=collection_name,
            top_k=top_k,
        )

    def search_laws(query: str, top_k: int = 5) -> str:
        results = _get_tool().search_laws(query, top_k)
        return json.dumps(results, ensure_ascii=False, indent=2)

    return search_laws


def _build_search_laws_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_laws",
            "description": (
                "检索与当前法律问题最相关的法条。优先用一个短而聚焦的 query；"
                "如果首轮结果明显不相关，只再改写一次，不要连续进行多次近似重复检索。"
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "简短、聚焦的法律检索短语。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关法条数量；不传时默认 5。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def create_law_retrieval_tool(
    storage_path: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    agent: Any = None,
) -> FunctionTool:
    """Create the FunctionTool wrapper for semantic law retrieval."""
    if storage_path is None:
        default_path = Path(__file__).parent.parent.parent.parent / "shared" / "cn_law"
        storage_path = str(default_path)

    search_func = create_law_search_function(
        storage_path=storage_path,
        collection_name=collection_name,
        top_k=top_k,
        agent=agent,
    )

    return FunctionTool(search_func, openai_tool_schema=_build_search_laws_schema())


__all__ = [
    "LawRetrievalTool",
    "create_law_retrieval_tool",
    "create_law_search_function",
]
