"""Build file-based law corpus indexes for local semantic retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REQUIRED_DOCUMENT_FIELDS = (
    "document_id",
    "source_title",
    "category",
    "article_ref",
    "title",
    "content",
    "effective_date",
    "source_url",
)


def validate_processed_document(document: dict[str, Any]) -> None:
    missing_fields = [
        field for field in REQUIRED_DOCUMENT_FIELDS if not str(document.get(field, "")).strip()
    ]
    if missing_fields:
        raise ValueError(f"processed law document missing fields: {', '.join(missing_fields)}")


def _build_embedding_text(document: dict[str, Any]) -> str:
    title = str(document.get("title") or "").strip()
    content = str(document.get("content") or "").strip()
    if title:
        return f"{title}\n{content}"
    return content


def _normalize_vector(vector: Iterable[float], expected_output_dim: int) -> list[float]:
    normalized = [float(value) for value in vector]
    if len(normalized) != expected_output_dim:
        raise ValueError(
            f"embedding dim mismatch: expected {expected_output_dim}, got {len(normalized)}"
        )
    return normalized


def build_index(
    documents: list[dict[str, Any]],
    output_dir: str | Path,
    embedding_model: Any,
    embedding_model_name: str,
    expected_output_dim: int | None = None,
) -> None:
    if not documents:
        raise ValueError("documents must not be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validated_documents: list[dict[str, Any]] = []
    for document in documents:
        validate_processed_document(document)
        validated_documents.append(dict(document))

    output_dim = int(expected_output_dim or embedding_model.get_output_dim())
    embeddings = embedding_model.embed_list(
        [_build_embedding_text(document) for document in validated_documents]
    )
    normalized_embeddings = [
        _normalize_vector(vector, expected_output_dim=output_dim) for vector in embeddings
    ]

    metadata_path = output_path / "law_metadata.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as file:
        for document in validated_documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    embeddings_path = output_path / "law_embeddings.float16.npy"
    np.save(embeddings_path, normalized_embeddings)

    manifest = {
        "collection_name": "cn_law_articles",
        "document_count": len(validated_documents),
        "vector_dim": output_dim,
        "vector_file": embeddings_path.name,
        "metadata_file": metadata_path.name,
        "query_embedding_model_hint": embedding_model_name,
    }

    manifest_path = output_path / "law_vector_index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


__all__ = [
    "build_index",
    "validate_processed_document",
]
