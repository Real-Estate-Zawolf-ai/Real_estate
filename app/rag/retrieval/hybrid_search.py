"""Hybrid retrieval for the Real Estate RAG system.

Pipeline:

    Metadata Filtering
            ↓
    Dense Vector Search (Qdrant)
            +
    Optional Lexical Search (BM25)
            ↓
    Score Normalization
            ↓
    Weighted Hybrid Ranking

Important:

    - Query embeddings are generated through the shared EmbeddingModel.
    - The same embedding configuration must be used during indexing.
    - Metadata filters are applied to both semantic and lexical retrieval.
    - BM25 is optional and requires an explicitly supplied in-memory corpus.
    - When no corpus is supplied, retrieval safely falls back to dense search.
    - This module does not automatically load the full Qdrant collection
      into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from ..embeddings import EmbeddingModel
from .metadata_filter import MetadataFilter, matches


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_embedding_model: EmbeddingModel | None = None


def _get_embedding_model() -> EmbeddingModel:
    """Lazily initialize and reuse the shared embedding model."""

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model


def embed_query(query: str) -> list[float]:
    """Embed a search query using the shared EmbeddingModel."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    return _get_embedding_model().embed_query(query.strip())


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Normalized result returned by the retrieval layer."""

    property_id: str | None
    chunk_id: str | None
    text: str
    metadata: dict[str, Any]
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0
    source_file: str | None = None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _flatten_metadata(
    payload_or_chunk: dict[str, Any],
) -> dict[str, Any]:
    """Return a normalized metadata dictionary.

    Supports both:

        {
            "metadata": {...},
            "text": "...",
            ...
        }

    and:

        {
            "property_id": "...",
            "price": ...,
            "text": "...",
            ...
        }

    Top-level values take precedence over nested metadata values.
    """

    metadata = payload_or_chunk.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        **metadata,
        **{
            key: value
            for key, value in payload_or_chunk.items()
            if key != "metadata"
        },
    }


def _result_key(
    result: SearchResult,
    fallback_prefix: str,
    index: int,
) -> str:
    """Build a stable merge key for semantic and lexical results.

    Chunk ID is preferred because multiple chunks can belong to the same
    property. Property ID is used only as a fallback.
    """

    if result.chunk_id:
        return str(result.chunk_id)

    if result.property_id:
        return str(result.property_id)

    return f"{fallback_prefix}-{index}"


# ---------------------------------------------------------------------------
# Qdrant metadata filter
# ---------------------------------------------------------------------------


def _to_qdrant_filter(
    requirements_filter: MetadataFilter | None,
) -> dict[str, Any] | None:
    """Convert MetadataFilter into a Qdrant filter dictionary."""

    if requirements_filter is None or requirements_filter.is_empty():
        return None

    must_conditions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Equality constraints
    # ------------------------------------------------------------------

    for constraint in requirements_filter.equality:
        must_conditions.append(
            {
                "key": constraint.field,
                "match": {
                    "value": constraint.value,
                },
            }
        )

    # ------------------------------------------------------------------
    # Range constraints
    # ------------------------------------------------------------------

    for constraint in requirements_filter.ranges:
        range_condition: dict[str, Any] = {}

        if constraint.gte is not None:
            range_condition["gte"] = constraint.gte

        if constraint.lte is not None:
            range_condition["lte"] = constraint.lte

        if range_condition:
            must_conditions.append(
                {
                    "key": constraint.field,
                    "range": range_condition,
                }
            )

    if not must_conditions:
        return None

    return {
        "must": must_conditions,
    }


# ---------------------------------------------------------------------------
# Dense / semantic search
# ---------------------------------------------------------------------------


def _qdrant_search(
    query: str,
    filt: MetadataFilter | None,
    collection_name: str,
    qdrant_client: Any,
    top_k: int,
) -> list[SearchResult]:
    """Perform dense vector search against Qdrant."""

    if top_k <= 0:
        return []

    if qdrant_client is None:
        raise ValueError("qdrant_client must be provided")

    qdrant_filter = _to_qdrant_filter(filt)

    search_kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "limit": top_k,
        "query_vector": embed_query(query),
    }

    if qdrant_filter is not None:
        search_kwargs["query_filter"] = qdrant_filter

    points = qdrant_client.search(**search_kwargs)

    results: list[SearchResult] = []

    for point in points:
        payload = getattr(point, "payload", None) or {}

        if not isinstance(payload, dict):
            payload = {}

        score = float(getattr(point, "score", 0.0) or 0.0)

        flattened_metadata = _flatten_metadata(payload)

        text = payload.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        property_id = payload.get(
            "property_id",
            flattened_metadata.get("property_id"),
        )

        chunk_id = payload.get(
            "chunk_id",
            flattened_metadata.get("chunk_id"),
        )

        source_file = payload.get(
            "source_file",
            flattened_metadata.get("source_file"),
        )

        results.append(
            SearchResult(
                property_id=(
                    str(property_id)
                    if property_id is not None
                    else None
                ),
                chunk_id=(
                    str(chunk_id)
                    if chunk_id is not None
                    else None
                ),
                text=text,
                metadata=flattened_metadata,
                semantic_score=score,
                combined_score=score,
                source_file=(
                    str(source_file)
                    if source_file is not None
                    else None
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Metadata filtering for lexical retrieval
# ---------------------------------------------------------------------------


def _matches_requirements(
    chunk: dict[str, Any],
    requirements_filter: MetadataFilter | None,
) -> bool:
    """Apply hard metadata constraints to a lexical-search chunk."""

    if requirements_filter is None or requirements_filter.is_empty():
        return True

    flattened_metadata = _flatten_metadata(chunk)

    return matches(flattened_metadata, requirements_filter)


# ---------------------------------------------------------------------------
# BM25 / lexical search
# ---------------------------------------------------------------------------


def _bm25_search(
    query: str,
    corpus_chunks: list[dict[str, Any]],
    top_k: int,
    requirements_filter: MetadataFilter | None = None,
) -> list[SearchResult]:
    """Perform BM25 retrieval over an explicitly supplied corpus.

    Hard metadata constraints are applied before BM25 scoring.

    The function intentionally does not load data from Qdrant. The caller
    is responsible for supplying the lexical corpus.
    """

    if top_k <= 0 or not corpus_chunks:
        return []

    filtered_chunks = [
        chunk
        for chunk in corpus_chunks
        if isinstance(chunk, dict)
        and _matches_requirements(chunk, requirements_filter)
    ]

    if not filtered_chunks:
        return []

    documents: list[str] = []

    for chunk in filtered_chunks:
        text = chunk.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        documents.append(text)

    if not documents:
        return []

    tokenized_corpus = [
        document.lower().split()
        for document in documents
    ]

    if not tokenized_corpus:
        return []

    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = query.lower().split()

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results: list[SearchResult] = []

    for index in ranked_indices[:top_k]:
        score = float(scores[index])

        # BM25 score <= 0 means there is no useful lexical match.
        if score <= 0:
            continue

        chunk = filtered_chunks[index]
        flattened_metadata = _flatten_metadata(chunk)

        text = chunk.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        property_id = chunk.get(
            "property_id",
            flattened_metadata.get("property_id"),
        )

        chunk_id = chunk.get(
            "chunk_id",
            flattened_metadata.get("chunk_id"),
        )

        source_file = chunk.get(
            "source_file",
            flattened_metadata.get("source_file"),
        )

        results.append(
            SearchResult(
                property_id=(
                    str(property_id)
                    if property_id is not None
                    else None
                ),
                chunk_id=(
                    str(chunk_id)
                    if chunk_id is not None
                    else None
                ),
                text=text,
                metadata=flattened_metadata,
                lexical_score=score,
                combined_score=score,
                source_file=(
                    str(source_file)
                    if source_file is not None
                    else None
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores into the [0, 1] range.

    If all scores are identical, every score is treated as 1.0.
    """

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    denominator = max_score - min_score

    return [
        (score - min_score) / denominator
        for score in scores
    ]


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    requirements_filter: MetadataFilter | None = None,
    collection_name: str = "properties",
    qdrant_client: Any = None,
    top_k: int = 10,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
    corpus_chunks: list[dict[str, Any]] | None = None,
    *,
    client: Any = None,
) -> list[SearchResult]:
    """Perform dense retrieval with optional BM25 lexical retrieval.

    Args:
        query:
            User search query.

        requirements_filter:
            Hard metadata constraints.

        collection_name:
            Qdrant collection containing property chunks.

        qdrant_client:
            Qdrant client instance.

        client:
            Backwards-compatible alias for ``qdrant_client``.

        top_k:
            Maximum number of final results.

        semantic_weight:
            Weight assigned to dense semantic retrieval.

        lexical_weight:
            Weight assigned to BM25 lexical retrieval.

        corpus_chunks:
            Optional in-memory corpus used for BM25 retrieval.
            If omitted, retrieval is semantic-only.

    Returns:
        Search results sorted by ``combined_score`` descending.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if semantic_weight < 0 or lexical_weight < 0:
        raise ValueError("retrieval weights cannot be negative")

    total_weight = semantic_weight + lexical_weight

    if total_weight <= 0:
        raise ValueError(
            "semantic_weight + lexical_weight must be greater than 0"
        )

    query = query.strip()

    if qdrant_client is None:
        qdrant_client = client

    if qdrant_client is None:
        raise ValueError("qdrant_client must be provided")

    # Normalize weights so they sum to 1.
    semantic_weight /= total_weight
    lexical_weight /= total_weight

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    semantic_results = _qdrant_search(
        query=query,
        filt=requirements_filter,
        collection_name=collection_name,
        qdrant_client=qdrant_client,
        top_k=top_k,
    )

    # ------------------------------------------------------------------
    # Semantic-only retrieval
    # ------------------------------------------------------------------

    if corpus_chunks is None:
        semantic_scores = [
            result.semantic_score
            for result in semantic_results
        ]

        normalized_semantic = _min_max_normalize(
            semantic_scores
        )

        for result, normalized_score in zip(
            semantic_results,
            normalized_semantic,
        ):
            result.combined_score = normalized_score

        semantic_results.sort(
            key=lambda result: result.combined_score,
            reverse=True,
        )

        return semantic_results[:top_k]

    # ------------------------------------------------------------------
    # Lexical retrieval
    # ------------------------------------------------------------------

    lexical_results = _bm25_search(
        query=query,
        corpus_chunks=corpus_chunks,
        top_k=top_k,
        requirements_filter=requirements_filter,
    )

    # ------------------------------------------------------------------
    # Normalize scores
    # ------------------------------------------------------------------

    semantic_scores = [
        result.semantic_score
        for result in semantic_results
    ]

    lexical_scores = [
        result.lexical_score
        for result in lexical_results
    ]

    normalized_semantic = _min_max_normalize(
        semantic_scores
    )

    normalized_lexical = _min_max_normalize(
        lexical_scores
    )

    # ------------------------------------------------------------------
    # Merge results
    # ------------------------------------------------------------------

    merged: dict[str, SearchResult] = {}

    # Add semantic results.
    for index, (result, normalized_score) in enumerate(
        zip(
            semantic_results,
            normalized_semantic,
        )
    ):
        key = _result_key(
            result,
            fallback_prefix="semantic",
            index=index,
        )

        result.combined_score = (
            semantic_weight * normalized_score
        )

        merged[key] = result

    # Add or merge lexical results.
    for index, (result, normalized_score) in enumerate(
        zip(
            lexical_results,
            normalized_lexical,
        )
    ):
        key = _result_key(
            result,
            fallback_prefix="lexical",
            index=index,
        )

        if key in merged:
            existing = merged[key]

            existing.lexical_score = result.lexical_score

            existing.combined_score += (
                lexical_weight * normalized_score
            )

        else:
            result.combined_score = (
                lexical_weight * normalized_score
            )

            merged[key] = result

    # ------------------------------------------------------------------
    # Final ranking
    # ------------------------------------------------------------------

    results = sorted(
        merged.values(),
        key=lambda result: result.combined_score,
        reverse=True,
    )

    return results[:top_k]