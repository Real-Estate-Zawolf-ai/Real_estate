"""Cross-encoder reranking for the Real Estate RAG system.

Retrieval flow:

    Metadata Filtering
            ↓
    Hybrid Search (Dense + Optional Lexical)
            ↓
    Top-N Candidates
            ↓
    Cross-Encoder Reranking
            ↓
    Final Top-K Results

The default model is BAAI/bge-reranker-v2-m3, configured through the
application settings. The model supports multilingual retrieval,
including Arabic and English.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from app.helpers.config import get_settings

from .hybrid_search import SearchResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

settings = get_settings()

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

CONFIGURED_MODEL_NAME = (
    settings.RERANKER_MODEL or DEFAULT_MODEL_NAME
)

RERANKER_DEVICE = settings.RERANKER_DEVICE


# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------

_reranker_model: Any = None


def _get_reranker() -> Any:
    """Load and cache the cross-encoder model."""

    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker model | model=%s | device=%s",
            CONFIGURED_MODEL_NAME,
            RERANKER_DEVICE or "auto",
        )

        kwargs: dict[str, Any] = {}

        if RERANKER_DEVICE:
            kwargs["device"] = RERANKER_DEVICE

        _reranker_model = CrossEncoder(
            CONFIGURED_MODEL_NAME,
            **kwargs,
        )

        logger.info(
            "Reranker model loaded successfully | model=%s",
            CONFIGURED_MODEL_NAME,
        )

    return _reranker_model


# ---------------------------------------------------------------------------
# Score prediction
# ---------------------------------------------------------------------------


def _predict_scores(
    query: str,
    candidates: list[SearchResult],
) -> list[float]:
    """Generate cross-encoder relevance scores for candidate chunks."""

    if not candidates:
        return []

    model = _get_reranker()

    pairs = [
        (query, candidate.text)
        for candidate in candidates
    ]

    raw_scores = model.predict(pairs)

    try:
        return [float(score) for score in raw_scores]
    except TypeError:
        return [float(raw_scores)]


# ---------------------------------------------------------------------------
# Apply reranker scores
# ---------------------------------------------------------------------------


def _apply_reranker_scores(
    candidates: list[SearchResult],
    scores: list[float],
) -> list[SearchResult]:
    """Return candidates with reranker scores applied."""

    if len(scores) != len(candidates):
        raise RuntimeError(
            "Reranker output count does not match candidate count"
        )

    reranked: list[SearchResult] = []

    for candidate, score in zip(candidates, scores):
        pre_rerank_score = candidate.combined_score

        metadata = dict(candidate.metadata)

        metadata["pre_rerank_score"] = pre_rerank_score
        metadata["rerank_score"] = score

        reranked.append(
            replace(
                candidate,
                metadata=metadata,
                combined_score=score,
            )
        )

    reranked.sort(
        key=lambda result: result.combined_score,
        reverse=True,
    )

    return reranked


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Rerank retrieved candidates using the cross-encoder.

    Args:
        query:
            User's search query.

        candidates:
            Candidate SearchResult objects produced by hybrid_search().

        top_k:
            Maximum number of results to return.

    Returns:
        SearchResult objects sorted by reranker score, best first.

    Notes:
        The original hybrid score is preserved as
        ``metadata["pre_rerank_score"]``.

        ``combined_score`` is replaced with the final reranker score.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if not candidates:
        return []

    if top_k <= 0:
        return []

    query = query.strip()

    scores = _predict_scores(
        query=query,
        candidates=candidates,
    )

    reranked = _apply_reranker_scores(
        candidates=candidates,
        scores=scores,
    )

    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Reranking with trace
# ---------------------------------------------------------------------------


def rerank_with_trace(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank candidates and return observability-friendly dictionaries.

    Each returned dictionary contains:

        - property_id
        - chunk_id
        - pre_rerank_score
        - rerank_score
        - source_file
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if not candidates:
        return []

    if top_k <= 0:
        return []

    query = query.strip()

    scores = _predict_scores(
        query=query,
        candidates=candidates,
    )

    if len(scores) != len(candidates):
        raise RuntimeError(
            "Reranker output count does not match candidate count"
        )

    trace = [
        {
            "property_id": candidate.property_id,
            "chunk_id": candidate.chunk_id,
            "pre_rerank_score": candidate.combined_score,
            "rerank_score": score,
            "source_file": candidate.source_file,
        }
        for candidate, score in zip(candidates, scores)
    ]

    trace.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return trace[:top_k]