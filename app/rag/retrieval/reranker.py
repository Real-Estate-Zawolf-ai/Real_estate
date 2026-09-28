"""
reranker.py

Reranks the candidate set from hybrid_search.py using a cross-encoder.

The cross-encoder scores (query, chunk) pairs jointly and is generally
more accurate than the bi-encoder similarity used for initial retrieval.

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

Open-source model:

    BAAI/bge-reranker-v2-m3

This model supports multilingual retrieval, including Arabic and English.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Any, List

from .hybrid_search import SearchResult


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Optional environment override.
# Examples:
#
#   RERANKER_MODEL=BAAI/bge-reranker-v2-m3
#
# This keeps the default model explicit while allowing configuration
# without changing application code.
CONFIGURED_MODEL_NAME = os.getenv(
    "RERANKER_MODEL",
    MODEL_NAME,
)

# Optional device override.
#
# Examples:
#
#   RERANKER_DEVICE=cpu
#   RERANKER_DEVICE=mps
#
# If not provided, sentence-transformers will choose automatically.
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE")


# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------
#
# Loading a cross-encoder is expensive, so initialize it only when needed
# and reuse the same model for subsequent requests.

_reranker_model: Any = None


def _get_reranker():
    """
    Load and cache the cross-encoder model.
    """
    global _reranker_model

    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker model | model=%s | device=%s",
            CONFIGURED_MODEL_NAME,
            RERANKER_DEVICE or "auto",
        )

        if RERANKER_DEVICE:
            _reranker_model = CrossEncoder(
                CONFIGURED_MODEL_NAME,
                device=RERANKER_DEVICE,
            )
        else:
            _reranker_model = CrossEncoder(
                CONFIGURED_MODEL_NAME,
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
    candidates: List[SearchResult],
) -> List[float]:
    """
    Generate cross-encoder relevance scores for candidate chunks.

    Keeping inference in one helper ensures rerank() and
    rerank_with_trace() use exactly the same prediction logic.
    """
    if not candidates:
        return []

    model = _get_reranker()

    pairs = [
        (query, candidate.text)
        for candidate in candidates
    ]

    raw_scores = model.predict(pairs)

    # sentence-transformers normally returns a NumPy array/list.
    # Converting explicitly gives the rest of the application a stable
    # Python representation.
    try:
        return [float(score) for score in raw_scores]
    except TypeError:
        return [float(raw_scores)]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: List[SearchResult],
    top_k: int = 5,
) -> List[SearchResult]:
    """
    Rerank retrieved candidates using the cross-encoder.

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
        The original hybrid score is preserved in the result metadata as
        ``pre_rerank_score``.

        ``combined_score`` is overwritten with the final cross-encoder
        reranking score so downstream generation code can treat it as the
        final relevance score.
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

    reranked: List[SearchResult] = []

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

    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Reranking with trace
# ---------------------------------------------------------------------------


def rerank_with_trace(
    query: str,
    candidates: List[SearchResult],
    top_k: int = 5,
) -> List[dict]:
    """
    Rerank candidates and return observability-friendly dictionaries.

    Each returned dictionary contains both:

        - pre_rerank_score
        - rerank_score

    This is useful for logging retrieval/reranking behavior and debugging
    relevance issues.
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

