"""
reranker.py

Reranks the candidate set from hybrid_search.py using a cross-encoder
model — which scores (query, chunk) pairs jointly and is much more
accurate than the bi-encoder similarity used for initial retrieval,
at the cost of being too slow to run over the whole corpus (hence:
retrieve broad with hybrid_search, then rerank the top-N here).

Open-source model: BAAI/bge-reranker-v2-m3 — multilingual (covers
Arabic + English, matching this project's language requirement).

Install:
    pip install sentence-transformers
"""

from dataclasses import replace
from typing import List, Optional

from .hybrid_search import SearchResult

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(MODEL_NAME)
    return _reranker_model


def rerank(
    query: str,
    candidates: List[SearchResult],
    top_k: int = 5,
) -> List[SearchResult]:
    """Score each candidate against the query with the cross-encoder and
    return the top_k, sorted best-first.

    Each returned SearchResult has `combined_score` overwritten with the
    reranker's score, so downstream code (the LLM prompt builder) can
    treat `combined_score` as "final relevance" regardless of whether
    reranking ran.
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    reranked = [
        replace(candidate, combined_score=float(score))
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda r: r.combined_score, reverse=True)
    return reranked[:top_k]


def rerank_with_trace(
    query: str,
    candidates: List[SearchResult],
    top_k: int = 5,
) -> List[dict]:
    """Same as `rerank`, but returns plain dicts including both the
    pre-rerank (hybrid) score and the post-rerank score side by side —
    useful for observability/tracing (per the project's requirement to
    log retrieval + reranking results for every request).
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    trace = [
        {
            "property_id": c.property_id,
            "chunk_id": c.chunk_id,
            "pre_rerank_score": c.combined_score,
            "rerank_score": float(score),
            "source_file": c.source_file,
        }
        for c, score in zip(candidates, scores)
    ]
    trace.sort(key=lambda t: t["rerank_score"], reverse=True)
    return trace[:top_k]