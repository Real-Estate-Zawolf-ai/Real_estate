"""
hybrid_search.py

Combines three retrieval signals into one candidate set, per the
project's retrieval architecture:

    Metadata Filtering  (hard constraints — see metadata_filter.py)
            +
    Semantic Search     (dense embeddings via Qdrant)
            +
    Lexical Search      (keyword/BM25 — catches exact terms embeddings
                          can miss: project names, unit codes, "التجمع")
            ↓
       Candidate Set  →  passed to reranker.py

Vector DB: Qdrant (matches the project's stated stack). The Qdrant
client call is isolated in `_qdrant_search` so swapping backends later
only touches one function.

Install:
    pip install qdrant-client rank-bm25 sentence-transformers
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .metadata_filter import MetadataFilter


@dataclass
class SearchResult:
    """One retrieved property chunk, before reranking."""
    property_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0
    source_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Embedding + Qdrant backend (semantic search)
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    """Lazily load a sentence-transformers embedding model.

    Swap MODEL_NAME for any open embedding model — 'BAAI/bge-m3' is a
    strong choice if you need good Arabic + English coverage, which
    this project requires.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        MODEL_NAME = "BAAI/bge-m3"
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def embed_query(query: str) -> List[float]:
    model = _get_embedding_model()
    return model.encode(query, normalize_embeddings=True).tolist()


def _to_qdrant_filter(filt: MetadataFilter):
    """Translate our backend-agnostic MetadataFilter into Qdrant's
    filter DSL (qdrant_client.models.Filter).
    """
    from qdrant_client import models as qmodels

    must = []
    for c in filt.equality:
        must.append(
            qmodels.FieldCondition(key=c.field, match=qmodels.MatchValue(value=c.value))
        )
    for r in filt.ranges:
        must.append(
            qmodels.FieldCondition(
                key=r.field,
                range=qmodels.Range(gte=r.gte, lte=r.lte),
            )
        )
    return qmodels.Filter(must=must) if must else None


def _qdrant_search(
    query: str,
    filt: MetadataFilter,
    collection_name: str,
    qdrant_client,
    top_k: int,
) -> List[SearchResult]:
    query_vector = embed_query(query)
    qdrant_filter = _to_qdrant_filter(filt)

    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    results = []
    for hit in hits:
        payload = hit.payload or {}
        results.append(SearchResult(
            property_id=payload.get("property_id", ""),
            chunk_id=str(hit.id),
            text=payload.get("text", ""),
            metadata=payload,
            semantic_score=float(hit.score),
            source_file=payload.get("source_file"),
        ))
    return results


# ---------------------------------------------------------------------------
# Lexical / keyword search (BM25)
# ---------------------------------------------------------------------------

def _bm25_search(
    query: str,
    corpus_chunks: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, float]:
    """BM25 over an in-memory corpus of chunks (each a dict with at least
    "chunk_id" and "text"). In production, `corpus_chunks` should come
    from the same ingestion output that feeds the vector DB, cached in
    memory or a lightweight keyword index (e.g. Qdrant's own sparse
    vectors, Elasticsearch, or Postgres full-text search) rather than
    re-tokenized on every query.

    Returns {chunk_id: bm25_score} for the top_k lexical matches.
    """
    from rank_bm25 import BM25Okapi

    if not corpus_chunks:
        return {}

    tokenized_corpus = [c["text"].split() for c in corpus_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())

    scored = sorted(
        zip((c["chunk_id"] for c in corpus_chunks), scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return {chunk_id: float(score) for chunk_id, score in scored if score > 0}


# ---------------------------------------------------------------------------
# Hybrid combination
# ---------------------------------------------------------------------------

def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_search(
    query: str,
    requirements_filter: MetadataFilter,
    collection_name: str,
    qdrant_client,
    corpus_chunks: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 20,
    semantic_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> List[SearchResult]:
    """Run semantic + lexical search under the same metadata filter and
    merge into a single ranked candidate list (pre-reranking).

    `corpus_chunks` is optional: if you don't have an in-memory/keyword
    index wired up yet, this degrades gracefully to semantic-only
    search (still filtered by hard constraints) — useful early in
    development before the lexical layer is built.
    """
    semantic_results = _qdrant_search(
        query, requirements_filter, collection_name, qdrant_client, top_k=top_k
    )

    if not corpus_chunks:
        for r in semantic_results:
            r.combined_score = r.semantic_score
        return sorted(semantic_results, key=lambda r: r.combined_score, reverse=True)

    lexical_scores = _bm25_search(query, corpus_chunks, top_k=top_k)

    # Merge score spaces: normalize each independently before combining,
    # since BM25 and cosine-similarity scores are on different scales.
    sem_ids = [r.chunk_id for r in semantic_results]
    sem_norm = _min_max_normalize([r.semantic_score for r in semantic_results])
    for r, norm_score in zip(semantic_results, sem_norm):
        r.semantic_score = norm_score

    lex_ids = list(lexical_scores.keys())
    lex_norm = _min_max_normalize(list(lexical_scores.values()))
    lexical_norm_map = dict(zip(lex_ids, lex_norm))

    by_chunk: Dict[str, SearchResult] = {r.chunk_id: r for r in semantic_results}

    # Fold in lexical-only hits that semantic search missed.
    chunk_lookup = {c["chunk_id"]: c for c in corpus_chunks}
    for chunk_id in lex_ids:
        if chunk_id not in by_chunk and chunk_id in chunk_lookup:
            chunk = chunk_lookup[chunk_id]
            by_chunk[chunk_id] = SearchResult(
                property_id=chunk.get("property_id", ""),
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}),
                source_file=chunk.get("source_file"),
            )

    for chunk_id, result in by_chunk.items():
        result.lexical_score = lexical_norm_map.get(chunk_id, 0.0)
        result.combined_score = (
            semantic_weight * result.semantic_score
            + lexical_weight * result.lexical_score
        )

    return sorted(by_chunk.values(), key=lambda r: r.combined_score, reverse=True)[:top_k]