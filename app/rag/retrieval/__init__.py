"""
Retrieval module for the AI Real Estate Assistant.

    Customer Requirements
            ↓
    metadata_filter.build_metadata_filter()   -> MetadataFilter (hard constraints)
            ↓
    hybrid_search.hybrid_search()             -> semantic + lexical candidates
            ↓
    reranker.rerank()                         -> final top-K, most relevant first
            ↓
    LLM (grounded generation)
"""

from .metadata_filter import (
    build_metadata_filter,
    matches,
    MetadataFilter,
    EqualityConstraint,
    RangeConstraint,
)
from . import hybrid_search
from .hybrid_search import SearchResult, embed_query
from .reranker import rerank, rerank_with_trace

__all__ = [
    "build_metadata_filter",
    "matches",
    "MetadataFilter",
    "EqualityConstraint",
    "RangeConstraint",
    "hybrid_search",
    "embed_query",
    "SearchResult",
    "rerank",
    "rerank_with_trace",
]
