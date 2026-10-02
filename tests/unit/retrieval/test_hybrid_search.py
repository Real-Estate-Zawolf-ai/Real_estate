import pytest
from importlib import import_module
from unittest.mock import MagicMock, patch
from app.rag.retrieval.hybrid_search import (
    hybrid_search,
    _min_max_normalize,
    _bm25_search,
    SearchResult
)
from app.rag.retrieval.metadata_filter import MetadataFilter


hybrid_search_module = import_module("app.rag.retrieval.hybrid_search")

def test_min_max_normalize():
    # Standard case
    assert _min_max_normalize([10, 20, 30]) == [0.0, 0.5, 1.0]
    # Empty case
    assert _min_max_normalize([]) == []
    # All same values
    assert _min_max_normalize([10, 10, 10]) == [1.0, 1.0, 1.0]

def test_bm25_search_basic():
    corpus = [
        {"chunk_id": "1", "text": "luxury apartment in New Cairo"},
        {"chunk_id": "2", "text": "cheap studio in Zayed"},
        {"chunk_id": "3", "text": "modern villa with pool"},
    ]
    # Search for "Cairo" - should match chunk 1
    results = _bm25_search("Cairo", corpus, top_k=1)
    chunk_ids = [r.chunk_id for r in results]

    assert chunk_ids == ["1"]
    assert "2" not in chunk_ids
    assert "3" not in chunk_ids
    assert results[0].lexical_score > 0

def test_hybrid_search_semantic_only():
    # Mock qdrant_client and the internal search function
    mock_client = MagicMock()
    filt = MetadataFilter()

    mock_results = [
        SearchResult(property_id="p1", chunk_id="c1", text="text1", metadata={}, semantic_score=0.9),
        SearchResult(property_id="p2", chunk_id="c2", text="text2", metadata={}, semantic_score=0.7),
    ]

    with patch.object(hybrid_search_module, "_qdrant_search", return_value=mock_results):
        # corpus_chunks=None triggers semantic-only path
        results = hybrid_search("query", filt, "coll", mock_client, corpus_chunks=None)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        # Raw semantic score is preserved; combined_score is min-max normalized.
        assert results[0].semantic_score == 0.9
        assert results[0].combined_score == pytest.approx(1.0)

def test_hybrid_search_combined():
    mock_client = MagicMock()
    filt = MetadataFilter()

    # Semantic results
    semantic_results = [
        SearchResult(property_id="p1", chunk_id="c1", text="Cairo apartment", metadata={}, semantic_score=0.8),
        SearchResult(property_id="p2", chunk_id="c2", text="Zayed villa", metadata={}, semantic_score=0.6),
    ]

    # Lexical corpus
    corpus = [
        {"chunk_id": "c1", "text": "Cairo apartment"},
        {"chunk_id": "c2", "text": "Zayed villa"},
        {"chunk_id": "c3", "text": "Cairo luxury"}, # lexical hit only
    ]

    with patch.object(hybrid_search_module, "_qdrant_search", return_value=semantic_results):
        # Search for "Cairo"
        results = hybrid_search("Cairo", filt, "coll", mock_client, corpus_chunks=corpus)

        # c1 should be top because it's in both
        assert results[0].chunk_id == "c1"
        # c3 should be present because it's a lexical hit
        chunk_ids = [r.chunk_id for r in results]
        assert "c3" in chunk_ids
