import pytest
from unittest.mock import MagicMock, patch
from app.rag.retrieval.reranker import rerank, rerank_with_trace
from app.rag.retrieval.hybrid_search import SearchResult

@pytest.fixture
def candidates():
    return [
        SearchResult(property_id="p1", chunk_id="c1", text="High relevance text", metadata={}, combined_score=0.5),
        SearchResult(property_id="p2", chunk_id="c2", text="Medium relevance text", metadata={}, combined_score=0.4),
        SearchResult(property_id="p3", chunk_id="c3", text="Low relevance text", metadata={}, combined_score=0.3),
    ]

def test_rerank_sorting(candidates):
    # Mock CrossEncoder.predict to return specific scores
    # We want c3 to actually be the best one according to the reranker
    mock_scores = [0.1, 0.2, 0.9] # scores for c1, c2, c3

    with patch("app.rag.retrieval.reranker._get_reranker") as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        results = rerank("query", candidates, top_k=2)

        assert len(results) == 2
        # c3 should now be first
        assert results[0].chunk_id == "c3"
        assert results[0].combined_score == 0.9
        assert results[1].chunk_id == "c2"
        assert results[1].combined_score == 0.2

def test_rerank_empty():
    assert rerank("query", [], top_k=5) == []

def test_rerank_with_trace(candidates):
    mock_scores = [0.1, 0.2, 0.9]

    with patch("app.rag.retrieval.reranker._get_reranker") as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        trace = rerank_with_trace("query", candidates, top_k=2)

        assert len(trace) == 2
        assert trace[0]["chunk_id"] == "c3"
        assert trace[0]["rerank_score"] == 0.9
        assert trace[0]["pre_rerank_score"] == 0.3 # original score from candidate
        assert trace[1]["chunk_id"] == "c2"
        assert trace[1]["rerank_score"] == 0.2
