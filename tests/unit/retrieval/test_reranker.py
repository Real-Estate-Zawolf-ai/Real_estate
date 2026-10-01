import pytest
from unittest.mock import MagicMock, patch

from app.rag.retrieval.reranker import rerank, rerank_with_trace
from app.rag.retrieval.hybrid_search import SearchResult


@pytest.fixture
def candidates():
    return [
        SearchResult(
            property_id="p1",
            chunk_id="c1",
            text="High relevance text",
            metadata={"category": "villa"},
            combined_score=0.5,
            source_file="properties.csv",
        ),
        SearchResult(
            property_id="p2",
            chunk_id="c2",
            text="Medium relevance text",
            metadata={"category": "apartment"},
            combined_score=0.4,
            source_file="properties.csv",
        ),
        SearchResult(
            property_id="p3",
            chunk_id="c3",
            text="Low relevance text",
            metadata={"category": "villa"},
            combined_score=0.3,
            source_file="properties.csv",
        ),
    ]


def test_rerank_sorting(candidates):
    """Results should be sorted by cross-encoder score."""
    mock_scores = [0.1, 0.2, 0.9]

    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        results = rerank("query", candidates, top_k=2)

    assert len(results) == 2

    assert results[0].chunk_id == "c3"
    assert results[0].combined_score == 0.9

    assert results[1].chunk_id == "c2"
    assert results[1].combined_score == 0.2


def test_rerank_preserves_original_score_in_metadata(candidates):
    """The original hybrid score should remain available for traceability."""
    mock_scores = [0.1, 0.2, 0.9]

    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        results = rerank("query", candidates, top_k=3)

    result_by_chunk = {
        result.chunk_id: result
        for result in results
    }

    assert result_by_chunk["c1"].metadata["pre_rerank_score"] == 0.5
    assert result_by_chunk["c1"].metadata["rerank_score"] == 0.1

    assert result_by_chunk["c2"].metadata["pre_rerank_score"] == 0.4
    assert result_by_chunk["c2"].metadata["rerank_score"] == 0.2

    assert result_by_chunk["c3"].metadata["pre_rerank_score"] == 0.3
    assert result_by_chunk["c3"].metadata["rerank_score"] == 0.9


def test_rerank_preserves_source_file(candidates):
    """Reranking must not lose source traceability."""
    mock_scores = [0.1, 0.2, 0.9]

    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        results = rerank("query", candidates, top_k=3)

    assert all(
        result.source_file == "properties.csv"
        for result in results
    )


def test_rerank_does_not_mutate_original_candidates(candidates):
    """Reranking should return replaced SearchResult objects."""
    original_scores = [
        candidate.combined_score
        for candidate in candidates
    ]

    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2, 0.9]
        mock_get.return_value = mock_model

        results = rerank("query", candidates, top_k=3)

    assert [candidate.combined_score for candidate in candidates] == original_scores
    assert results[0] is not candidates[2]


def test_rerank_empty():
    assert rerank("query", [], top_k=5) == []


def test_rerank_empty_query():
    assert rerank("", [], top_k=5) == []
    assert rerank("   ", [], top_k=5) == []


def test_rerank_non_string_query():
    assert rerank(None, [], top_k=5) == []


@pytest.mark.parametrize("top_k", [0, -1, -10])
def test_rerank_invalid_top_k(candidates, top_k):
    assert rerank("query", candidates, top_k=top_k) == []


def test_rerank_does_not_load_model_for_invalid_input():
    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        assert rerank("", [], top_k=5) == []
        assert rerank("query", [], top_k=5) == []
        assert rerank("query", [], top_k=0) == []

        mock_get.assert_not_called()


def test_rerank_raises_on_score_count_mismatch(candidates):
    """A model returning the wrong number of scores is a runtime error."""
    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()

        # Only two scores for three candidates.
        mock_model.predict.return_value = [0.1, 0.9]
        mock_get.return_value = mock_model

        with pytest.raises(
            RuntimeError,
            match="Reranker output count does not match candidate count",
        ):
            rerank("query", candidates, top_k=3)


def test_rerank_with_trace(candidates):
    mock_scores = [0.1, 0.2, 0.9]

    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_scores
        mock_get.return_value = mock_model

        trace = rerank_with_trace(
            "query",
            candidates,
            top_k=2,
        )

    assert len(trace) == 2

    assert trace[0]["chunk_id"] == "c3"
    assert trace[0]["rerank_score"] == 0.9
    assert trace[0]["pre_rerank_score"] == 0.3
    assert trace[0]["source_file"] == "properties.csv"

    assert trace[1]["chunk_id"] == "c2"
    assert trace[1]["rerank_score"] == 0.2
    assert trace[1]["pre_rerank_score"] == 0.4
    assert trace[1]["source_file"] == "properties.csv"


def test_rerank_with_trace_empty():
    assert rerank_with_trace("query", [], top_k=5) == []


def test_rerank_with_trace_invalid_input(candidates):
    assert rerank_with_trace("", candidates, top_k=5) == []
    assert rerank_with_trace("   ", candidates, top_k=5) == []
    assert rerank_with_trace(None, candidates, top_k=5) == []

    assert rerank_with_trace("query", candidates, top_k=0) == []
    assert rerank_with_trace("query", candidates, top_k=-1) == []


def test_rerank_with_trace_raises_on_score_count_mismatch(candidates):
    with patch(
        "app.rag.retrieval.reranker._get_reranker"
    ) as mock_get:
        mock_model = MagicMock()

        # Only two scores for three candidates.
        mock_model.predict.return_value = [0.1, 0.9]
        mock_get.return_value = mock_model

        with pytest.raises(
            RuntimeError,
            match="Reranker output count does not match candidate count",
        ):
            rerank_with_trace(
                "query",
                candidates,
                top_k=3,
            )
