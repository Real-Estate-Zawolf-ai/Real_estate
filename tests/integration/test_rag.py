"""Integration checks for the retrieval-to-answer RAG flow.

External services are replaced with small fakes so these tests exercise
the project's RAG components without requiring Qdrant, Groq, or API keys.
"""

from importlib import import_module
from unittest.mock import MagicMock

import pytest

from app.rag import generation
from app.rag.retrieval.hybrid_search import SearchResult
from app.rag.retrieval.metadata_filter import build_metadata_filter, matches

retrieval = import_module("app.rag.retrieval.hybrid_search")


@pytest.fixture
def property_chunks():
    return [
        SearchResult(
            property_id="A102",
            chunk_id="chunk-a102",
            text="3 bedroom apartment in New Cairo, price 4,800,000 EGP",
            metadata={
                "location": "New Cairo",
                "bedrooms": 3,
                "price": 4_800_000,
                "availability": "available",
                "payment_plan": "installments",
            },
            semantic_score=0.92,
            source_file="a102.pdf",
        ),
        SearchResult(
            property_id="B205",
            chunk_id="chunk-b205",
            text="2 bedroom apartment in Sheikh Zayed, price 4,000,000 EGP",
            metadata={
                "location": "Sheikh Zayed",
                "bedrooms": 2,
                "price": 4_000_000,
                "availability": "available",
            },
            semantic_score=0.99,
            source_file="b205.pdf",
        ),
    ]


def _fake_completion(text):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text))]
    completion.model_dump.return_value = {"id": "fake-completion"}
    return completion


def test_rag_retrieves_matching_properties_and_generates_grounded_answer(
    monkeypatch, property_chunks
):
    requirements = {
        "location": "New Cairo",
        "bedrooms": 3,
        "max_price": 5_000_000,
    }
    metadata_filter = build_metadata_filter(requirements)

    # Simulate Qdrant applying the deterministic metadata constraints. The
    # semantically closer but unsuitable B205 must not reach generation.
    def fake_qdrant_search(query, filt, collection_name, qdrant_client, top_k):
        assert query == "3 bedroom apartment in New Cairo under 5 million"
        assert collection_name == "properties"
        return [chunk for chunk in property_chunks if matches(chunk.metadata, filt)][:top_k]

    monkeypatch.setattr(retrieval, "_qdrant_search", fake_qdrant_search)

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(
        'وجدت لك الوحدة A102 في القاهرة الجديدة بسعر 4,800,000 جنيه.\n'
        '{"cited_property_ids": ["A102"]}'
    )
    monkeypatch.setattr(generation, "_get_client", lambda: fake_client)

    query = "3 bedroom apartment in New Cairo under 5 million"
    candidates = retrieval.hybrid_search(
        query=query,
        requirements_filter=metadata_filter,
        collection_name="properties",
        qdrant_client=MagicMock(),
        top_k=5,
    )
    result = generation.generate_response(query, candidates)

    assert [candidate.property_id for candidate in candidates] == ["A102"]
    assert result.cited_property_ids == ["A102"]
    assert result.grounding_property_ids == ["A102"]
    _, call_kwargs = fake_client.chat.completions.create.call_args
    assert "A102" in call_kwargs["messages"][0]["content"]
    assert "B205" not in call_kwargs["messages"][0]["content"]


def test_rag_handles_no_matching_properties_without_citing_any(monkeypatch):
    monkeypatch.setattr(retrieval, "_qdrant_search", lambda *args, **kwargs: [])
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(
        'لم أجد عقارات مطابقة. هل تود توسيع نطاق البحث؟\n'
        '{"cited_property_ids": []}'
    )
    monkeypatch.setattr(generation, "_get_client", lambda: fake_client)

    result_candidates = retrieval.hybrid_search(
        query="villa in Mars",
        requirements_filter=build_metadata_filter({"location": "Mars"}),
        collection_name="properties",
        qdrant_client=MagicMock(),
    )
    result = generation.generate_response("villa in Mars", result_candidates)

    assert result_candidates == []
    assert result.cited_property_ids == []
    assert result.grounding_property_ids == []
    # Nothing was retrieved, so the model is not asked to talk about properties at all.
    fake_client.chat.completions.create.assert_not_called()
    assert "couldn't find any listings" in result.answer_text
