from unittest.mock import MagicMock

from app.rag.generation import GenerationResult
from app.rag.retrieval.hybrid_search import SearchResult
from app.workflows import rag


def test_index_chunks_embeds_and_upserts(monkeypatch):
    client = MagicMock()
    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.0] * 384]
    monkeypatch.setattr(rag, "create_client", lambda: client)
    monkeypatch.setattr(rag, "EmbeddingModel", lambda: embedder)
    monkeypatch.setattr(rag, "ensure_collection", MagicMock())
    monkeypatch.setattr(rag, "upsert_chunks", lambda client, name, chunks, vectors: len(chunks))

    count = rag.index_chunks([{"id": "c1", "text": "property text", "metadata": {}}])

    assert count == 1
    embedder.embed_texts.assert_called_once_with(["property text"])
    client.close.assert_called_once()


def test_answer_query_connects_search_and_generation(monkeypatch):
    candidate = SearchResult("P1", "C1", "Apartment", {}, combined_score=0.9)
    monkeypatch.setattr(rag, "search_index", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(
        rag,
        "generate_response",
        lambda **kwargs: GenerationResult(
            answer_text="Property P1 matches.",
            cited_property_ids=["P1"],
            grounding_property_ids=["P1"],
        ),
    )

    result = rag.answer_query("find an apartment")

    assert result["answer"] == "Property P1 matches."
    assert result["properties"][0]["property_id"] == "P1"
