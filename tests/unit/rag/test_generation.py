"""
tests/unit/test_generation.py

Unit tests for app/rag/generation.py.

The Groq client is always mocked here — these tests verify prompt
construction, citation extraction, and the hallucination-detection
logging path without making any real API calls or requiring
GROQ_API_KEY to be set.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.rag import generation
from app.rag.retrieval.hybrid_search import SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def candidates():
    return [
        SearchResult(
            property_id="A102",
            chunk_id="c1",
            text="3BR apartment in New Cairo, installment available",
            metadata={
                "location": "New Cairo",
                "price": 4800000,
                "bedrooms": 3,
                "payment_plan": "installment",
                "availability": "available",
            },
            source_file="brochure_A102.pdf",
        ),
        SearchResult(
            property_id="B205",
            chunk_id="c2",
            text="3BR apartment in New Cairo, sea view",
            metadata={
                "location": "New Cairo",
                "price": 4950000,
                "bedrooms": 3,
                "payment_plan": "installment",
                "availability": "available",
            },
            source_file="brochure_B205.pdf",
        ),
    ]


def make_fake_completion(content: str):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    completion.model_dump = lambda: {"id": "fake-completion"}
    return completion


# ---------------------------------------------------------------------------
# _format_evidence_block / build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_evidence_block_includes_all_candidates(self, candidates):
        block = generation._format_evidence_block(candidates)
        assert "A102" in block
        assert "B205" in block
        assert "4800000" in block

    def test_evidence_block_empty_when_no_candidates(self):
        block = generation._format_evidence_block([])
        assert "no properties retrieved" in block

    def test_build_messages_structure(self, candidates):
        msgs = generation.build_messages("عايز شقة 3 غرف في التجمع", candidates)
        assert msgs[0]["role"] == "system"
        assert "A102" in msgs[0]["content"]
        assert msgs[-1] == {"role": "user", "content": "عايز شقة 3 غرف في التجمع"}

    def test_build_messages_includes_conversation_history(self, candidates):
        history = [
            {"role": "user", "content": "عايز شقة في التجمع"},
            {"role": "assistant", "content": "تمام، معاك ميزانية معينة؟"},
        ]
        msgs = generation.build_messages("حوالي 5 مليون", candidates, conversation_history=history)
        assert msgs[1:3] == history
        assert msgs[-1]["content"] == "حوالي 5 مليون"


# ---------------------------------------------------------------------------
# _extract_cited_ids
# ---------------------------------------------------------------------------

class TestExtractCitedIds:
    def test_extracts_valid_json_block(self):
        raw = 'لقيت لك عقارين: A102 و B205.\n{"cited_property_ids": ["A102", "B205"]}'
        answer, cited = generation._extract_cited_ids(raw)
        assert answer == "لقيت لك عقارين: A102 و B205."
        assert cited == ["A102", "B205"]

    def test_missing_json_block_returns_empty_list(self):
        raw = "رد عادي من غير JSON في الآخر"
        answer, cited = generation._extract_cited_ids(raw)
        assert answer == raw
        assert cited == []

    def test_malformed_json_falls_back_gracefully(self):
        raw = 'رد فيه جيسون تالف\n{"cited_property_ids": [A102]}'  # missing quotes -> invalid JSON
        answer, cited = generation._extract_cited_ids(raw)
        assert cited == []

    def test_empty_citation_list(self):
        raw = 'مفيش عقارات مطابقة للأسف.\n{"cited_property_ids": []}'
        answer, cited = generation._extract_cited_ids(raw)
        assert cited == []
        assert "مفيش عقارات" in answer


# ---------------------------------------------------------------------------
# generate_response (Groq client mocked)
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_generate_response_happy_path(self, candidates):
        fake_completion = make_fake_completion(
            'لقيت لك شقتين في التجمع: A102 و B205.\n{"cited_property_ids": ["A102", "B205"]}'
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        with patch.object(generation, "_get_client", return_value=fake_client):
            result = generation.generate_response("عايز شقة 3 غرف في التجمع", candidates)

        assert isinstance(result, generation.GenerationResult)
        assert "A102" in result.answer_text
        assert result.cited_property_ids == ["A102", "B205"]
        assert result.grounding_property_ids == ["A102", "B205"]
        assert result.model == generation.MODEL_NAME

        # Confirm the Groq call was made with the right model + our messages.
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["model"] == "openai/gpt-oss-120b"
        assert kwargs["messages"][-1]["content"] == "عايز شقة 3 غرف في التجمع"

    def test_generate_response_no_candidates(self):
        fake_completion = make_fake_completion(
            'للأسف مفيش عقارات مطابقة دلوقتي.\n{"cited_property_ids": []}'
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        with patch.object(generation, "_get_client", return_value=fake_client):
            result = generation.generate_response("عايز فيلا في المريخ بمليون جنيه", [])

        assert result.cited_property_ids == []
        assert result.grounding_property_ids == []

    def test_generate_response_logs_hallucinated_citation(self, candidates, caplog):
        fake_completion = make_fake_completion(
            'في عقار كويس ليك اسمه Z999.\n{"cited_property_ids": ["Z999"]}'
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        with patch.object(generation, "_get_client", return_value=fake_client):
            with caplog.at_level(logging.WARNING):
                result = generation.generate_response("test query", candidates)

        assert result.cited_property_ids == ["Z999"]
        assert result.grounding_property_ids == ["A102", "B205"]
        assert any("hallucination" in record.message for record in caplog.records)

    def test_generate_response_passes_temperature_and_max_tokens(self, candidates):
        fake_completion = make_fake_completion('رد.\n{"cited_property_ids": []}')
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_completion

        with patch.object(generation, "_get_client", return_value=fake_client):
            generation.generate_response(
                "test", candidates, temperature=0.5, max_tokens=200
            )

        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 200

    def test_client_never_constructed_until_needed(self, candidates):
        """Importing/using build_messages must never require GROQ_API_KEY —
        only generate_response (which calls _get_client) should."""
        # No patching, no env var set — should not raise.
        generation.build_messages("test", candidates)