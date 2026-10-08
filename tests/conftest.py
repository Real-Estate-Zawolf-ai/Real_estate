"""Shared fixtures.

``stub_rag`` replaces the RAG workflow (Qdrant + embeddings + Groq) with a
recording fake, so agent, API, voice and WhatsApp tests exercise the real
LangGraph orchestration offline. These are mocked-integration tests: they do
not prove the external providers work.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.workflows.rag as rag
from app.agents.service import ConversationService


LISTINGS = [
    {
        "property_id": "501189949",
        "chunk_id": "c-501189949",
        "text": "Apartment for sale in New Cairo. Price: EGP 7,941,000. 3 bedrooms.",
        "metadata": {"property_type": "Apartment", "district": "New Cairo", "price": 7_941_000, "bedrooms": 3},
        "source_file": "test_properties.csv",
        "score": 0.91,
    },
    {
        "property_id": "501189810",
        "chunk_id": "c-501189810",
        "text": "Apartment for sale in New Cairo. Price: EGP 25,000,000. 4 bedrooms.",
        "metadata": {"property_type": "Apartment", "district": "New Cairo", "price": 25_000_000, "bedrooms": 4},
        "source_file": "test_properties.csv",
        "score": 0.84,
    },
]


class RagStub:
    def __init__(self) -> None:
        self.searches: list[dict] = []
        self.follow_ups: list[dict] = []
        self.fail_with: Exception | None = None
        self.results = [dict(item) for item in LISTINGS]

    def answer_query(self, query, requirements=None, **kwargs):
        if self.fail_with:
            raise self.fail_with
        self.searches.append({"query": query, "requirements": dict(requirements or {}), "history": kwargs.get("conversation_history") or []})
        ids = [item["property_id"] for item in self.results]
        return {
            "answer": f"I found {len(ids)} matching properties: " + ", ".join(ids) if ids else "No matching properties were found.",
            "cited_property_ids": ids[:1],
            "grounding_property_ids": ids,
            "properties": [dict(item) for item in self.results],
        }

    def answer_about_properties(self, query, properties, requirements=None, **kwargs):
        if self.fail_with:
            raise self.fail_with
        ids = [item["property_id"] for item in properties]
        self.follow_ups.append({"query": query, "property_ids": ids})
        return {
            "answer": f"Details for {ids[0]} from the listing evidence.",
            "cited_property_ids": ids[:1],
            "grounding_property_ids": ids,
            "properties": [dict(item) for item in properties],
        }


@pytest.fixture(autouse=True)
def fresh_booking_calendar():
    """Each test starts with no viewing bookings (the store is process-global)."""
    from app.agents.tools.scheduling import scheduling_service

    scheduling_service.__init__()
    yield


@pytest.fixture
def stub_rag(monkeypatch) -> RagStub:
    stub = RagStub()
    monkeypatch.setattr(rag, "answer_query", stub.answer_query)
    monkeypatch.setattr(rag, "answer_about_properties", stub.answer_about_properties)
    return stub


@pytest.fixture
def service(stub_rag) -> ConversationService:
    """A fresh conversation service with its own in-memory checkpointer."""
    return ConversationService()


@pytest.fixture
def thread_id() -> str:
    return f"web:{uuid4().hex}"


# ---------------------------------------------------------------------------
# Fake Gemini Live session
# ---------------------------------------------------------------------------


def live_audio(data: bytes) -> SimpleNamespace:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data))
    return SimpleNamespace(tool_call=None, go_away=None, server_content=SimpleNamespace(
        model_turn=SimpleNamespace(parts=[part]), interrupted=False, turn_complete=False,
        input_transcription=None, output_transcription=None))


def live_transcript(role: str, text: str) -> SimpleNamespace:
    field = "input_transcription" if role == "user" else "output_transcription"
    content = {"model_turn": None, "interrupted": False, "turn_complete": False,
               "input_transcription": None, "output_transcription": None}
    content[field] = SimpleNamespace(text=text)
    return SimpleNamespace(tool_call=None, go_away=None, server_content=SimpleNamespace(**content))


def live_tool_call(message: str, call_id: str = "call-1") -> SimpleNamespace:
    call = SimpleNamespace(id=call_id, name="ask_aqar_agent", args={"message": message})
    return SimpleNamespace(tool_call=SimpleNamespace(function_calls=[call]), go_away=None, server_content=None)


def live_event(**flags) -> SimpleNamespace:
    content = {"model_turn": None, "interrupted": False, "turn_complete": False,
               "input_transcription": None, "output_transcription": None, **flags}
    return SimpleNamespace(tool_call=None, go_away=None, server_content=SimpleNamespace(**content))


class FakeLiveSession:
    """Releases one scripted batch of server messages per client input.

    When the script runs out, ``receive()`` yields nothing — exactly what the
    google-genai SDK does once the provider connection has closed.
    """

    def __init__(self, turns: list[list[SimpleNamespace]]) -> None:
        self.turns = list(turns)
        self.audio: list[bytes] = []
        self.stream_ends = 0
        self.tool_responses: list = []
        self.system_instruction: str | None = None
        self._input = asyncio.Event()

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        if audio is not None:
            self.audio.append(audio.data)
        if audio_stream_end:
            self.stream_ends += 1
        self._input.set()

    async def send_tool_response(self, function_responses):
        self.tool_responses.extend(function_responses)
        self._input.set()

    async def receive(self):
        await self._input.wait()
        self._input.clear()
        if not self.turns:
            return
        for message in self.turns.pop(0):
            yield message


@pytest.fixture
def fake_live(monkeypatch):
    """Patch the voice bridge to talk to a scripted FakeLiveSession."""
    import app.channels.website as website

    holder = SimpleNamespace(session=None)

    def install(turns):
        session = FakeLiveSession(turns)
        holder.session = session

        @asynccontextmanager
        async def connect(system_instruction=None, **_):
            session.system_instruction = system_instruction
            yield session

        monkeypatch.setattr(website, "connect_live_session", connect)
        return session

    holder.install = install
    return holder
