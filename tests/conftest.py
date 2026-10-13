"""Shared fixtures.

``stub_rag`` replaces the RAG workflow (Qdrant + embeddings + Groq) with a
recording fake, so agent, API, voice and WhatsApp tests exercise the real
LangGraph orchestration offline. These are mocked-integration tests: they do
not prove the external providers work.

The suite is hermetic: it never reads the developer's ``.env`` and runs against
a throw-away SQLite database through the same SQL code as production. Tests that
need real PostgreSQL are marked ``postgres`` and run only when
``TEST_DATABASE_URL`` points at one.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

# --- environment, before any `app` module reads settings ----------------------
# Remember the real credentials (process env + .env) for the opt-in `live` tests
# before the hermetic defaults below wipe them from this process.
from tests.live_env import LIVE_ENV  # noqa: E402,F401

from app.helpers.config import Settings, reset_settings_cache  # noqa: E402

Settings.model_config["env_file"] = None  # never load the real .env in tests

_TMP = Path(tempfile.mkdtemp(prefix="aqar-tests-"))
_TEST_ENV = {
    "APP_ENV": "test",
    "DATABASE_URL": f"sqlite:///{(_TMP / 'test.sqlite3').as_posix()}",
    "SESSION_SECRET": "test-session-secret-0123456789abcdef0123456789abcdef",
    "ADMIN_API_KEY": "test-admin-key-0123456789abcdef",
    "RATE_LIMIT_ENABLED": "false",
    "CORS_ALLOWED_ORIGINS": "*",
    "LOG_LEVEL": "WARNING",
}
for _key, _value in _TEST_ENV.items():
    os.environ[_key] = _value
for _key in (
    "POSTGRES_HOST", "GROQ_API_KEY", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
    "WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET", "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_GATEWAY_SECRET", "STAFF_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
):
    os.environ.pop(_key, None)
reset_settings_cache()

import app.workflows.rag as rag  # noqa: E402
from app.agents.service import ConversationService  # noqa: E402


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
def offline_conversational_llm(monkeypatch):
    """Small talk uses the deterministic fallback reply unless a test opts in."""
    from app.agents.subgraphs.conversational import graph as conversational

    def unavailable():
        raise RuntimeError("Conversational LLM is disabled in tests")

    monkeypatch.setattr(conversational, "_get_llm_client", unavailable)


@pytest.fixture(autouse=True)
def fresh_database():
    """Every test starts with an empty database and clean process-wide state."""
    from app.api.ratelimit import reset_limits
    from app.api.sessions import reset_secret_cache
    from app.database.session import create_all, drop_all, reset_engine

    from app.api.routes.properties import reset_catalogue_cache

    reset_settings_cache()
    reset_secret_cache()
    reset_limits()
    reset_catalogue_cache()
    reset_engine()
    drop_all()
    create_all()
    yield
    reset_limits()


@pytest.fixture
def settings_env(monkeypatch):
    """Change settings for one test: ``settings_env(RATE_LIMIT_ENABLED="true")``."""

    def apply(**values: str) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))
        reset_settings_cache()

    yield apply
    reset_settings_cache()


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


def live_go_away(seconds: str = "5s") -> SimpleNamespace:
    """The provider announces it will end this connection soon."""
    return SimpleNamespace(tool_call=None, go_away=SimpleNamespace(time_left=seconds), server_content=None)


def live_tool_cancellation(*call_ids: str) -> SimpleNamespace:
    """The customer interrupted: these pending tool calls are no longer wanted."""
    return SimpleNamespace(
        tool_call=None, go_away=None, server_content=None,
        tool_call_cancellation=SimpleNamespace(ids=list(call_ids)),
    )


@pytest.fixture
def fake_live(monkeypatch):
    """Patch the voice bridge to talk to scripted FakeLiveSessions.

    ``install(turns)`` serves one session for every connection;
    ``install_sequence(a, b, ...)`` serves a different script per connection
    (a connection beyond the list gets an empty, immediately-closing session).
    """
    import app.channels.website as website

    holder = SimpleNamespace(session=None, sessions=[], connections=0, instructions=[])

    def patch(factory):
        @asynccontextmanager
        async def connect(system_instruction=None, **_):
            holder.connections += 1
            holder.instructions.append(system_instruction)
            session = factory(holder.connections)
            session.system_instruction = system_instruction
            holder.session = session
            holder.sessions.append(session)
            yield session

        monkeypatch.setattr(website, "connect_live_session", connect)

    def install(turns):
        session = FakeLiveSession(turns)
        patch(lambda _n: session)
        holder.session = session
        return session

    def install_sequence(*scripts):
        patch(lambda n: FakeLiveSession(scripts[n - 1] if n <= len(scripts) else []))
        return holder

    holder.install = install
    holder.install_sequence = install_sequence
    return holder
