"""Opt-in tests against the REAL providers.

They call live APIs with real credentials, so they never run by default::

    RUN_LIVE_TESTS=1 pytest -m live -v

Credentials come from the environment or ``.env``. Each test skips (and says
why) when the credential it needs is absent, so the report shows exactly which
integrations were and were not exercised:

* ``GEMINI_API_KEY``                      embeddings + Gemini Live
* ``QDRANT_URL`` (+ ``QDRANT_API_KEY``)   vector search (a local Qdrant works too)
* ``GROQ_API_KEY``                        grounded generation + reranking
* ``WHATSAPP_ACCESS_TOKEN`` + ``WHATSAPP_PHONE_NUMBER_ID``   Meta token check (read-only)

Tests that index data use a throw-away collection and delete it afterwards. None
of them sends a WhatsApp message or touches an existing collection.
"""

from __future__ import annotations

import asyncio
import math
import os
from uuid import uuid4

import pytest

from tests.live_env import LIVE_ENV

pytestmark = pytest.mark.live

LISTINGS = [
    {"property_id": "L-1", "text": "Apartment for sale in New Cairo. 3 bedrooms, 2 bathrooms, 185 sqm. Price: EGP 4,850,000. Clubhouse and parking.",
     "metadata": {"property_type": "Apartment", "district": "New Cairo", "bedrooms": 3, "bathrooms": 2, "price": 4_850_000, "area": 185}},
    {"property_id": "L-2", "text": "Villa for sale in Sheikh Zayed. 5 bedrooms, private garden, 420 sqm. Price: EGP 12,800,000.",
     "metadata": {"property_type": "Villa", "district": "Sheikh Zayed", "bedrooms": 5, "bathrooms": 4, "price": 12_800_000, "area": 420}},
    {"property_id": "L-3", "text": "شقة للبيع في التجمع الخامس، 3 غرف نوم، 160 متر. السعر 5,200,000 جنيه. قريبة من الخدمات.",
     "metadata": {"property_type": "Apartment", "district": "New Cairo", "bedrooms": 3, "bathrooms": 2, "price": 5_200_000, "area": 160}},
    {"property_id": "L-4", "text": "Chalet for sale in North Coast with sea view, 3 bedrooms and beach access. Price: EGP 6,950,000.",
     "metadata": {"property_type": "Chalet", "district": "North Coast", "bedrooms": 3, "bathrooms": 3, "price": 6_950_000, "area": 160}},
    {"property_id": "L-5", "text": "Studio for rent in Maadi, fully furnished, 55 sqm. Rent: EGP 18,000 per month.",
     "metadata": {"property_type": "Studio", "district": "Maadi", "bedrooms": 1, "bathrooms": 1, "price": 18_000, "area": 55}},
    {"property_id": "L-6", "text": "فيلا للبيع في الشيخ زايد، 4 غرف نوم مع حديقة خاصة. السعر 9,900,000 جنيه.",
     "metadata": {"property_type": "Villa", "district": "Sheikh Zayed", "bedrooms": 4, "bathrooms": 3, "price": 9_900_000, "area": 330}},
]


@pytest.fixture(autouse=True)
def live_enabled():
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: set RUN_LIVE_TESTS=1")


def need(*names: str) -> dict[str, str]:
    missing = [name for name in names if not LIVE_ENV.get(name)]
    if missing:
        pytest.skip("not verified - missing credential(s): " + ", ".join(missing))
    return {name: LIVE_ENV[name] for name in names}


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture
def live_settings(settings_env):
    """Apply the real credentials this process captured at start-up."""
    keep = ("GEMINI_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "GEMINI_LIVE_MODEL", "GEMINI_VOICE",
            "EMBEDDING_DIMENSION", "EMBEDDING_MODEL", "GROQ_MODEL", "WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID")
    settings_env(**{key: LIVE_ENV[key] for key in keep if LIVE_ENV.get(key)})
    return LIVE_ENV


# ---------------------------------------------------------------------------
# Gemini embeddings
# ---------------------------------------------------------------------------


def test_gemini_embeddings_are_multilingual(live_settings):
    need("GEMINI_API_KEY")
    from app.rag.embeddings import EmbeddingModel

    model = EmbeddingModel()
    arabic = model.embed_query("شقة في التجمع الخامس")
    english = model.embed_query("apartment in New Cairo")
    unrelated = model.embed_query("recipe for chocolate cake")
    assert len(arabic) == model.dimension
    assert cosine(arabic, english) > cosine(arabic, unrelated) + 0.05  # cross-language meaning, not keywords


# ---------------------------------------------------------------------------
# Qdrant + embeddings + retrieval
# ---------------------------------------------------------------------------


@pytest.fixture
def scratch_collection(live_settings, settings_env):
    need("GEMINI_API_KEY", "QDRANT_URL")
    from app.database.vector_db import create_client

    name = f"aqar_live_{uuid4().hex[:8]}"
    settings_env(QDRANT_COLLECTION=name)
    yield name
    client = create_client()
    try:
        client.delete_collection(name)
    finally:
        client.close()


def test_index_search_roundtrip_with_filters_in_two_languages(scratch_collection):
    import app.workflows.rag as rag

    chunks = [{"id": f"live-{item['property_id']}", "text": item["text"], "metadata": {"property_id": item["property_id"], **item["metadata"]}}
              for item in LISTINGS]
    assert rag.index_chunks(chunks) == len(LISTINGS)

    english = rag.search_index("3 bedroom apartment in New Cairo", {"district": "New Cairo", "property_type": "Apartment"}, top_k=5)
    assert {item.property_id for item in english} <= {"L-1", "L-3"}  # the hard filters are honoured
    assert english

    arabic = rag.search_index("فيلا في الشيخ زايد", {"district": "Sheikh Zayed"}, top_k=5)
    assert {item.property_id for item in arabic} == {"L-2", "L-6"}

    budget = rag.search_index("villa", {"max_price": 10_000_000}, top_k=5)
    assert all(item.metadata.get("price", 0) <= 10_000_000 for item in budget)
    assert "L-2" not in {item.property_id for item in budget}  # EGP 12.8M is over budget

    nothing = rag.search_index("anything", {"district": "Maadi", "property_type": "Villa"}, top_k=5)
    assert nothing == []  # a filter that matches nothing returns nothing, never a near miss


def test_search_refuses_a_collection_of_the_wrong_vector_size(scratch_collection, settings_env):
    import app.workflows.rag as rag
    from app.database.vector_db import VectorDimensionMismatch, create_client, models

    client = create_client()
    client.create_collection(scratch_collection, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    client.close()
    with pytest.raises(VectorDimensionMismatch):
        rag.index_chunks([{"id": "x", "text": "a listing", "metadata": {}}])


def test_grounded_answer_cites_only_retrieved_listings(scratch_collection):
    need("GROQ_API_KEY")
    import app.workflows.rag as rag

    rag.index_chunks([{"id": f"live-{i['property_id']}", "text": i["text"], "metadata": {"property_id": i["property_id"], **i["metadata"]}} for i in LISTINGS])
    result = rag.answer_query("Do you have a villa in Sheikh Zayed and what does it cost?", {"district": "Sheikh Zayed"}, top_k=3)
    known = {item["property_id"] for item in result["properties"]}
    assert set(result["cited_property_ids"]) <= known  # no invented listing ids
    assert "12,800,000" in result["answer"] or "9,900,000" in result["answer"] or "12.8" in result["answer"]


# ---------------------------------------------------------------------------
# Gemini Live
# ---------------------------------------------------------------------------


def test_gemini_live_accepts_our_config_and_calls_the_agent_tool(live_settings):
    need("GEMINI_API_KEY")
    from app.voice.gemini_live import AGENT_TOOL_NAME, build_voice_instruction, connect_live_session

    async def run() -> tuple[bool, list[str]]:
        tool_called, transcripts = False, []
        async with connect_live_session(build_voice_instruction("ar")) as session:
            await session.send_realtime_input(text="I am looking for an apartment in New Cairo. What do you have?")
            async with asyncio.timeout(40):
                async for message in session.receive():
                    call = getattr(message, "tool_call", None)
                    if call is not None:
                        names = [item.name for item in call.function_calls or []]
                        tool_called = AGENT_TOOL_NAME in names
                        return tool_called, transcripts
                    content = getattr(message, "server_content", None)
                    text = getattr(getattr(content, "output_transcription", None), "text", None)
                    if text:
                        transcripts.append(text)
                    if content is not None and getattr(content, "turn_complete", False):
                        break
        return tool_called, transcripts

    called, transcripts = asyncio.run(run())
    assert called, f"model answered from its own knowledge instead of calling the tool: {''.join(transcripts)[:200]}"


# ---------------------------------------------------------------------------
# Meta WhatsApp (read-only)
# ---------------------------------------------------------------------------


def test_meta_whatsapp_token_and_phone_number_id_are_valid(live_settings):
    need("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID")
    import httpx

    response = httpx.get(
        f"https://graph.facebook.com/{LIVE_ENV.get('WHATSAPP_API_VERSION', 'v23.0')}/{LIVE_ENV['WHATSAPP_PHONE_NUMBER_ID']}",
        params={"fields": "display_phone_number,verified_name"},
        headers={"Authorization": f"Bearer {LIVE_ENV['WHATSAPP_ACCESS_TOKEN']}"},
        timeout=20,
    )
    assert response.status_code == 200, f"Meta rejected the credentials (HTTP {response.status_code})"
    assert response.json().get("display_phone_number")


def _live_exchange(language: str, customer_text: str, reply: str | None = None) -> tuple[str | None, str]:
    """Send one text turn to the real Live model.

    Returns (the ``message`` it put in the tool call, what it said after receiving
    ``reply`` as the tool result).
    """
    from google.genai import types

    from app.voice.gemini_live import AGENT_TOOL_NAME, build_voice_instruction, connect_live_session

    async def run() -> tuple[str | None, str]:
        tool_message, spoken = None, []
        async with connect_live_session(build_voice_instruction(language)) as session:
            await session.send_realtime_input(text=customer_text)
            async with asyncio.timeout(60):
                while True:
                    async for message in session.receive():
                        call = getattr(message, "tool_call", None)
                        if call is not None:
                            for item in call.function_calls or []:
                                if item.name == AGENT_TOOL_NAME:
                                    tool_message = str((item.args or {}).get("message", ""))
                                    if reply is None:
                                        return tool_message, ""
                                    await session.send_tool_response(function_responses=[
                                        types.FunctionResponse(id=item.id, name=item.name, response={"reply": reply})
                                    ])
                        content = getattr(message, "server_content", None)
                        text = getattr(getattr(content, "output_transcription", None), "text", None)
                        if text:
                            spoken.append(text)
                        if content is not None and getattr(content, "turn_complete", False) and (tool_message is None or spoken):
                            return tool_message, "".join(spoken)

    return asyncio.run(run())


def test_gemini_live_arabic_conversation_stays_in_arabic(live_settings):
    need("GEMINI_API_KEY")
    import re

    tool_message, spoken = _live_exchange(
        "ar", "عايز شقة في التجمع الخامس ميزانيتي خمسة مليون",
        reply="لقيت لك شقة في التجمع الخامس، تلات غرف نوم، سعرها 5,200,000 جنيه.",
    )
    assert tool_message and re.search(r"[؀-ۿ]", tool_message), f"tool message was not forwarded in Arabic: {tool_message!r}"
    assert re.search(r"[؀-ۿ]", spoken), f"the spoken reply was not Arabic: {spoken!r}"


@pytest.mark.parametrize("utterance", [
    "رقم تليفوني صفر واحد صفر واحد اتنين تلاتة اربعة خمسة ستة سبعة تمانية",
    "my phone number is zero one zero, one two three, four five six seven eight",
])
def test_phone_number_dictated_to_the_live_model_reaches_the_agent_intact(live_settings, utterance):
    """Digits spoken in words must come out of the model and through our parser as one valid number."""
    need("GEMINI_API_KEY")
    from app.agents.understanding import extract_qualification
    from app.helpers.phone import normalize_phone

    tool_message, _ = _live_exchange("ar" if utterance[0] == "ر" else "en", utterance)
    assert tool_message, "the model did not forward the phone number to the agent"
    captured = extract_qualification(tool_message).get("phone")
    assert normalize_phone(captured) == "+201012345678", f"captured {captured!r} from tool message {tool_message!r}"
