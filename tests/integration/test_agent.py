from app.agents.subgraphs.conversational.graph import generate_response
from app.agents.state import route_agent, router
from app.agents.subgraphs.qualification.state import analyze_lead
from app.schemas.interaction import ChatRequest, ChatResponse


def test_router_selects_property_agent():
    routed = router({"query": "I need a 3 bedroom apartment"})
    assert routed["intent"] == "property_search"
    assert route_agent(routed) == "property"
    assert router({"query": "عاوز عقارات في الشيخ زايد"})["intent"] == "property_search"


def test_router_distinguishes_acknowledgement_and_unrelated_question_from_property_context():
    history = [{"role": "user", "content": "I want apartments in Sheikh Zayed"}]

    assert router({"query": "تمام", "messages": history})["intent"] == "conversational"
    assert router({"query": "شكراً", "messages": history})["intent"] == "conversational"
    assert router({"query": "إيه الأخبار؟", "messages": history})["intent"] == "conversational"
    assert router({"query": "What's the weather?", "messages": history})["intent"] == "conversational"


def test_router_routes_property_refinements_only_with_recent_property_context():
    history = [{"role": "user", "content": "I want apartments in Sheikh Zayed"}]

    bedrooms = router({"query": "طب عايز 3 غرف", "messages": history})
    budget = router({"query": "under 10 million", "messages": history})
    arabic_budget = router({"query": "أقل من 10 مليون", "messages": history})
    assert bedrooms["intent"] == budget["intent"] == "property_search"
    assert arabic_budget["intent"] == "property_search"
    assert route_agent(bedrooms) == route_agent(budget) == "property"

    assert router({"query": "under 10 million"})["intent"] == "qualification"


def test_chat_schema_accepts_history_and_keeps_response_shape():
    request = ChatRequest(query="تمام", history=[{"role": "user", "content": "apartments"}])
    assert request.history[0]["content"] == "apartments"

    response = ChatResponse(answer="Here are options", response_type="property")
    assert response.model_dump().keys() == {
        "answer", "response_type", "cited_property_ids", "grounding_property_ids", "properties"
    }


def test_conversational_agent_uses_configured_groq_model_and_recent_history(monkeypatch):
    from app.agents.subgraphs.conversational import graph as conversational

    captured = {}

    class FakeLLMClient:
        model = "configured-groq-model"

        def generate_messages(self, messages, **kwargs):
            captured["model"] = self.model
            captured["messages"] = messages
            captured.update(kwargs)
            return "تمام، أنا معاك."

    monkeypatch.setattr(conversational, "_get_llm_client", lambda: FakeLLMClient())

    result = generate_response({
        "query": "تمام",
        "messages": [
            {"role": "user", "content": "عاوز عقارات في الشيخ زايد"},
            {"role": "assistant", "content": "لقيت لك اختيارات."},
        ],
    })

    assert result["response"] == "تمام، أنا معاك."
    assert captured["model"] == "configured-groq-model"
    assert captured["messages"][-3:] == [
        {"role": "user", "content": "عاوز عقارات في الشيخ زايد"},
        {"role": "assistant", "content": "لقيت لك اختيارات."},
        {"role": "user", "content": "تمام"},
    ]


def test_qualification_analysis_is_deterministic():
    result = analyze_lead(
        {"qualification": {"budget": 5_000_000, "purpose": "buy", "timeline": "now"}}
    )
    assert result["qualification_complete"] is True
    assert result["lead_score"] >= 60
