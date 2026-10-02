from app.agents.state import route_agent, router
from app.agents.subgraphs.qualification.state import analyze_lead


def test_router_selects_property_agent():
    routed = router({"query": "I need a 3 bedroom apartment"})
    assert routed["intent"] == "property_search"
    assert route_agent(routed) == "property"


def test_qualification_analysis_is_deterministic():
    result = analyze_lead(
        {"qualification": {"budget": 5_000_000, "purpose": "buy", "timeline": "now"}}
    )
    assert result["qualification_complete"] is True
    assert result["lead_score"] >= 60
