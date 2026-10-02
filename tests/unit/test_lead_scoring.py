import pytest

from app.leads.qualification import is_qualified, missing_qualification_fields
from app.leads.scoring import calculate_lead_score, score_band


def test_complete_high_intent_lead_scores_hot():
    lead = {
        "budget": 5_000_000,
        "purpose": "buy",
        "timeline": "this month",
        "location": "New Cairo",
        "property_type": "apartment",
        "financing_ready": True,
        "requested_viewing": True,
    }
    assert calculate_lead_score(lead) == 100
    assert score_band(calculate_lead_score(lead)) == "hot"
    assert is_qualified(lead)


def test_missing_qualification_fields_are_reported():
    assert missing_qualification_fields({"budget": 1_000_000}) == ["purpose", "timeline"]
    assert calculate_lead_score({}) == 0


def test_score_band_rejects_invalid_score():
    with pytest.raises(ValueError):
        score_band(101)
