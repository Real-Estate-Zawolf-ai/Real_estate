from app.agents.tools.property import PropertySearchService


def _service() -> PropertySearchService:
    service = PropertySearchService()
    service.replace_all(
        [
            {"property_id": "P1", "text": "Apartment in New Cairo", "price": 4_000_000},
            {"property_id": "P2", "text": "Villa in Zayed", "price": 8_000_000},
        ]
    )
    return service


def test_property_search_uses_query_terms():
    assert [item["property_id"] for item in _service().search("New Cairo", {})] == ["P1"]


def test_property_search_applies_range_filter():
    assert [item["property_id"] for item in _service().search("", {"max_price": 5_000_000})] == ["P1"]


def test_property_catalogue_returns_copies():
    service = _service()
    records = service.list_all()
    records[0]["property_id"] = "changed"
    assert service.list_all()[0]["property_id"] == "P1"
