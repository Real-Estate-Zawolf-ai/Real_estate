import pytest
from app.rag.retrieval.metadata_filter import (
    build_metadata_filter,
    matches,
    MetadataFilter,
    RangeConstraint,
    EqualityConstraint
)

def test_build_metadata_filter_basic():
    requirements = {
        "location": "New Cairo",
        "bedrooms": 3,
        "max_price": 5000000,
    }
    filt = build_metadata_filter(requirements, always_available_only=False)

    # Check equality
    eq_fields = [c.field for c in filt.equality]
    assert "location" in eq_fields
    assert "bedrooms" in eq_fields

    # Check ranges
    range_fields = [r.field for r in filt.ranges]
    assert "price" in range_fields
    assert filt.ranges[0].lte == 5000000

def test_build_metadata_filter_availability():
    requirements = {}
    filt = build_metadata_filter(requirements, always_available_only=True)

    # Should add availability == available by default
    assert any(c.field == "availability" and c.value == "available" for c in filt.equality)

def test_matches_success():
    property_metadata = {
        "location": "New Cairo",
        "bedrooms": 3,
        "price": 4000000,
        "availability": "available"
    }
    filt = MetadataFilter(
        equality=[EqualityConstraint(field="location", value="New Cairo")],
        ranges=[RangeConstraint(field="price", lte=5000000)]
    )
    assert matches(property_metadata, filt) is True

def test_matches_fail_equality():
    property_metadata = {
        "location": "Sheikh Zayed",
        "bedrooms": 3,
    }
    filt = MetadataFilter(
        equality=[EqualityConstraint(field="location", value="New Cairo")],
    )
    assert matches(property_metadata, filt) is False

def test_matches_fail_range():
    property_metadata = {
        "price": 6000000,
    }
    filt = MetadataFilter(
        ranges=[RangeConstraint(field="price", lte=5000000)],
    )
    assert matches(property_metadata, filt) is False

def test_matches_missing_field():
    property_metadata = {
        "location": "New Cairo",
    }
    filt = MetadataFilter(
        ranges=[RangeConstraint(field="price", lte=5000000)],
    )
    # If field is missing, it should not match
    assert matches(property_metadata, filt) is False
