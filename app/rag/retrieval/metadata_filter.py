"""
metadata_filter.py

Translates extracted customer requirements (structured, from the
conversational agent's requirement-extraction step) into explicit
filters over property metadata — the "hard constraints" layer.

Per the project's AI design principle: hard constraints (price,
bedrooms, location, purpose, availability) must be enforced by
deterministic filtering, never left to semantic similarity alone.
This keeps the system from recommending a 6M EGP apartment when the
customer said "max 5M".

This module is vector-DB-agnostic: it builds a `MetadataFilter` object
that `hybrid_search.py` translates into the specific query-DSL of
whichever backend is in use (Qdrant, pgvector, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RangeConstraint:
    """A numeric range constraint, e.g. price <= 5,000,000."""
    field: str
    gte: Optional[float] = None
    lte: Optional[float] = None


@dataclass
class EqualityConstraint:
    """An exact-match constraint, e.g. location == 'New Cairo'."""
    field: str
    value: Any


@dataclass
class MetadataFilter:
    """A backend-agnostic representation of the hard constraints to
    apply before/alongside semantic search.
    """
    equality: List[EqualityConstraint] = field(default_factory=list)
    ranges: List[RangeConstraint] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.equality and not self.ranges

    def describe(self) -> str:
        """Human-readable summary — useful for logging/tracing so you can
        see exactly what constraints were applied to a given query.
        """
        parts = [f"{c.field} == {c.value!r}" for c in self.equality]
        for r in self.ranges:
            if r.gte is not None and r.lte is not None:
                parts.append(f"{r.gte} <= {r.field} <= {r.lte}")
            elif r.gte is not None:
                parts.append(f"{r.field} >= {r.gte}")
            elif r.lte is not None:
                parts.append(f"{r.field} <= {r.lte}")
        return " AND ".join(parts) if parts else "(no constraints)"


# Maps requirement-extraction field names to property-metadata field
# names. Adjust if your extracted-requirement schema and property
# schema diverge (they're identical in the example data model, but
# real CRM data is rarely that clean).
REQUIREMENT_TO_METADATA_FIELD = {
    "location": "location",
    "property_type": "property_type",
    "purpose": "purpose",          # "sale" | "rent"
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "payment_type": "payment_plan",
    "max_price": ("price", "lte"),
    "min_price": ("price", "gte"),
    "min_area_sqm": ("area_sqm", "gte"),
    "max_area_sqm": ("area_sqm", "lte"),
}


def build_metadata_filter(
    requirements: Dict[str, Any],
    always_available_only: bool = True,
) -> MetadataFilter:
    """Convert extracted customer requirements into a MetadataFilter.

    `requirements` is the structured dict produced by the agent's
    requirement-extraction step, e.g.:
        {
            "purpose": "sale",
            "location": "New Cairo",
            "property_type": "apartment",
            "bedrooms": 3,
            "max_price": 5000000,
            "payment_type": "installment",
        }

    Missing/None fields are simply skipped — we only filter on what the
    customer actually specified.
    """
    equality: List[EqualityConstraint] = []
    ranges_by_field: Dict[str, RangeConstraint] = {}

    for req_field, value in requirements.items():
        if value is None:
            continue

        mapping = REQUIREMENT_TO_METADATA_FIELD.get(req_field)
        if mapping is None:
            continue  # unknown/unmapped requirement field — ignore, don't error

        if isinstance(mapping, tuple):
            meta_field, bound = mapping
            rc = ranges_by_field.setdefault(meta_field, RangeConstraint(field=meta_field))
            setattr(rc, bound, float(value))
        else:
            equality.append(EqualityConstraint(field=mapping, value=value))

    if always_available_only:
        equality.append(EqualityConstraint(field="availability", value="available"))

    return MetadataFilter(equality=equality, ranges=list(ranges_by_field.values()))


def matches(property_metadata: Dict[str, Any], filt: MetadataFilter) -> bool:
    """Pure-Python evaluation of a MetadataFilter against a single
    property's metadata dict. Useful for:
      - Unit testing filter logic without a real vector DB.
      - A fallback/dev-mode in-memory filter path.
    Production retrieval should push these constraints down into the
    vector DB's native filter syntax for efficiency (see hybrid_search.py).
    """
    for c in filt.equality:
        if property_metadata.get(c.field) != c.value:
            return False

    for r in filt.ranges:
        val = property_metadata.get(r.field)
        if val is None:
            return False
        if r.gte is not None and val < r.gte:
            return False
        if r.lte is not None and val > r.lte:
            return False

    return True