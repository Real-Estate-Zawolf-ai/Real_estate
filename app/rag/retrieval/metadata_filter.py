"""
metadata_filter.py

Translates extracted customer requirements (structured, from the
conversational agent's requirement-extraction step) into explicit
filters over property metadata — the "hard constraints" layer.

Hard constraints such as price, bedrooms, location, transaction type,
and availability must be enforced by deterministic filtering rather
than semantic similarity alone.

This module is vector-DB-agnostic. It builds a MetadataFilter object
that hybrid_search.py can translate into the query syntax of the
selected backend (e.g. Qdrant or pgvector).
"""

from __future__ import annotations

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
    """
    Backend-agnostic representation of hard constraints to apply
    before or alongside semantic search.
    """

    equality: List[EqualityConstraint] = field(default_factory=list)
    ranges: List[RangeConstraint] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True when no filtering constraints are defined."""
        return not self.equality and not self.ranges

    def describe(self) -> str:
        """
        Return a human-readable representation of the constraints.

        Useful for logging and tracing.
        """
        parts = [
            f"{constraint.field} == {constraint.value!r}"
            for constraint in self.equality
        ]

        for constraint in self.ranges:
            if (
                constraint.gte is not None
                and constraint.lte is not None
            ):
                parts.append(
                    f"{constraint.gte} <= "
                    f"{constraint.field} <= "
                    f"{constraint.lte}"
                )
            elif constraint.gte is not None:
                parts.append(
                    f"{constraint.field} >= {constraint.gte}"
                )
            elif constraint.lte is not None:
                parts.append(
                    f"{constraint.field} <= {constraint.lte}"
                )

        return " AND ".join(parts) if parts else "(no constraints)"


# ---------------------------------------------------------------------------
# Requirement → canonical metadata mapping
# ---------------------------------------------------------------------------
#
# These canonical names must stay aligned with the normalization layer:
#
#   property_id
#   property_type
#   transaction_type
#   price
#   currency
#   area
#   area_unit
#   bedrooms
#   bathrooms
#   location
#   city
#   district
#   neighborhood
#   compound
#   developer
#   finishing
#   delivery
#   availability
#   ...
#
# Additional fields such as payment_plan can come from structured
# property/CRM data when available.

REQUIREMENT_TO_METADATA_FIELD = {
    "location": "location",
    "property_type": "property_type",
    "transaction_type": "transaction_type",
    "purpose": "transaction_type",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "payment_type": "payment_plan",
    "payment_plan": "payment_plan",
    "max_price": ("price", "lte"),
    "min_price": ("price", "gte"),
    "min_area": ("area", "gte"),
    "max_area": ("area", "lte"),
    "min_area_sqm": ("area", "gte"),
    "max_area_sqm": ("area", "lte"),
}


def build_metadata_filter(
    requirements: Dict[str, Any],
    always_available_only: bool = False,
) -> MetadataFilter:
    """
    Convert extracted customer requirements into a MetadataFilter.

    Example requirements:

        {
            "transaction_type": "sale",
            "location": "New Cairo",
            "property_type": "apartment",
            "bedrooms": 3,
            "max_price": 5000000,
        }

    Missing or None fields are skipped. Only requirements explicitly
    provided by the customer are converted into constraints.

    Args:
        requirements:
            Structured customer requirements.

        always_available_only:
            If True, only properties marked as "available" are allowed.

            This is disabled by default because the current property
            dataset does not guarantee that every property contains a
            canonical "availability" field.

    Returns:
        MetadataFilter containing equality and range constraints.
    """
    if not isinstance(requirements, dict):
        raise TypeError("requirements must be a dictionary")

    equality: List[EqualityConstraint] = []
    ranges_by_field: Dict[str, RangeConstraint] = {}

    for req_field, value in requirements.items():
        if value is None:
            continue

        mapping = REQUIREMENT_TO_METADATA_FIELD.get(req_field)

        if mapping is None:
            # Unknown requirement fields are ignored rather than
            # causing retrieval to fail.
            continue

        if isinstance(mapping, tuple):
            metadata_field, bound = mapping

            constraint = ranges_by_field.setdefault(
                metadata_field,
                RangeConstraint(field=metadata_field),
            )

            setattr(constraint, bound, float(value))

        else:
            equality.append(
                EqualityConstraint(
                    field=mapping,
                    value=value,
                )
            )

    # Only add this hard constraint when the caller explicitly enables it.
    #
    # The current dataset does not consistently provide:
    #
    #     availability = "available"
    #
    # Therefore this must not be enabled by default.
    if always_available_only:
        equality.append(
            EqualityConstraint(
                field="availability",
                value="available",
            )
        )

    return MetadataFilter(
        equality=equality,
        ranges=list(ranges_by_field.values()),
    )


def matches(
    property_metadata: Dict[str, Any],
    filt: MetadataFilter,
) -> bool:
    """
    Evaluate a MetadataFilter against a single property's metadata.

    This pure-Python implementation is useful for:

    - Unit testing filter logic.
    - Development/in-memory retrieval.
    - Verifying filter behavior before integrating with a vector DB.

    Production retrieval should push these constraints down into the
    vector database's native filtering mechanism.
    """
    if not isinstance(property_metadata, dict):
        return False

    if not isinstance(filt, MetadataFilter):
        raise TypeError("filt must be a MetadataFilter")

    # ------------------------------------------------------------------
    # Equality constraints
    # ------------------------------------------------------------------
    for constraint in filt.equality:
        if property_metadata.get(constraint.field) != constraint.value:
            return False

    # ------------------------------------------------------------------
    # Range constraints
    # ------------------------------------------------------------------
    for constraint in filt.ranges:
        value = property_metadata.get(constraint.field)

        if value is None:
            return False

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False

        if (
            constraint.gte is not None
            and numeric_value < constraint.gte
        ):
            return False

        if (
            constraint.lte is not None
            and numeric_value > constraint.lte
        ):
            return False

    return True
