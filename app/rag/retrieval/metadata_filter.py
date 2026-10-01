"""Build deterministic hard filters from customer requirements.

This module translates structured customer requirements into a
vector-database-agnostic ``MetadataFilter`` representation.

Hard constraints such as price, area, bedrooms, location,
transaction type, and availability should be enforced deterministically
rather than relying on semantic similarity alone.

The resulting filter can be translated by the retrieval layer into the
native filtering syntax of Qdrant, pgvector, or another backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@dataclass
class RangeConstraint:
    """A numeric range constraint, such as ``price <= 5_000_000``."""

    field: str
    gte: float | None = None
    lte: float | None = None


@dataclass
class EqualityConstraint:
    """An exact-match constraint, such as ``location == 'New Cairo'``."""

    field: str
    value: Any


@dataclass
class MetadataFilter:
    """Backend-agnostic representation of hard metadata constraints."""

    equality: list[EqualityConstraint] = field(default_factory=list)
    ranges: list[RangeConstraint] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True when no filtering constraints are defined."""

        return not self.equality and not self.ranges

    def describe(self) -> str:
        """Return a human-readable representation for logging/tracing."""

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
#   payment_plan
#   availability
#   finishing
#   delivery
#
# ---------------------------------------------------------------------------

REQUIREMENT_TO_METADATA_FIELD: dict[
    str,
    str | tuple[str, str],
] = {
    # Location
    "location": "location",
    "city": "city",
    "district": "district",
    "neighborhood": "neighborhood",
    "compound": "compound",

    # Property classification
    "property_type": "property_type",
    "transaction_type": "transaction_type",
    "purpose": "transaction_type",
    "developer": "developer",

    # Property attributes
    "finishing": "finishing",
    "delivery": "delivery",
    "availability": "availability",

    # Financial / payment
    "currency": "currency",
    "payment_type": "payment_plan",
    "payment_plan": "payment_plan",

    # Exact numeric values
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",

    # Price ranges
    "min_price": ("price", "gte"),
    "max_price": ("price", "lte"),

    # Area ranges
    "min_area": ("area", "gte"),
    "max_area": ("area", "lte"),
    "min_area_sqm": ("area", "gte"),
    "max_area_sqm": ("area", "lte"),

    # Bedroom ranges
    "min_bedrooms": ("bedrooms", "gte"),
    "max_bedrooms": ("bedrooms", "lte"),

    # Bathroom ranges
    "min_bathrooms": ("bathrooms", "gte"),
    "max_bathrooms": ("bathrooms", "lte"),
}


def build_metadata_filter(
    requirements: dict[str, Any],
    always_available_only: bool = False,
) -> MetadataFilter:
    """Convert structured customer requirements into hard filters.

    Example:
        {
            "transaction_type": "sale",
            "location": "New Cairo",
            "property_type": "apartment",
            "min_bedrooms": 3,
            "max_price": 5_000_000,
        }

    ``None`` values and unknown requirement fields are ignored.

    Args:
        requirements:
            Structured requirements extracted from the customer request.

        always_available_only:
            When True, add the explicit constraint
            ``availability == "available"``.

            This remains disabled by default because the current property
            dataset does not consistently guarantee a canonical availability
            value.

    Returns:
        A backend-agnostic ``MetadataFilter``.
    """

    if not isinstance(requirements, dict):
        raise TypeError("requirements must be a dictionary")

    equality: list[EqualityConstraint] = []
    ranges_by_field: dict[str, RangeConstraint] = {}

    for req_field, value in requirements.items():
        if value is None:
            continue

        mapping = REQUIREMENT_TO_METADATA_FIELD.get(req_field)

        if mapping is None:
            # Unknown requirements are intentionally ignored.
            # Retrieval should not fail because an extractor produced
            # an unsupported field.
            continue

        if isinstance(mapping, tuple):
            metadata_field, bound = mapping

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Requirement '{req_field}' must be numeric, "
                    f"got {value!r}"
                ) from exc

            constraint = ranges_by_field.setdefault(
                metadata_field,
                RangeConstraint(field=metadata_field),
            )

            setattr(constraint, bound, numeric_value)
            continue

        equality.append(
            EqualityConstraint(
                field=mapping,
                value=value,
            )
        )

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


# ---------------------------------------------------------------------------
# Pure-Python evaluator
# ---------------------------------------------------------------------------


def matches(
    property_metadata: dict[str, Any],
    filt: MetadataFilter,
) -> bool:
    """Evaluate a MetadataFilter against one property's metadata.

    This function is useful for:

    - Unit testing filter behavior.
    - Development and in-memory retrieval.
    - Validating filters before backend integration.

    Production retrieval should push these constraints into the vector
    database's native filtering mechanism whenever possible.
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