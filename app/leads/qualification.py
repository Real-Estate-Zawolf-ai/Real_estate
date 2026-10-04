"""Lead qualification helpers shared by agents and API services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_QUALIFICATION_FIELDS = ("budget", "purpose", "timeline")


def missing_qualification_fields(
    qualification: Mapping[str, Any] | None,
) -> list[str]:
    """Return required fields that have no meaningful value."""
    values = qualification or {}
    return [
        field
        for field in REQUIRED_QUALIFICATION_FIELDS
        if values.get(field) is None or values.get(field) == ""
    ]


def is_qualified(qualification: Mapping[str, Any] | None) -> bool:
    """Return whether the minimum sales-qualification data is present."""
    return not missing_qualification_fields(qualification)
