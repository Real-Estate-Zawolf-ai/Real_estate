"""Deterministic lead scoring for the real-estate sales workflow."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def calculate_lead_score(signals: Mapping[str, Any] | None) -> int:
    """Calculate a transparent 0–100 score from common qualification signals."""
    data = signals or {}
    score = 0

    if data.get("budget") not in (None, "", 0):
        score += 25
    if data.get("purpose") not in (None, ""):
        score += 15
    if data.get("timeline") not in (None, ""):
        score += 20
    if data.get("preferred_location") or data.get("location"):
        score += 10
    if data.get("property_type"):
        score += 10
    if data.get("financing_ready") is True:
        score += 10
    if data.get("requested_viewing") is True:
        score += 10

    return min(score, 100)


def score_band(score: int) -> str:
    """Map a numeric score to an operational priority band."""
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score >= 70:
        return "hot"
    if score >= 40:
        return "warm"
    return "cold"
