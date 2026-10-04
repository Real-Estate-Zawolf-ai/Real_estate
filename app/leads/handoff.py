"""Rules for escalating a conversation to a human sales representative."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.leads.scoring import calculate_lead_score


@dataclass(frozen=True)
class HandoffDecision:
    required: bool
    reason: str | None = None
    priority: str = "normal"


def evaluate_handoff(context: Mapping[str, Any] | None) -> HandoffDecision:
    """Return a handoff decision for explicit requests, risk, or high intent."""
    data = context or {}
    if data.get("requested_human"):
        return HandoffDecision(True, "customer_requested_human", "high")
    if data.get("complaint") or data.get("legal_or_financial_risk"):
        return HandoffDecision(True, "sensitive_request", "high")
    if data.get("requested_viewing") and calculate_lead_score(data) >= 70:
        return HandoffDecision(True, "high_intent_lead", "high")
    return HandoffDecision(False)
