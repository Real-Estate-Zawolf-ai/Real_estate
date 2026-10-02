from app.agents.state import ConversationState
from app.leads.qualification import missing_qualification_fields
from app.leads.scoring import calculate_lead_score


class QualificationState(
    ConversationState,
    total=False,
):

    qualification: dict

    missing_fields: list[str]

    qualification_complete: bool

def analyze_lead(
    state: QualificationState,
) -> dict:
    qualification = state.get("qualification", {})
    missing = missing_qualification_fields(qualification)

    return {
        "missing_fields": missing,
        "qualification_complete": not missing,
        "lead_score": calculate_lead_score(qualification),
    }


def qualify_lead(
    state: QualificationState,
) -> dict:
    missing = state.get("missing_fields", [])
    if missing:
        return {"response": f"What is your {missing[0]}?"}
    return {"qualification_complete": True, "response": "Your requirements are complete."}


def prepare_crm(state: QualificationState) -> dict:
    """Expose a serializable lead summary for the CRM/handoff layer."""
    return {
        "qualification": {
            **state.get("qualification", {}),
            "score": state.get("lead_score", 0),
            "complete": state.get("qualification_complete", False),
        }
    }


# Backwards-compatible names used by earlier callers.
check_qualification = analyze_lead
ask_question = qualify_lead
