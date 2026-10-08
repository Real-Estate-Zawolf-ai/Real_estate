from app.agents.messages import say
from app.agents.state import ConversationState, describe_requirements
from app.agents.tools.crm import crm_service
from app.leads.qualification import missing_qualification_fields
from app.leads.scoring import calculate_lead_score


# Order in which missing details are asked for, one question per turn.
QUESTION_ORDER = (
    ("property_type", "ask_type"),
    ("preferred_location", "ask_location"),
    ("budget", "ask_budget"),
    ("purpose", "ask_purpose"),
    ("timeline", "ask_timeline"),
    ("payment_method", "ask_payment_method"),
)


class QualificationState(
    ConversationState,
    total=False,
):

    qualification: dict

    missing_fields: list[str]

    qualification_complete: bool

    lead_score: int

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
    language = state.get("language", "en")
    qualification = state.get("qualification") or {}
    summary = describe_requirements(state.get("requirements") or {}, qualification, language)

    for field, phrase in QUESTION_ORDER:
        if qualification.get(field) in (None, ""):
            question = say(language, phrase)
            response = f"{say(language, 'noted', summary=summary)} {question}" if summary else question
            return {"response": response, "active_agent": "qualification"}

    return {
        "qualification_complete": True,
        "response": say(language, "qualification_complete", summary=summary),
        "active_agent": "qualification",
    }


def prepare_crm(state: QualificationState) -> dict:
    """Expose a serializable lead summary and sync it to the CRM."""
    qualification = {
        **state.get("qualification", {}),
        "score": state.get("lead_score", 0),
        "complete": state.get("qualification_complete", False),
    }
    update: dict = {"qualification": qualification}

    customer_id = state.get("customer_id")
    if customer_id:
        lead_id = crm_service.ensure_lead(customer_id, source=state.get("channel", "website"))
        crm_service.upsert_customer(customer_id, **(state.get("customer") or {}))
        crm_service.update_lead(customer_id, {"qualification": qualification, "status": "qualifying"})
        update["lead_id"] = lead_id
    return update


# Backwards-compatible names used by earlier callers.
check_qualification = analyze_lead
ask_question = qualify_lead
