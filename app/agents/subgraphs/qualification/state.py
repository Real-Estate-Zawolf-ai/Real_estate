from typing_extensions import TypedDict

from app.agents.state import ConversationState


class QualificationState(
    ConversationState,
    total=False,
):

    qualification: dict

    missing_fields: list[str]

    qualification_complete: bool

def check_qualification(
    state: QualificationState,
) -> dict:

    qualification = state.get(
        "qualification",
        {},
    )

    required = [
        "budget",
        "purpose",
        "timeline",
    ]

    missing = [
        field
        for field in required
        if field not in qualification
    ]

    return {
        "missing_fields": missing,
        "qualification_complete": not missing,
    }

def ask_question(
    state: QualificationState,
) -> dict:

    missing = state["missing_fields"]

    question = f"What is your {missing[0]}?"

    return {
        "response": question
    }
