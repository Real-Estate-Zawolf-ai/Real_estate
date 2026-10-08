"""Lead-capture endpoint used by the public website."""

from fastapi import APIRouter, status

from app.agents.tools.crm import crm_service
from app.api.sessions import verify_session_id, website_thread
from app.schemas.lead import LeadCaptureRequest


router = APIRouter(prefix="/api/leads", tags=["leads"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def capture_lead(request: LeadCaptureRequest) -> dict[str, str]:
    customer = {
        "name": request.name.strip(),
        "phone": request.phone.strip(),
        "message": request.message.strip(),
        "property_id": request.property,
    }
    # When the visitor has chatted with the assistant, attach the form to the
    # same customer so advisors see the requirements collected in the chat.
    raw_session = verify_session_id(request.session_id)
    if raw_session:
        customer_id = website_thread(raw_session)
        lead_id = crm_service.ensure_lead(customer_id, source="website_form")
        crm_service.upsert_customer(customer_id, **customer)
        crm_service.update_lead(customer_id, {"status": "contact_requested", "form_message": customer["message"]})
        return {"id": lead_id, "status": "created"}

    lead_id = crm_service.create_lead(customer)
    return {"id": lead_id, "status": "created"}
