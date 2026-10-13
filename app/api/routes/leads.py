"""Lead-capture endpoint used by the public website."""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.agents.tools.crm import crm_service
from app.api.ratelimit import limit_leads
from app.api.sessions import verify_session_id, website_thread
from app.helpers.phone import normalize_phone
from app.notifications.staff import notify_staff
from app.schemas.lead import LeadCaptureRequest


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/leads", tags=["leads"])


def _capture(request: LeadCaptureRequest, phone: str) -> str:
    name = request.name.strip()
    message = request.message.strip()
    raw_session = verify_session_id(request.session_id)
    # When the visitor has chatted with the assistant, attach the form to the
    # same customer so advisors see the requirements collected in the chat.
    if raw_session:
        customer_id = website_thread(raw_session)
        lead_id = crm_service.ensure_lead(customer_id, source="website_form")
        crm_service.upsert_customer(customer_id, name=name, phone=phone)
        crm_service.update_lead(
            customer_id,
            {"status": "contact_requested", "form_message": message[:2000], "property_ref": request.property},
        )
    else:
        lead_id = crm_service.create_lead(
            {"name": name, "phone": phone, "message": message, "property_id": request.property}
        )
    notify_staff(
        "lead_form",
        "New contact request from the website\n"
        f"Name: {name}\nPhone: {phone} (not verified)\n"
        f"Property: {request.property or '-'}\nMessage: {message[:300] or '-'}\nLead: {lead_id[:8]}",
        ref_type="lead",
        ref_id=lead_id,
    )
    return lead_id


@router.post("", status_code=status.HTTP_201_CREATED)
async def capture_lead(request: LeadCaptureRequest, http_request: Request) -> dict[str, str]:
    limit_leads(http_request)
    phone = normalize_phone(request.phone)
    if phone is None:
        raise HTTPException(status_code=422, detail="Please enter a valid mobile number, e.g. 01012345678")
    try:
        lead_id = await run_in_threadpool(_capture, request, phone)
    except Exception as error:
        logger.exception("Lead capture failed")
        raise HTTPException(status_code=503, detail="Could not save your request right now") from error
    return {"id": lead_id, "status": "created"}
