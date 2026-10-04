"""Lead-capture endpoint used by the public website."""

from fastapi import APIRouter, status

from app.agents.tools.crm import crm_service
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
    lead_id = crm_service.create_lead(customer)
    return {"id": lead_id, "status": "created"}
