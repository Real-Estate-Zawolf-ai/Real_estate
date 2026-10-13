"""Operator (staff) API: bookings, handoffs, leads and alert delivery.

Every route needs the ``X-Admin-Key`` header. This is what lets an advisor turn
a customer's viewing *request* into a confirmed appointment, claim and resolve
human-handoff tickets, and see the alerts the system tried to send.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.agents.tools.crm import crm_service
from app.agents.tools.scheduling import scheduling_service
from app.api.ratelimit import client_ip
from app.api.security import audit, require_admin
from app.crm.handoffs import HandoffNotFoundError, HandoffService, InvalidHandoffTransition
from app.database import models as db
from app.database.session import session_scope
from app.helpers.config import get_settings
from app.notifications.staff import retry_pending
from app.scheduling.booking import BookingNotFoundError, InvalidBookingTransition


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

_handoffs = HandoffService()


class StaffAction(BaseModel):
    by: str = Field(default="staff", min_length=1, max_length=100)
    reason: str | None = Field(default=None, max_length=255)


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, (BookingNotFoundError, HandoffNotFoundError, LookupError)):
        return HTTPException(status_code=404, detail="Not found")
    if isinstance(error, (InvalidBookingTransition, InvalidHandoffTransition)):
        return HTTPException(status_code=409, detail=str(error))
    raise error


# --- bookings -----------------------------------------------------------------


@router.get("/bookings")
async def list_bookings(
    status: str | None = Query(default=None, pattern="^(pending_confirmation|confirmed|cancelled|expired|completed)$"),
    upcoming: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return await run_in_threadpool(scheduling_service.staff_list, status=status, upcoming_only=upcoming, limit=limit)


@router.get("/bookings/{booking_id}")
async def get_booking(booking_id: UUID) -> dict[str, Any]:
    booking = await run_in_threadpool(scheduling_service.staff_get, booking_id)
    if booking is None:
        raise HTTPException(status_code=404, detail="Not found")
    return booking


@router.post("/bookings/{booking_id}/confirm")
async def confirm_booking(booking_id: UUID, request: Request, action: StaffAction | None = None) -> dict[str, Any]:
    action = action or StaffAction()
    try:
        booking = await run_in_threadpool(scheduling_service.staff_confirm, booking_id, by=action.by)
    except (BookingNotFoundError, InvalidBookingTransition) as error:
        raise _translate(error) from error
    audit(request, "booking.confirm", str(booking_id), {"by": action.by})
    return booking


@router.post("/bookings/{booking_id}/cancel")
async def cancel_booking(booking_id: UUID, request: Request, action: StaffAction | None = None) -> dict[str, Any]:
    action = action or StaffAction()
    try:
        booking = await run_in_threadpool(scheduling_service.staff_cancel, booking_id, by=action.by, reason=action.reason)
    except (BookingNotFoundError, InvalidBookingTransition) as error:
        raise _translate(error) from error
    audit(request, "booking.cancel", str(booking_id), {"by": action.by, "reason": action.reason})
    return booking


@router.post("/bookings/{booking_id}/complete")
async def complete_booking(booking_id: UUID, request: Request) -> dict[str, Any]:
    try:
        booking = await run_in_threadpool(scheduling_service.staff_complete, booking_id)
    except (BookingNotFoundError, InvalidBookingTransition) as error:
        raise _translate(error) from error
    audit(request, "booking.complete", str(booking_id))
    return booking


# --- handoffs -----------------------------------------------------------------


@router.get("/handoffs")
async def list_handoffs(
    status: str | None = Query(default=None, pattern="^(open|claimed|resolved)$"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    return await run_in_threadpool(_handoffs.list, status, limit=limit)


@router.post("/handoffs/{handoff_id}/claim")
async def claim_handoff(handoff_id: UUID, request: Request, action: StaffAction | None = None) -> dict[str, Any]:
    action = action or StaffAction()
    try:
        handoff = await run_in_threadpool(_handoffs.claim, handoff_id, action.by)
    except (HandoffNotFoundError, InvalidHandoffTransition) as error:
        raise _translate(error) from error
    audit(request, "handoff.claim", str(handoff_id), {"by": action.by})
    return handoff


@router.post("/handoffs/{handoff_id}/resolve")
async def resolve_handoff(handoff_id: UUID, request: Request, action: StaffAction | None = None) -> dict[str, Any]:
    action = action or StaffAction()
    try:
        handoff = await run_in_threadpool(_handoffs.resolve, handoff_id, action.by)
    except (HandoffNotFoundError, InvalidHandoffTransition) as error:
        raise _translate(error) from error
    audit(request, "handoff.resolve", str(handoff_id), {"by": action.by})
    return handoff


# --- leads --------------------------------------------------------------------


def _list_leads(status: str | None, limit: int) -> list[dict[str, Any]]:
    with session_scope() as session:
        query = (
            select(db.Lead, db.Customer)
            .join(db.Customer, db.Customer.id == db.Lead.customer_id)
            .order_by(db.Lead.updated_at.desc(), db.Lead.id.desc())
            .limit(limit)
        )
        if status:
            query = query.where(db.Lead.status == status)
        return [
            {
                "id": str(lead.public_id),
                "status": lead.status,
                "score": lead.score,
                "source": lead.source,
                "intent": lead.intent,
                "budget_max": float(lead.budget_max) if lead.budget_max is not None else None,
                "preferred_location": lead.preferred_location,
                "property_type": lead.preferred_property_type,
                "property_ref": lead.property_ref,
                "customer": {
                    "ref": str(customer.public_id),
                    "name": customer.name,
                    "phone": customer.phone,
                    "phone_verified": customer.phone_verified,
                    "email": customer.email,
                    "channel": customer.source,
                },
                "updated_at": lead.updated_at.isoformat() if lead.updated_at else None,
            }
            for lead, customer in session.execute(query).all()
        ]


@router.get("/leads")
async def list_leads(status: str | None = Query(default=None, max_length=50), limit: int = Query(default=100, ge=1, le=500)):
    return await run_in_threadpool(_list_leads, status, limit)


# --- alerts -------------------------------------------------------------------


def _list_notifications(status: str | None, limit: int) -> list[dict[str, Any]]:
    with session_scope() as session:
        query = select(db.StaffNotification).order_by(db.StaffNotification.id.desc()).limit(limit)
        if status:
            query = query.where(db.StaffNotification.status == status)
        return [
            {
                "id": row.id,
                "kind": row.kind,
                "status": row.status,
                "attempts": row.attempts,
                "last_error": row.last_error,
                "message": row.message,
                "ref_type": row.ref_type,
                "ref_id": row.ref_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "sent_at": row.sent_at.isoformat() if row.sent_at else None,
            }
            for row in session.scalars(query)
        ]


@router.get("/notifications")
async def list_notifications(status: str | None = Query(default=None, max_length=20), limit: int = Query(default=50, ge=1, le=200)):
    return await run_in_threadpool(_list_notifications, status, limit)


@router.post("/notifications/retry")
async def retry_notifications(request: Request) -> dict[str, Any]:
    delivered = await run_in_threadpool(retry_pending)
    audit(request, "notifications.retry", detail={"delivered": delivered})
    return {"delivered": delivered, "channels": get_settings().staff_channels}


# --- diagnostics ----------------------------------------------------------------


@router.get("/diagnostics/client-ip")
async def diagnose_client_ip(request: Request) -> dict[str, Any]:
    """Show how the rate limiter identifies *this* caller.

    After deploying behind a proxy, call this from a known address and check
    ``resolved`` matches it; if it shows the proxy's address, raise
    TRUSTED_PROXY_HOPS or fix the proxy chain.
    """
    return {
        "resolved": client_ip(request),
        "peer": request.client.host if request.client else None,
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "trusted_proxy_hops": get_settings().TRUSTED_PROXY_HOPS,
    }


__all__ = ["router", "crm_service"]
