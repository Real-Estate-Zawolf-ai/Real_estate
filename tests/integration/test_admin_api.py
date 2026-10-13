"""The staff API: turning requests into confirmed viewings, handling hand-offs, auditing."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from app.database import models as db
from app.database.session import session_scope
from tests.integration.test_api_channels import client  # noqa: F401  (fixture)
from tests.integration.test_booking_flows import bookings, say

KEY = {"X-Admin-Key": "test-admin-key-0123456789abcdef"}


@pytest.fixture
def pending(service, stub_rag):
    """One viewing request waiting for an advisor (WhatsApp: the sender is pre-verified)."""
    thread = "whatsapp:201012345678"
    for text in ("apartments in New Cairo", "book a viewing of the first one tomorrow", "the first", "yes"):
        say(service, thread, text, channel="whatsapp")
    return bookings()[0]["id"]


def audit_actions():
    with session_scope() as session:
        return [row.action for row in session.query(db.AdminAudit).order_by(db.AdminAudit.id)]


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/admin/bookings", "/admin/handoffs", "/admin/leads", "/admin/notifications", "/admin/diagnostics/client-ip"])
def test_every_admin_route_rejects_missing_and_wrong_keys(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.get(path, headers={"X-Admin-Key": "test-admin-key-0123456789abcdef "}).status_code == 401  # near miss
    assert client.get(path, headers=KEY).status_code == 200


def test_admin_is_disabled_entirely_when_no_key_is_configured(client, settings_env):
    settings_env(ADMIN_API_KEY="")
    assert client.get("/admin/bookings", headers=KEY).status_code == 503
    assert client.post("/properties", json={"property_id": "x"}, headers=KEY).status_code == 503


def test_admin_post_routes_are_protected_too(client, pending):
    for path in (f"/admin/bookings/{pending}/confirm", f"/admin/bookings/{pending}/cancel", f"/admin/bookings/{pending}/complete",
                 f"/admin/handoffs/{uuid4()}/claim", f"/admin/handoffs/{uuid4()}/resolve", "/admin/notifications/retry"):
        assert client.post(path).status_code == 401, path
    assert bookings()[0]["status"] == "pending_confirmation"


def test_repeated_wrong_keys_lock_the_client_out(client, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_ADMIN_FAILURES_PER_15MIN="4")
    codes = [client.get("/admin/bookings", headers={"X-Admin-Key": f"guess-{i}"}).status_code for i in range(6)]
    assert codes == [401, 401, 401, 401, 429, 429]
    assert client.get("/admin/bookings", headers=KEY).status_code == 429  # even the right key, while locked out


def test_path_parameters_are_validated(client):
    assert client.get("/admin/bookings/not-a-uuid", headers=KEY).status_code == 422
    assert client.get(f"/admin/bookings/{uuid4()}", headers=KEY).status_code == 404
    assert client.get("/admin/bookings?status=bogus", headers=KEY).status_code == 422
    assert client.get("/admin/bookings?limit=100000", headers=KEY).status_code == 422


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------


def test_staff_see_pending_requests_with_the_contact_they_need(client, pending):
    (booking,) = client.get("/admin/bookings", headers=KEY).json()
    assert booking["status"] == "pending_confirmation" and booking["contact_phone"] == "+201012345678"
    assert booking["property_id"] == "501189949" and booking["hold_expires_at"]
    assert client.get("/admin/bookings?status=confirmed", headers=KEY).json() == []
    assert client.get(f"/admin/bookings/{pending}", headers=KEY).json()["id"] == pending


def test_confirming_a_booking_tells_the_whatsapp_customer(client, pending, monkeypatch, settings_env):
    sent = []
    settings_env(WHATSAPP_ACCESS_TOKEN="token", WHATSAPP_PHONE_NUMBER_ID="123")
    monkeypatch.setattr(httpx, "post", lambda url, **kw: sent.append(kw["json"]) or httpx.Response(200, request=httpx.Request("POST", url)))

    response = client.post(f"/admin/bookings/{pending}/confirm", headers=KEY, json={"by": "Sara"})
    assert response.status_code == 200
    assert (response.json()["status"], response.json()["confirmed_by"]) == ("confirmed", "Sara")
    assert "is confirmed" in sent[0]["text"]["body"] and sent[0]["to"] == "201012345678"

    again = client.post(f"/admin/bookings/{pending}/confirm", headers=KEY)  # idempotent
    assert again.status_code == 200 and again.json()["status"] == "confirmed"
    assert audit_actions() == ["booking.confirm", "booking.confirm"]


def test_staff_can_cancel_with_a_reason_and_the_slot_reopens(client, pending, service):
    response = client.post(f"/admin/bookings/{pending}/cancel", headers=KEY, json={"by": "Sara", "reason": "advisor sick"})
    assert (response.status_code, response.json()["status"], response.json()["cancelled_by"]) == (200, "cancelled", "staff")
    assert response.json()["cancel_reason"] == "advisor sick"
    # The same slot can now be booked by another customer.
    other = "whatsapp:201155555555"
    for text in ("apartments in New Cairo", "book a viewing of the first one tomorrow", "the first", "yes"):
        result = say(service, other, text, channel="whatsapp")
    assert result.appointment is not None


def test_booking_state_machine_is_enforced(client, pending):
    assert client.post(f"/admin/bookings/{pending}/complete", headers=KEY).status_code == 409  # not confirmed yet
    client.post(f"/admin/bookings/{pending}/confirm", headers=KEY)
    assert client.post(f"/admin/bookings/{pending}/complete", headers=KEY).json()["status"] == "completed"
    assert client.post(f"/admin/bookings/{pending}/cancel", headers=KEY).status_code == 409  # already held
    assert client.post(f"/admin/bookings/{pending}/confirm", headers=KEY).status_code == 409
    assert client.post(f"/admin/bookings/{uuid4()}/confirm", headers=KEY).status_code == 404


def test_an_expired_hold_cannot_be_confirmed(client, pending):
    with session_scope() as session:
        from datetime import datetime, timedelta, timezone

        session.query(db.Booking).update({"hold_expires_at": datetime.now(timezone.utc) - timedelta(hours=1)})
    response = client.post(f"/admin/bookings/{pending}/confirm", headers=KEY)
    assert response.status_code == 409 and "expired" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Handoffs
# ---------------------------------------------------------------------------


@pytest.fixture
def handoff(service, stub_rag):
    thread = f"web:{uuid4().hex}"
    say(service, thread, "apartment in New Cairo, budget 30 million, for investment")
    say(service, thread, "I want to talk to a sales agent")
    say(service, thread, "01012345678")
    return thread


def test_a_handoff_ticket_carries_the_context_an_advisor_needs(client, handoff):
    (ticket,) = client.get("/admin/handoffs", headers=KEY).json()
    assert ticket["status"] == "open" and ticket["priority"] == "high" and ticket["customer_id"] == handoff
    assert ticket["summary"]["requirements"]["district"] == "New Cairo"
    assert ticket["summary"]["qualification"]["purpose"] == "investment"
    assert ticket["summary"]["contact"]["phone"] == "+201012345678"
    assert ticket["summary"]["contact"]["phone_verified"] is False  # given in chat, not confirmed by read-back
    assert ticket["summary"]["lead_score"] > 0


def test_handoff_lifecycle_claim_then_resolve(client, handoff):
    ticket_id = client.get("/admin/handoffs", headers=KEY).json()[0]["id"]
    claimed = client.post(f"/admin/handoffs/{ticket_id}/claim", headers=KEY, json={"by": "Omar"}).json()
    assert (claimed["status"], claimed["claimed_by"]) == ("claimed", "Omar")
    assert client.get("/admin/handoffs?status=open", headers=KEY).json() == []

    resolved = client.post(f"/admin/handoffs/{ticket_id}/resolve", headers=KEY, json={"by": "Omar"}).json()
    assert resolved["status"] == "resolved" and resolved["resolved_at"]
    assert client.post(f"/admin/handoffs/{ticket_id}/resolve", headers=KEY).status_code == 409
    assert client.post(f"/admin/handoffs/{ticket_id}/claim", headers=KEY).status_code == 409
    assert client.post(f"/admin/handoffs/{uuid4()}/claim", headers=KEY).status_code == 404
    assert audit_actions() == ["handoff.claim", "handoff.resolve"]


def test_asking_twice_does_not_create_two_tickets_or_two_alerts(service, handoff):
    say(service, handoff, "please connect me to a human")
    say(service, handoff, "I want to talk to a sales agent")
    from app.agents.tools.crm import crm_service

    assert len(crm_service.list_handoffs()) == 1
    with session_scope() as session:
        assert session.query(db.StaffNotification).filter_by(kind="handoff_requested").count() == 1


def test_a_handoff_without_a_number_waits_for_one(service, thread_id, stub_rag):
    asked = say(service, thread_id, "I want to talk to a sales agent")
    assert asked.handoff == {"status": "awaiting_contact"}
    refused = say(service, thread_id, "my number is 12345")
    assert refused.handoff == {"status": "awaiting_contact"} or refused.handoff is None or refused.handoff.get("status") != "requested"
    assert say(service, thread_id, "01012345678").handoff["status"] == "requested"


# ---------------------------------------------------------------------------
# Leads and alerts
# ---------------------------------------------------------------------------


def test_leads_listing_shows_score_budget_and_contact(client, handoff):
    (lead,) = client.get("/admin/leads", headers=KEY).json()
    assert lead["customer"]["phone"] == "+201012345678" and lead["customer"]["channel"] == "web"
    assert lead["budget_max"] == 30_000_000 and lead["preferred_location"] == "New Cairo" and lead["score"] > 0
    assert client.get("/admin/leads?status=nonexistent", headers=KEY).json() == []


def test_alert_outbox_is_visible_and_retryable(client, pending):
    alerts = client.get("/admin/notifications", headers=KEY).json()
    assert [a["kind"] for a in alerts] == ["booking_requested"] and alerts[0]["status"] == "no_channel"
    retry = client.post("/admin/notifications/retry", headers=KEY).json()
    assert retry == {"delivered": 0, "channels": []}
    assert audit_actions() == ["notifications.retry"]


def test_client_identification_diagnostic(client, settings_env):
    settings_env(TRUSTED_PROXY_HOPS="1")
    body = client.get("/admin/diagnostics/client-ip", headers={**KEY, "X-Forwarded-For": "1.1.1.1, 203.0.113.9"}).json()
    assert body["resolved"] == "203.0.113.9" and body["trusted_proxy_hops"] == 1
