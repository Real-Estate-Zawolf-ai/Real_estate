"""CRM behaviour shared by every channel, and the website contact form."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agents.tools.crm import crm_service
from app.crm import CRMClient, CreateCustomer, CreateInteraction, CreateLead, CustomerService, InteractionService, LeadService
from app.crm.customers import DuplicateCustomerError
from app.crm.interactions import InteractionChannel
from app.crm.leads import LeadNotFoundError
from app.database import models as db
from app.database.session import session_scope
from tests.integration.test_api_channels import chat, client  # noqa: F401  (fixtures)
from tests.integration.test_booking_flows import say


def lead_row(external_id):
    with session_scope() as session:
        customer = session.query(db.Customer).filter_by(external_id=external_id).one()
        lead = session.query(db.Lead).filter_by(customer_id=customer.id).order_by(db.Lead.id.desc()).first()
        return lead and {"status": lead.status, "score": lead.score, "budget": lead.budget_max, "location": lead.preferred_location,
                         "intent": lead.intent, "property": lead.property_ref, "source": lead.source, "meta": lead.metadata_json}


# ---------------------------------------------------------------------------
# Conversation -> CRM
# ---------------------------------------------------------------------------


def test_every_search_that_shares_requirements_becomes_a_scored_lead(service, thread_id, stub_rag):
    say(service, thread_id, "villa in Sheikh Zayed, my budget is 9 million, for investment, within 3 months")
    lead = lead_row(thread_id)
    assert lead["status"] == "qualifying" and lead["budget"] == 9_000_000 and lead["intent"] == "investment"
    assert lead["location"] == "Sheikh Zayed" and lead["source"] == "website" and lead["score"] >= 70
    assert lead["meta"]["qualification"]["band"] == "hot"
    assert lead["meta"]["requirements"]["district"] == "Sheikh Zayed"


def test_small_talk_does_not_create_a_lead(service, thread_id, stub_rag):
    say(service, thread_id, "hello")
    with session_scope() as session:
        assert session.query(db.Lead).count() == 0
        assert session.query(db.Interaction).count() == 1  # the conversation itself is still recorded


def test_the_selected_property_is_remembered_on_the_lead(service, thread_id, stub_rag):
    say(service, thread_id, "apartments in New Cairo")
    say(service, thread_id, "tell me more about the second one")
    assert lead_row(thread_id)["property"] == "501189810"


def test_lead_status_only_moves_forward(service, thread_id, stub_rag):
    crm_service.ensure_lead(thread_id, source="website")
    crm_service.update_lead(thread_id, {"status": "viewing_requested"})
    crm_service.update_lead(thread_id, {"status": "qualifying"})  # a later chat turn must not demote it
    crm_service.update_lead(thread_id, {"status": "new"})
    assert crm_service.get_lead(thread_id)["status"] == "viewing_requested"
    crm_service.update_lead(thread_id, {"status": "contacted"})
    assert crm_service.get_lead(thread_id)["status"] == "contacted"


def test_a_closed_lead_is_not_reused_for_a_returning_customer(service, thread_id, stub_rag):
    first = crm_service.ensure_lead(thread_id, source="website")
    crm_service.update_lead(thread_id, {"status": "won"})
    second = crm_service.ensure_lead(thread_id, source="website")
    assert second != first


def test_customers_on_different_channels_are_separate_records(service, stub_rag):
    web, wa = f"web:{uuid4().hex}", "whatsapp:201012345678"
    say(service, web, "villa in Sheikh Zayed under 9 million")
    say(service, wa, "apartment in New Cairo", channel="whatsapp")
    assert lead_row(web)["source"] == "website" and lead_row(wa)["source"] == "whatsapp"
    with session_scope() as session:
        assert session.query(db.Customer).count() == 2


def test_interactions_keep_the_channel_and_modality(service, thread_id, stub_rag):
    say(service, thread_id, "villa in Sheikh Zayed")
    say(service, thread_id, "tell me more", modality="voice")
    with session_scope() as session:
        channels = [row.channel for row in session.query(db.Interaction).order_by(db.Interaction.id)]
    assert channels == ["website:text", "website:voice"]


def test_a_crm_outage_never_breaks_the_conversation(service, thread_id, stub_rag, monkeypatch):
    import app.agents.service as agents

    def down(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(agents.crm_service, "sync_conversation", down)
    monkeypatch.setattr(agents.crm_service, "record_interaction", down)
    assert say(service, thread_id, "villa in Sheikh Zayed").properties  # the customer still gets their answer


def test_notes_are_timestamped_and_appended(service, thread_id, stub_rag):
    crm_service.ensure_lead(thread_id, source="website")
    crm_service.add_note(thread_id, "Prefers a call after 6 pm")
    crm_service.add_note(thread_id, "Has seen unit 501189949")
    notes = crm_service.get_lead(thread_id)["notes"]
    assert len(notes) == 2 and notes[0].startswith("[") and "Prefers a call after 6 pm" in notes[0]
    with pytest.raises(ValueError):
        crm_service.add_note(thread_id, "   ")
    with pytest.raises(KeyError):
        crm_service.add_note("web:unknown", "x")


# ---------------------------------------------------------------------------
# Typed CRM services (no longer in-memory)
# ---------------------------------------------------------------------------


def test_customers_leads_and_interactions_persist_through_the_typed_services():
    client_ = CRMClient()
    customer = CustomerService(client_).create(CreateCustomer(full_name="Nora Ali", email="NORA@example.com", phone_number="0101 234 5678"))
    assert customer.email == "nora@example.com"
    # A second CRMClient is a different connection: the data is in the database, not in the first object.
    assert CustomerService(CRMClient()).get(customer.id).full_name == "Nora Ali"

    lead = LeadService(client_).create(CreateLead(customer_id=customer.id, budget_max=5_000_000, property_id="501189949"))
    assert LeadService().update_status(lead.id, "qualified").status == "qualified"
    assert LeadService().update_score(lead.id, 80).score == 80
    with pytest.raises(ValueError):
        LeadService().update_score(lead.id, 101)

    InteractionService().record(CreateInteraction(customer_id=customer.id, lead_id=lead.id, channel=InteractionChannel.WHATSAPP, content="Hello"))
    InteractionService().record(CreateInteraction(customer_id=customer.id, channel=InteractionChannel.WEBSITE, content="Second"))
    assert [item.content for item in InteractionService().list_for_customer(customer.id)] == ["Hello", "Second"]


def test_duplicate_customers_and_dangling_references_are_rejected():
    service = CustomerService()
    service.create(CreateCustomer(full_name="A", email="a@example.com", phone_number="01012345678"))
    with pytest.raises(DuplicateCustomerError):
        service.create(CreateCustomer(full_name="B", email="A@EXAMPLE.COM"))
    with pytest.raises(DuplicateCustomerError):
        service.create(CreateCustomer(full_name="C", phone_number="+20 101 234 5678"))  # same number, different format
    with pytest.raises(ValueError):
        LeadService().create(CreateLead(customer_id=uuid4()))
    with pytest.raises(LeadNotFoundError):
        LeadService().update_status(uuid4(), "won")


# ---------------------------------------------------------------------------
# Website contact form
# ---------------------------------------------------------------------------


def form(client, **extra):
    body = {"name": "Nora", "phone": "0101 234 5678", "message": "Please call me", "property": "501189949", **extra}
    return client.post("/api/leads", json=body)


def test_contact_form_creates_a_lead_and_alerts_staff(client):
    response = form(client)
    assert response.status_code == 201 and response.json()["status"] == "created"
    with session_scope() as session:
        lead = session.query(db.Lead).one()
        customer = session.query(db.Customer).one()
        assert (lead.status, lead.source, lead.property_ref) == ("contact_requested", "website_form", "501189949")
        assert (customer.name, customer.phone, customer.phone_verified) == ("Nora", "+201012345678", False)  # a form does not verify
        alert = session.query(db.StaffNotification).one()
        assert "+201012345678" in alert.message and "not verified" in alert.message and alert.kind == "lead_form"


def test_contact_form_attaches_to_the_chat_customer(client, stub_rag):
    session_id = chat(client, "villa in Sheikh Zayed, my budget is 9 million")["session_id"]
    assert form(client, session_id=session_id).status_code == 201
    with session_scope() as session:
        assert session.query(db.Customer).count() == 1  # the same person, not a second record
        lead = session.query(db.Lead).one()
        assert lead.status == "contact_requested" and lead.budget_max == 9_000_000


@pytest.mark.parametrize("phone", ["12345", "not a number", "0101", "+1"])
def test_contact_form_refuses_numbers_an_advisor_could_not_call(client, phone):
    response = form(client, phone=phone)
    detail = response.json()["detail"]
    # Very short input is rejected by the request schema, the rest by phone validation.
    assert response.status_code == 422 and (isinstance(detail, list) or "valid mobile number" in detail)
    with session_scope() as session:
        assert session.query(db.Lead).count() == 0


def test_contact_form_is_rate_limited(client, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_LEADS_PER_HOUR="2")
    assert [form(client).status_code for _ in range(4)] == [201, 201, 429, 429]


def test_a_forged_session_id_on_the_form_does_not_attach_to_anyone(client, stub_rag):
    victim = chat(client, "villa in Sheikh Zayed, my budget is 9 million")["session_id"]
    raw, stamp, sig = victim.split(".")
    assert form(client, session_id=f"{raw}.{stamp}.{'0' * len(sig)}").status_code == 201
    with session_scope() as session:
        assert session.query(db.Customer).count() == 2  # the form became its own customer; the victim's record is untouched
