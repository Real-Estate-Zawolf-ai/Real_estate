"""End-to-end agent journeys through the real LangGraph graph.

RAG (Qdrant/embeddings/Groq) is stubbed by ``stub_rag``; everything else —
routing, state, checkpointing, scheduling, CRM and handoff — is real.
"""

from datetime import datetime
from uuid import uuid4

from app.agents.tools.crm import crm_service


def turn(service, thread_id, text, **kwargs):
    kwargs.setdefault("channel", "website")
    kwargs.setdefault("customer_id", thread_id)
    return service.run_turn(thread_id, text, **kwargs)


def verify_phone(service, thread_id, number="01012345678"):
    """Give a number and confirm the read-back, as a real customer must."""
    readback = turn(service, thread_id, f"my number is {number}")
    assert readback.scheduling_stage == "verify_contact"
    return turn(service, thread_id, "yes")


def test_text_property_search_returns_real_workflow_results(service, thread_id, stub_rag):
    result = turn(service, thread_id, "I want a 3 bedroom apartment in New Cairo under 30 million")

    assert result.intent == "property_search"
    assert result.agent == "property"
    assert [item["property_id"] for item in result.properties] == ["501189949", "501189810"]
    # Hard filters reach the RAG workflow in the index's canonical vocabulary.
    assert stub_rag.searches[0]["requirements"] == {
        "property_type": "Apartment", "district": "New Cairo", "bedrooms": 3, "max_price": 30_000_000,
    }


def test_requirements_survive_changes_and_follow_ups(service, thread_id, stub_rag):
    turn(service, thread_id, "Something in New Cairo, my budget is 10 million")
    turn(service, thread_id, "actually make it a villa")
    follow_up = turn(service, thread_id, "with 4 bedrooms or more")

    assert follow_up.requirements == {
        "district": "New Cairo", "max_price": 10_000_000, "property_type": "Villa", "min_bedrooms": 4,
    }
    assert stub_rag.searches[-1]["requirements"]["district"] == "New Cairo"
    assert stub_rag.searches[-1]["requirements"]["max_price"] == 10_000_000
    # Earlier turns are passed to grounded generation as history.
    assert len(stub_rag.searches[-1]["history"]) >= 4


def test_follow_up_about_selected_property_uses_that_listing_only(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    result = turn(service, thread_id, "tell me more about the second one")

    assert result.intent == "property_information"
    assert result.selected_property_id == "501189810"
    assert stub_rag.follow_ups[-1]["property_ids"] == ["501189810"]


def test_customers_do_not_share_state(service, stub_rag):
    first, second = f"web:{uuid4().hex}", f"web:{uuid4().hex}"
    turn(service, first, "villa in Sheikh Zayed under 9 million")
    other = turn(service, second, "hello")

    assert other.requirements == {}
    assert service.get_state(second).get("candidates") is None


def test_viewing_is_only_reserved_after_explicit_confirmation(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    turn(service, thread_id, "I want to visit the first one")
    offered = turn(service, thread_id, "tomorrow")
    assert offered.scheduling_stage == "choose_slot"

    contact = turn(service, thread_id, "the first")
    assert contact.scheduling_stage == "need_contact"
    readback = turn(service, thread_id, "0101 234 5678")
    assert readback.scheduling_stage == "verify_contact"
    assert "0101 234 5678" in readback.response  # the number is read back to the customer
    assert readback.appointment is None
    confirm = turn(service, thread_id, "yes that's right")
    assert confirm.scheduling_stage == "confirm"
    assert confirm.appointment is None  # nothing booked yet

    booked = turn(service, thread_id, "yes please")
    assert booked.appointment is not None
    assert booked.appointment["status"] == "pending_confirmation"
    assert booked.appointment["property_id"] == "501189949"
    assert "advisor will call you to confirm" in booked.response
    assert crm_service.get_lead(thread_id)["status"] == "viewing_requested"


def test_declining_confirmation_books_nothing(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    turn(service, thread_id, "book a viewing for the second one")
    turn(service, thread_id, "tomorrow")
    turn(service, thread_id, "the first")
    verify_phone(service, thread_id)
    result = turn(service, thread_id, "no")

    assert result.appointment is None
    assert result.scheduling_stage is None


def test_taken_slot_is_not_offered_again(service, stub_rag):
    def book(thread):
        turn(service, thread, "apartments in New Cairo")
        turn(service, thread, "visit the first one")
        offered = turn(service, thread, "tomorrow")
        turn(service, thread, "the first")
        verify_phone(service, thread)
        return offered, turn(service, thread, "yes")

    first_offer, first_booking = book(f"web:{uuid4().hex}")
    second_offer, second_booking = book(f"web:{uuid4().hex}")

    def offered_times(result):
        listed = result.response.split(": ", 1)[1].split(". ", 1)[0]
        return [item.strip() for item in listed.split(",")]

    taken = datetime.fromisoformat(first_booking.appointment["starts_at"]).strftime("%H:%M")
    assert taken in offered_times(first_offer)
    assert taken not in offered_times(second_offer)
    assert second_booking.appointment["starts_at"] != first_booking.appointment["starts_at"]


def test_switching_agents_keeps_context(service, thread_id, stub_rag):
    turn(service, thread_id, "apartment in New Cairo, budget 30 million, for investment")
    turn(service, thread_id, "the first one")
    sales = turn(service, thread_id, "what's the payment plan?")
    assert sales.agent == "sales"
    assert stub_rag.follow_ups[-1]["property_ids"] == ["501189949"]

    handoff = turn(service, thread_id, "I want to talk to a sales agent")
    assert handoff.agent == "handoff"
    assert handoff.handoff == {"status": "awaiting_contact"}
    done = turn(service, thread_id, "01012345678")
    assert done.handoff["status"] == "requested"

    ticket = [item for item in crm_service.list_handoffs() if item["customer_id"] == thread_id][-1]
    assert ticket["summary"]["requirements"]["district"] == "New Cairo"
    assert ticket["summary"]["qualification"]["purpose"] == "investment"
    assert ticket["summary"]["selected_property_id"] == "501189949"


def test_qualification_asks_for_missing_details_and_updates_crm(service, thread_id, stub_rag):
    result = turn(service, thread_id, "I'm looking for a property")
    assert result.agent == "qualification"
    assert "type of property" in result.response
    assert crm_service.get_lead(thread_id)["status"] == "qualifying"


def test_arabic_conversation_stays_arabic(service, thread_id, stub_rag):
    greeting = turn(service, thread_id, "السلام عليكم")
    assert greeting.language == "ar"
    assert "أهلاً" in greeting.response

    turn(service, thread_id, "عايز شقة في التجمع")
    schedule = turn(service, thread_id, "عايز احجز معاينة للتاني")
    assert schedule.language == "ar"
    assert "يوم" in schedule.response


def test_search_failure_is_recoverable(service, thread_id, stub_rag):
    stub_rag.fail_with = TimeoutError("qdrant timed out")
    failed = turn(service, thread_id, "villa in Sheikh Zayed")
    assert failed.error and "property_search_unavailable" in failed.error
    assert "try again" in failed.response.lower()
    assert failed.requirements["district"] == "Sheikh Zayed"  # nothing lost

    stub_rag.fail_with = None
    recovered = turn(service, thread_id, "try again please, 4 bedrooms")
    assert recovered.error is None
    assert stub_rag.searches[-1]["requirements"]["district"] == "Sheikh Zayed"


def test_clearing_a_filter_reruns_the_search(service, thread_id, stub_rag):
    turn(service, thread_id, "villa in Sheikh Zayed under 9 million")
    result = turn(service, thread_id, "Without Sheikh Zayed", clear_requirements=["district"])

    assert result.intent == "property_search"
    assert "district" not in stub_rag.searches[-1]["requirements"]
    assert stub_rag.searches[-1]["requirements"]["property_type"] == "Villa"


def test_blank_message_is_rejected(service, thread_id):
    import pytest

    with pytest.raises(ValueError):
        turn(service, thread_id, "   ")


def test_ordinal_that_picks_the_property_does_not_also_pick_a_slot(service, thread_id, stub_rag):
    # Regression (found in a real Gemini Live call): "the second apartment,
    # tomorrow" used "second" as the 2nd time slot and skipped the offer.
    turn(service, thread_id, "apartments in New Cairo")
    result = turn(service, thread_id, "I would like to book a viewing of the second apartment, tomorrow")

    assert result.selected_property_id == "501189810"
    assert result.scheduling_stage == "choose_slot"
    assert "Available times" in result.response


def test_new_time_replaces_earlier_choice_before_confirmation(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    turn(service, thread_id, "book a viewing of the first one tomorrow")
    turn(service, thread_id, "11am")
    changed = turn(service, thread_id, "actually make it 3pm")  # while we wait for a phone number
    assert changed.scheduling_stage == "need_contact"
    verify_phone(service, thread_id)
    confirm = turn(service, thread_id, "I changed my mind, make it 4pm")
    assert "16:00" in confirm.response


def test_booking_survives_phone_digits_and_stray_numbers(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    turn(service, thread_id, "book a viewing of the first one tomorrow")
    turn(service, thread_id, "11am")
    searches_before = len(stub_rag.searches)
    readback = turn(service, thread_id, "My phone number is 010 123 45678")

    assert readback.scheduling_stage == "verify_contact"
    assert len(stub_rag.searches) == searches_before  # no accidental new search
    assert "max_price" not in readback.requirements
    assert turn(service, thread_id, "yes").scheduling_stage == "confirm"
    assert turn(service, thread_id, "yes").appointment["status"] == "pending_confirmation"


def test_explicit_new_search_leaves_the_booking_flow(service, thread_id, stub_rag):
    turn(service, thread_id, "apartments in New Cairo")
    turn(service, thread_id, "book a viewing of the first one tomorrow")
    result = turn(service, thread_id, "actually show me villas in Sheikh Zayed instead")
    assert result.intent == "property_search"
    assert result.requirements["property_type"] == "Villa"
