"""The viewing-booking journey end to end: verification, booking, cancellation, status, alerts."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from app.agents.tools.crm import crm_service
from app.agents.tools.scheduling import scheduling_service
from app.database import models as db
from app.database.session import session_scope
from app.scheduling.booking import BookingConflictError, PhoneNotVerifiedError


def say(service, thread, text, **kwargs):
    kwargs.setdefault("channel", "website")
    kwargs.setdefault("customer_id", thread)
    return service.run_turn(thread, text, **kwargs)


def bookings():
    with session_scope() as session:
        return [
            {"status": b.status, "phone": b.contact_phone, "property": b.property_id, "start": b.starts_at, "id": str(b.public_id)}
            for b in session.query(db.Booking).order_by(db.Booking.id)
        ]


def alerts(kind=None):
    with session_scope() as session:
        query = session.query(db.StaffNotification).order_by(db.StaffNotification.id)
        return [row.message for row in query if kind is None or row.kind == kind]


def choose_slot(service, thread, *, which=1, **kwargs):
    """Search, pick listing ``which`` and a time tomorrow; stop where the phone is needed."""
    say(service, thread, "apartments in New Cairo", **kwargs)
    say(service, thread, f"book a viewing of the {'first' if which == 1 else 'second'} one tomorrow", **kwargs)
    return say(service, thread, "the first", **kwargs)


def website_booking(service, thread, number="01012345678"):
    choose_slot(service, thread)
    say(service, thread, f"my number is {number}")
    say(service, thread, "yes")  # the number read-back
    return say(service, thread, "yes")  # the booking confirmation


# ---------------------------------------------------------------------------
# Phone verification
# ---------------------------------------------------------------------------


def test_the_number_is_read_back_and_nothing_is_booked_until_it_is_confirmed(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    readback = say(service, thread_id, "my number is 010 1234 5678")
    assert readback.scheduling_stage == "verify_contact"
    assert "0101 234 5678" in readback.response and readback.appointment is None
    assert bookings() == []

    confirm = say(service, thread_id, "yes")
    assert confirm.scheduling_stage == "confirm" and bookings() == []  # verified, still not booked

    booked = say(service, thread_id, "yes")
    assert booked.appointment["status"] == "pending_confirmation"
    assert [b["phone"] for b in bookings()] == ["+201012345678"]


def test_an_invalid_number_is_refused_with_an_explanation(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    result = say(service, thread_id, "my number is 12345678901")
    assert result.scheduling_stage == "need_contact"
    assert "valid mobile number" in result.response
    assert say(service, thread_id, "01012345678").scheduling_stage == "verify_contact"  # a good one is accepted afterwards


def test_denying_the_read_back_asks_for_the_right_number(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    say(service, thread_id, "01012345678")
    rejected = say(service, thread_id, "no")
    assert rejected.scheduling_stage == "need_contact" and "correct number" in rejected.response

    say(service, thread_id, "01123456789")
    say(service, thread_id, "yes")
    say(service, thread_id, "yes")
    assert [b["phone"] for b in bookings()] == ["+201123456789"]  # the corrected number, not the first guess


def test_giving_a_new_number_instead_of_answering_re_verifies_it(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    say(service, thread_id, "01012345678")
    again = say(service, thread_id, "no it's 01155555555")  # a correction, not a rejection
    assert again.scheduling_stage == "verify_contact" and "0115 555 5555" in again.response
    say(service, thread_id, "yes")
    say(service, thread_id, "yes")
    assert [b["phone"] for b in bookings()] == ["+201155555555"]


def test_changing_the_number_after_verification_requires_verifying_the_new_one(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    say(service, thread_id, "01012345678")
    assert say(service, thread_id, "yes").scheduling_stage == "confirm"

    changed = say(service, thread_id, "actually use 01299999999")
    assert changed.scheduling_stage == "verify_contact"  # not "confirm": the new number is unverified
    assert bookings() == []
    assert say(service, thread_id, "yes").scheduling_stage == "confirm"
    assert say(service, thread_id, "yes").appointment is not None
    assert [b["phone"] for b in bookings()] == ["+201299999999"]


def test_yes_to_the_booking_cannot_skip_verification_of_an_unverified_number(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    result = say(service, thread_id, "01012345678 yes book it")  # number and consent in one breath
    assert result.scheduling_stage == "verify_contact" and result.appointment is None
    assert bookings() == []


def test_voice_reads_the_number_back_digit_by_digit(service, thread_id, stub_rag):
    say(service, thread_id, "apartments in New Cairo", modality="voice")
    say(service, thread_id, "book a viewing of the first one tomorrow", modality="voice")
    say(service, thread_id, "the first", modality="voice")
    readback = say(service, thread_id, "my number is zero one zero one two three four five six seven eight", modality="voice")
    assert readback.scheduling_stage == "verify_contact"
    assert "0 1 0 1 2 3 4 5 6 7 8" in readback.response


def test_arabic_digits_and_words_are_understood_and_the_flow_stays_arabic(service, thread_id, stub_rag):
    say(service, thread_id, "عايز شقة في التجمع")
    say(service, thread_id, "عايز احجز معاينة للاول بكره")
    say(service, thread_id, "الاول")
    readback = say(service, thread_id, "رقمي ٠١٠١٢٣٤٥٦٧٨")
    assert readback.language == "ar" and readback.scheduling_stage == "verify_contact"
    assert "0101 234 5678" in readback.response and "صح كده" in readback.response
    assert say(service, thread_id, "أيوه").scheduling_stage == "confirm"
    booked = say(service, thread_id, "أيوه")
    assert booked.appointment is not None and "مستشار" in booked.response


def test_whatsapp_customers_are_verified_by_the_channel_itself(service, stub_rag):
    thread = "whatsapp:201012345678"
    kwargs = {"channel": "whatsapp"}
    say(service, thread, "apartments in New Cairo", **kwargs)
    say(service, thread, "book a viewing of the first one tomorrow", **kwargs)
    confirm = say(service, thread, "the first", **kwargs)
    assert confirm.scheduling_stage == "confirm"  # no number requested, no read-back
    booked = say(service, thread, "yes", **kwargs)
    assert booked.appointment is not None
    assert [b["phone"] for b in bookings()] == ["+201012345678"]
    with session_scope() as session:
        customer = session.query(db.Customer).filter_by(external_id=thread).one()
        assert (customer.phone_verified, customer.phone_verification_method) == (True, "whatsapp_sender")


def test_a_whatsapp_customer_who_gives_another_number_must_verify_it(service, stub_rag):
    thread = "whatsapp:201012345678"
    kwargs = {"channel": "whatsapp"}
    say(service, thread, "apartments in New Cairo", **kwargs)
    say(service, thread, "book a viewing of the first one tomorrow", **kwargs)
    say(service, thread, "the first", **kwargs)
    other = say(service, thread, "my other number is 01155555555", **kwargs)
    assert other.scheduling_stage == "verify_contact"


def test_the_booking_service_refuses_an_unverified_customer_whatever_the_conversation_says(service, thread_id, stub_rag):
    crm_service.upsert_customer(thread_id, phone="01012345678")  # stored, never verified
    slot = scheduling_service.get_available_slots("501189949", datetime.now().date() + timedelta(days=2))[0]
    with pytest.raises(PhoneNotVerifiedError):
        scheduling_service.request_viewing(
            property_id="501189949", customer_id=thread_id, starts_at=slot["starts_at"], ends_at=slot["ends_at"]
        )
    assert bookings() == []


def test_verification_is_stored_with_the_customer_and_tied_to_that_number(service, thread_id, stub_rag):
    website_booking(service, thread_id)
    with session_scope() as session:
        customer = session.query(db.Customer).filter_by(external_id=thread_id).one()
        assert (customer.phone, customer.phone_verified, customer.phone_verification_method) == ("+201012345678", True, "readback")
    crm_service.upsert_customer(thread_id, phone="01255555555")  # a different number...
    with session_scope() as session:
        assert session.query(db.Customer).filter_by(external_id=thread_id).one().phone_verified is False  # ...is not verified


# ---------------------------------------------------------------------------
# Booking, alerts and conflicts
# ---------------------------------------------------------------------------


def test_a_booking_alerts_staff_and_advances_the_lead(service, thread_id, stub_rag):
    booked = website_booking(service, thread_id)
    assert "advisor will call you to confirm" in booked.response  # never claims a confirmed visit

    (message,) = alerts("booking_requested")
    assert "501189949" in message and "+201012345678" in message and "needs confirmation" in message
    assert crm_service.get_lead(thread_id)["status"] == "viewing_requested"
    assert crm_service.get_lead(thread_id)["appointment_id"] == booked.appointment["id"]


def test_only_slots_the_calendar_offers_can_be_booked(service, thread_id, stub_rag):
    crm_service.upsert_customer(thread_id, phone="01012345678", phone_verified=True)
    tomorrow = datetime.now().date() + timedelta(days=2)
    odd_time = datetime.combine(tomorrow, datetime.min.time()).astimezone().replace(hour=3)  # 3 a.m.: nobody is working
    with pytest.raises(BookingConflictError):
        scheduling_service.request_viewing(
            property_id="501189949", customer_id=thread_id,
            starts_at=odd_time.isoformat(), ends_at=(odd_time + timedelta(hours=1)).isoformat(),
        )


def test_the_second_customer_to_confirm_the_same_slot_is_offered_alternatives(service, stub_rag):
    first, second = f"web:{uuid4().hex}", f"web:{uuid4().hex}"
    for thread, number in ((first, "01012345678"), (second, "01122222222")):
        choose_slot(service, thread)
        say(service, thread, f"my number is {number}")
        say(service, thread, "yes")  # both are now waiting at "confirm" for the same slot

    assert say(service, first, "yes").appointment is not None
    late = say(service, second, "yes")
    assert late.appointment is None and late.scheduling_stage == "choose_slot"
    assert "just taken" in late.response
    assert len(bookings()) == 1


# ---------------------------------------------------------------------------
# Cancellation and status
# ---------------------------------------------------------------------------


def test_a_customer_can_cancel_their_viewing_and_staff_are_told(service, thread_id, stub_rag):
    website_booking(service, thread_id)
    ask = say(service, thread_id, "cancel my viewing")
    assert ask.scheduling_stage == "cancel_confirm" and "Cancel your viewing" in ask.response

    kept = say(service, thread_id, "no")
    assert "kept your viewing" in kept.response and bookings()[0]["status"] == "pending_confirmation"

    say(service, thread_id, "cancel my viewing")
    done = say(service, thread_id, "yes")
    assert "cancelled" in done.response and done.appointment["status"] == "cancelled"
    assert bookings()[0]["status"] == "cancelled"
    assert any("cancelled by customer" in message for message in alerts("booking_cancelled"))


def test_a_cancelled_slot_can_be_booked_by_someone_else(service, stub_rag):
    first, second = f"web:{uuid4().hex}", f"web:{uuid4().hex}"
    website_booking(service, first)
    say(service, first, "cancel my viewing")
    say(service, first, "yes")
    again = website_booking(service, second, number="01122222222")
    assert again.appointment is not None  # the freed slot was offered and taken
    assert sorted(b["status"] for b in bookings()) == ["cancelled", "pending_confirmation"]


def test_cancelling_with_several_viewings_asks_which_one(service, thread_id, stub_rag):
    website_booking(service, thread_id)
    second = choose_slot(service, thread_id, which=2)  # a second viewing of another listing, same customer
    assert second.scheduling_stage == "confirm"  # the number was already verified: no second read-back
    say(service, thread_id, "yes")
    assert len([b for b in bookings() if b["status"] == "pending_confirmation"]) == 2

    ask = say(service, thread_id, "cancel my appointment")
    assert ask.scheduling_stage == "cancel_pick" and "1." in ask.response and "2." in ask.response
    confirm = say(service, thread_id, "the second")
    assert confirm.scheduling_stage == "cancel_confirm"
    say(service, thread_id, "yes")
    assert sorted(b["status"] for b in bookings()) == ["cancelled", "pending_confirmation"]


def test_a_customer_cannot_cancel_someone_elses_booking(service, thread_id, stub_rag):
    website_booking(service, thread_id)
    booking_id = bookings()[0]["id"]
    with pytest.raises(LookupError):
        scheduling_service.cancel(booking_id, customer_id=f"web:{uuid4().hex}")
    assert bookings()[0]["status"] == "pending_confirmation"

    stranger = f"web:{uuid4().hex}"
    assert "don't have an upcoming viewing" in say(service, stranger, "cancel my viewing").response


def test_cancel_in_the_middle_of_booking_abandons_that_booking(service, thread_id, stub_rag):
    choose_slot(service, thread_id)
    result = say(service, thread_id, "cancel")
    assert result.scheduling_stage is None and "haven't booked anything" in result.response
    assert bookings() == []


def test_the_customer_can_ask_whether_the_viewing_is_confirmed(service, thread_id, stub_rag):
    assert "don't have any viewing requests" in say(service, thread_id, "is my viewing confirmed?").response
    website_booking(service, thread_id)
    assert "waiting for an advisor" in say(service, thread_id, "is my viewing confirmed?").response

    scheduling_service.staff_confirm(__import__("uuid").UUID(bookings()[0]["id"]), by="Sara")
    assert "is confirmed" in say(service, thread_id, "is my viewing confirmed?").response


def test_staff_actions_reach_whatsapp_customers_but_not_website_visitors(service, stub_rag, monkeypatch, settings_env):
    sent = []
    settings_env(WHATSAPP_ACCESS_TOKEN="token", WHATSAPP_PHONE_NUMBER_ID="123")
    monkeypatch.setattr(httpx, "post", lambda url, **kw: sent.append((url, kw["json"])) or httpx.Response(200, request=httpx.Request("POST", url)))

    thread = "whatsapp:201012345678"
    for text in ("apartments in New Cairo", "book a viewing of the first one tomorrow", "the first", "yes"):
        say(service, thread, text, channel="whatsapp")
    scheduling_service.staff_confirm(__import__("uuid").UUID(bookings()[0]["id"]), by="Sara")

    (url, body), = [call for call in sent if "/messages" in call[0]]
    assert body["to"] == "201012345678" and "is confirmed" in body["text"]["body"]

    web = f"web:{uuid4().hex}"
    website_booking(service, web, number="01122222222")
    before = len(sent)
    scheduling_service.staff_confirm(__import__("uuid").UUID(bookings()[1]["id"]), by="Sara")
    assert len(sent) == before  # a website visitor has no push channel
