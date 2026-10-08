from datetime import date

import pytest

from app.agents.understanding import (
    classify_intent,
    detect_language,
    extract_date,
    extract_qualification,
    extract_requirements,
    extract_time,
    is_affirmative,
    is_negative,
    resolve_property_reference,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "I want a 3 bedroom apartment in New Cairo under 5 million",
            {"property_type": "Apartment", "district": "New Cairo", "bedrooms": 3, "max_price": 5_000_000},
        ),
        (
            "عايز شقة 3 غرف في التجمع بحدود 5 مليون",
            {"property_type": "Apartment", "district": "New Cairo", "bedrooms": 3, "max_price": 5_000_000},
        ),
        (
            "عايز فيلا في الشيخ زايد للبيع 4 غرف او اكتر",
            {"property_type": "Villa", "district": "Sheikh Zayed", "min_bedrooms": 4, "transaction_type": "sale"},
        ),
        (
            "villa between 5 and 8 million for rent",
            {"property_type": "Villa", "transaction_type": "rent", "min_price": 5_000_000, "max_price": 8_000_000},
        ),
        ("villa or apartment please", {}),  # ambiguous type: no filter guessed
    ],
)
def test_extract_requirements_uses_index_vocabulary(text, expected):
    assert extract_requirements(text) == expected


def test_financing_amounts_are_not_price_filters():
    text = "My budget is 6M, down payment 500k and 40k monthly, for investment"
    assert extract_requirements(text) == {"max_price": 6_000_000}
    assert extract_qualification(text) == {
        "budget": 6_000_000,
        "down_payment": 500_000,
        "monthly_installment": 40_000,
        "purpose": "investment",
    }


def test_arabic_financing_signals():
    signals = extract_qualification("ميزانيتي 4 مليون ومقدم 400 الف وقسط 30 الف في الشهر")
    assert signals["budget"] == 4_000_000
    assert signals["down_payment"] == 400_000
    assert signals["monthly_installment"] == 30_000
    assert signals["payment_method"] == "installments"


def test_contact_details_are_not_money():
    signals = extract_qualification("my number is 01012345678, I'm Omar")
    assert signals == {"phone": "01012345678", "name": "Omar"}


def test_words_inside_other_words_do_not_trigger_filters():
    # "please" contains "lease", "down" contains "own".
    assert "transaction_type" not in extract_requirements("500k down payment please")


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Can I speak to a human?", "human_handoff"),
        ("عايز اكلم حد من المبيعات", "human_handoff"),
        ("I'd like to schedule a viewing", "schedule_visit"),
        ("ممكن احجز معاينة", "schedule_visit"),
        ("What's the payment plan?", "payment_plan"),
        ("is it still available?", "availability"),
        ("show me villas", "property_search"),
        ("hello", "greeting"),
        ("ok", "conversational"),
    ],
)
def test_classify_intent(text, intent):
    assert classify_intent(text) == intent


def test_language_detection():
    assert detect_language("عايز شقة") == "ar"
    assert detect_language("I want a flat") == "en"
    assert detect_language("12345", fallback="ar") == "ar"


def test_dates_and_times():
    today = date(2026, 10, 8)  # Thursday
    assert extract_date("tomorrow", today) == date(2026, 10, 9)
    assert extract_date("بكرة", today) == date(2026, 10, 9)
    assert extract_date("Saturday", today) == date(2026, 10, 10)
    assert extract_date("15/10", today) == date(2026, 10, 15)
    assert extract_date("2026-10-20", today) == date(2026, 10, 20)
    assert extract_time("at 4") == (16, 0)
    assert extract_time("10am") == (10, 0)
    assert extract_time("الساعة 12") == (12, 0)
    assert extract_time("3 bedrooms") is None


def test_confirmation_words():
    assert is_affirmative("Yes please") and is_affirmative("تمام")
    assert is_negative("no thanks") and is_negative("لا")
    assert not is_affirmative("maybe later")


def test_resolve_property_reference():
    candidates = [{"property_id": "501189949"}, {"property_id": "501189810"}]
    assert resolve_property_reference("show me the second one", candidates) == candidates[1]
    assert resolve_property_reference("عايز التاني", candidates) == candidates[1]
    assert resolve_property_reference("tell me about 501189949", candidates) == candidates[0]
    assert resolve_property_reference("anything cheaper?", candidates) is None


@pytest.mark.parametrize(
    "text",
    ["My phone number is 010 123 45678", "my phone number is 0 1 0 1 2 3 4 5 6 7 8", "010-1234-5678"],
)
def test_spoken_phone_numbers_are_contact_not_price(text):
    # Regression (real Gemini Live call): "010 123 45678" became max_price 45678.
    assert extract_requirements(text) == {}
    assert extract_qualification(text)["phone"] == "01012345678"


@pytest.mark.parametrize(
    "text",
    [
        "My phone number is zero one zero, one two three four, five six seven eight.",
        "رقمي صفر واحد صفر واحد اتنين تلاتة اربعة خمسة ستة سبعة تمانية",
    ],
)
def test_phone_numbers_read_aloud_as_words(text):
    assert extract_qualification(text)["phone"] == "01012345678"


def test_short_digit_runs_and_counts_are_not_phones():
    assert "phone" not in extract_qualification("one two three four five six seven eight")
    assert extract_requirements("three bedrooms in one of the compounds") == {"bedrooms": 3}
