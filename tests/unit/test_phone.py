import pytest

from app.helpers.phone import display_phone, is_valid_phone, mask_phone, normalize_phone, spoken_phone


@pytest.mark.parametrize(
    "raw",
    [
        "01012345678",
        "0101 234 5678",
        "010-1234-5678",
        "(010) 1234 5678",
        "+201012345678",
        "+20 101 234 5678",
        "201012345678",
        "00201012345678",
        "٠١٠١٢٣٤٥٦٧٨",  # Arabic-Indic digits, as speech-to-text and Arabic keyboards produce
        "۰۱۰۱۲۳۴۵۶۷۸",  # Persian digits
    ],
)
def test_egyptian_mobile_formats_normalize_to_one_e164_string(raw):
    assert normalize_phone(raw) == "+201012345678"


@pytest.mark.parametrize("prefix", ["010", "011", "012", "015"])
def test_every_egyptian_operator_prefix_is_accepted(prefix):
    assert normalize_phone(f"{prefix}12345678") == f"+20{prefix[1:]}12345678"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "abc",
        "0101234",  # too short
        "010123456789",  # too long for a local number
        "01312345678",  # 013 is not an operator prefix
        "02 2345 6789",  # a landline cannot be reached by WhatsApp or an advisor's mobile call list
        "0101abc5678",  # letters inside the number
        "+0101234567",
        "++201012345678",
        "20+1012345678",
        "+999",
        "123456789",  # a bare local number from an unknown country
        "+2001012345678",  # wrong digit count after +20
    ],
)
def test_invalid_numbers_are_rejected_not_guessed(raw):
    assert normalize_phone(raw) is None
    assert is_valid_phone(raw) is False


def test_international_numbers_need_an_explicit_country_code():
    assert normalize_phone("+447911123456") == "+447911123456"
    assert normalize_phone("00447911123456") == "+447911123456"
    assert normalize_phone("+1 (415) 555-2671") == "+14155552671"


def test_normalization_is_idempotent():
    once = normalize_phone("0101 234 5678")
    assert normalize_phone(once) == once


def test_masking_hides_the_middle_of_a_number_in_logs():
    masked = mask_phone("+201012345678")
    assert masked.startswith("+201") and masked.endswith("78")
    assert "1234567" not in masked
    assert mask_phone(None) == "***"
    assert mask_phone("123") == "***"


def test_readback_formats_are_unambiguous():
    assert display_phone("+201012345678") == "0101 234 5678"
    # Voice reads digit by digit so speech can never merge or reorder them.
    assert spoken_phone("+201012345678") == "0 1 0 1 2 3 4 5 6 7 8"
    assert spoken_phone("+447911123456") == "4 4 7 9 1 1 1 2 3 4 5 6"
