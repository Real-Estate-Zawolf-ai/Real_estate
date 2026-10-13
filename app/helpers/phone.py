"""Phone-number normalisation, validation and presentation.

Numbers reach the assistant as typed text, speech-to-text output (Arabic-Indic
digits, spaces between digits) and WhatsApp sender ids (digits, no ``+``). Every
path normalises to E.164 here, so the same customer always has the same string
and a booking can never be created for a number we could not parse.
"""

from __future__ import annotations

import re


_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Egyptian mobile operators: 010 Vodafone, 011 Etisalat, 012 Orange, 015 WE.
_EG_LOCAL = re.compile(r"^01[0125]\d{8}$")
_EG_INTERNATIONAL = re.compile(r"^201[0125]\d{8}$")


def normalize_phone(raw: str | None) -> str | None:
    """Return the number in E.164 form (``+201012345678``) or None if invalid.

    Accepts Egyptian local format (``01012345678``), international with or
    without ``+``/``00``, and any other international number of 8-15 digits that
    starts with ``+``/``00``. Anything else (too short, letters, a bare local
    number from another country) is rejected rather than guessed.
    """
    if not raw:
        return None
    text = str(raw).translate(_DIGIT_TABLE).strip()
    # Only digits and common separators are allowed; "0101abc" is not a number.
    if re.search(r"[^\d+\s().\-]", text) or text.count("+") > 1 or ("+" in text and not text.startswith("+")):
        return None
    has_plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None

    if not has_plus and digits.startswith("00"):
        digits, has_plus = digits[2:], True

    if not has_plus:
        if _EG_LOCAL.match(digits):
            return "+20" + digits[1:]
        if _EG_INTERNATIONAL.match(digits):
            return "+" + digits
        return None

    if digits.startswith("20"):
        return "+" + digits if _EG_INTERNATIONAL.match(digits) else None
    if 8 <= len(digits) <= 15 and digits[0] != "0":
        return "+" + digits
    return None


def is_valid_phone(raw: str | None) -> bool:
    return normalize_phone(raw) is not None


def mask_phone(phone: str | None) -> str:
    """Log-safe form: keeps the country prefix and the last two digits."""
    value = normalize_phone(phone) or str(phone or "")
    if len(value) <= 6:
        return "***"
    return f"{value[:4]}{'*' * (len(value) - 6)}{value[-2:]}"


def display_phone(phone: str) -> str:
    """Human-readable form for reading back to the customer (``0101 234 5678``)."""
    value = normalize_phone(phone) or phone
    if value.startswith("+20") and len(value) == 13:
        local = "0" + value[3:]
        return f"{local[:4]} {local[4:7]} {local[7:]}"
    return value


def spoken_phone(phone: str) -> str:
    """Digit-by-digit form for voice read-back, so speech never merges digits."""
    value = normalize_phone(phone) or phone
    if value.startswith("+20") and len(value) == 13:
        value = "0" + value[3:]
    return " ".join(value.lstrip("+"))
