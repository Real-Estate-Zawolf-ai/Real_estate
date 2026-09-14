from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NormalizedProperty:
    """
    Canonical representation of a real-estate property.

    Missing values remain None.
    Original text is preserved.
    """

    property_type: str | None = None
    transaction_type: str | None = None

    price: float | None = None
    currency: str | None = None

    area: float | None = None
    area_unit: str | None = None

    bedrooms: int | None = None
    bathrooms: int | None = None

    location: str | None = None
    compound: str | None = None
    developer: str | None = None

    finishing: str | None = None
    delivery: str | None = None

    amenities: list[str] = field(default_factory=list)

    raw_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class PropertyNormalizer:
    """
    Normalize real-estate information from OCR/text.

    Input:
        Cleaned text produced by the OCR pipeline.

    Output:
        NormalizedProperty

    Pipeline:

        OCR
          ↓
        DocumentParser
          ↓
        OCRPostProcessor
          ↓
        PropertyNormalizer
          ↓
        FieldMapper
          ↓
        Embeddings / Retrieval

    Does NOT:
        - perform OCR
        - create embeddings
        - perform retrieval
        - call an LLM
        - store data in a database
    """

    PROPERTY_TYPE_PATTERNS = {
        "apartment": [
            r"\bapartment\b",
            r"\bflat\b",
            r"شقه",
            r"شقة",
        ],
        "villa": [
            r"\bvilla\b",
            r"فيلا",
        ],
        "townhouse": [
            r"\btown\s+house\b",
            r"تاون\s+هاوس",
        ],
        "twin_house": [
            r"\btwin\s+house\b",
            r"توين\s+هاوس",
        ],
        "chalet": [
            r"\bchalet\b",
            r"شاليه",
        ],
        "studio": [
            r"\bstudio\b",
            r"استوديو",
            r"ستوديو",
        ],
        "duplex": [
            r"\bduplex\b",
            r"دوبلكس",
        ],
        "penthouse": [
            r"\bpenthouse\b",
            r"بنتهاوس",
        ],
        "office": [
            r"\boffice\b",
            r"مكتب",
            r"اداري",
            r"إداري",
        ],
        "shop": [
            r"\bshop\b",
            r"\bretail\b",
            r"محل",
            r"تجاري",
        ],
        "land": [
            r"\bland\b",
            r"ارض",
            r"أرض",
        ],
    }

    TRANSACTION_PATTERNS = {
        "sale": [
            r"\bfor\s+sale\b",
            r"\bsale\b",
            r"\bbuy\b",
            r"للبيع",
            r"بيع",
        ],
        "rent": [
            r"\bfor\s+rent\b",
            r"\brent\b",
            r"\brental\b",
            r"للايجار",
            r"للإيجار",
            r"ايجار",
            r"إيجار",
        ],
    }

    FINISHING_PATTERNS = {
        "fully_finished": [
            r"fully\s+finished",
            r"\bfinished\b",
            r"تشطيب\s+كامل",
            r"متشطب",
            r"مشطب",
        ],
        "semi_finished": [
            r"semi[\s-]+finished",
            r"نصف\s+تشطيب",
            r"نص\s+تشطيب",
        ],
        "core_and_shell": [
            r"core\s+(?:and|&)\s+shell",
            r"على\s+الطوب",
            r"طوب\s+احمر",
            r"طوب احمر",
        ],
    }

    CURRENCY_PATTERNS = [
        (r"(?:egp|le|l\.e\.|جنيه|جنية|ج\.م)", "EGP"),
        (r"(?:usd|\$|دولار)", "USD"),
        (r"(?:eur|€|يورو)", "EUR"),
        (r"(?:gbp|£|جنيه\s+استرليني)", "GBP"),
    ]

    AREA_PATTERNS = [
        r"(\d+(?:[.,]\d+)?)\s*(?:sqm|sq\.?\s*m|m2|m²|متر\s+مربع|متر)",
        r"(?:area|مساحة)\s*[:\-]?\s*(\d+(?:[.,]\d+)?)",
    ]

    BEDROOM_PATTERNS = [
        r"(\d+)\s*(?:bedrooms?|beds?|br)\b",
        r"(?:bedrooms?|beds?|br)\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*(?:غرف?\s+نوم|غرف نوم)",
        r"(?:غرف?\s+نوم|غرف نوم)\s*[:\-]?\s*(\d+)",
    ]

    BATHROOM_PATTERNS = [
        r"(\d+)\s*(?:bathrooms?|baths?|ba)\b",
        r"(?:bathrooms?|baths?|ba)\s*[:\-]?\s*(\d+)",
        r"(\d+)\s*(?:حمامات?|حمام)\b",
        r"(?:حمامات?|حمام)\s*[:\-]?\s*(\d+)",
    ]

    def normalize(self, text: str) -> NormalizedProperty:
        """
        Normalize a real-estate text document.
        """

        if not text or not text.strip():
            logger.warning(
                "Property normalization skipped: empty text"
            )
            return NormalizedProperty(raw_text="")

        text = text.strip()

        logger.info(
            "Starting property normalization | characters=%d",
            len(text),
        )

        normalized = NormalizedProperty(raw_text=text)

        normalized.property_type = self._extract_property_type(text)

        normalized.transaction_type = (
            self._extract_transaction_type(text)
        )

        normalized.price, normalized.currency = (
            self._extract_price(text)
        )

        normalized.area, normalized.area_unit = (
            self._extract_area(text)
        )

        normalized.bedrooms = self._extract_integer(
            text,
            self.BEDROOM_PATTERNS,
        )

        normalized.bathrooms = self._extract_integer(
            text,
            self.BATHROOM_PATTERNS,
        )

        normalized.finishing = (
            self._extract_from_patterns(
                text,
                self.FINISHING_PATTERNS,
            )
        )

        normalized.location = self._extract_location(text)

        normalized.compound = self._extract_compound(text)

        normalized.developer = self._extract_developer(text)

        normalized.delivery = self._extract_delivery(text)

        normalized.amenities = self._extract_amenities(text)

        logger.info(
            "Property normalization completed | "
            "type=%s | transaction=%s | price=%s | "
            "area=%s | bedrooms=%s | bathrooms=%s",
            normalized.property_type,
            normalized.transaction_type,
            normalized.price,
            normalized.area,
            normalized.bedrooms,
            normalized.bathrooms,
        )

        return normalized

    def normalize_to_dict(
        self,
        text: str,
    ) -> dict[str, Any]:
        """
        Normalize property text and return a dictionary.
        """

        property_data = self.normalize(text)

        return {
            "property_type": property_data.property_type,
            "transaction_type": property_data.transaction_type,
            "price": property_data.price,
            "currency": property_data.currency,
            "area": property_data.area,
            "area_unit": property_data.area_unit,
            "bedrooms": property_data.bedrooms,
            "bathrooms": property_data.bathrooms,
            "location": property_data.location,
            "compound": property_data.compound,
            "developer": property_data.developer,
            "finishing": property_data.finishing,
            "delivery": property_data.delivery,
            "amenities": property_data.amenities,
            "raw_text": property_data.raw_text,
            "extra": property_data.extra,
        }

    def _extract_property_type(
        self,
        text: str,
    ) -> str | None:
        return self._extract_from_patterns(
            text,
            self.PROPERTY_TYPE_PATTERNS,
        )

    def _extract_transaction_type(
        self,
        text: str,
    ) -> str | None:
        return self._extract_from_patterns(
            text,
            self.TRANSACTION_PATTERNS,
        )

    def _extract_price(
        self,
        text: str,
    ) -> tuple[float | None, str | None]:
        """
        Extract price and currency.

        Examples:
            3,500,000 EGP
            3.5 million EGP
            ٣,٥٠٠,٠٠٠ جنيه
        """

        normalized_text = self._normalize_numbers(text)

        currency = self._detect_currency(normalized_text)

        # Compact million notation.
        million_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*"
            r"(?:million|m|مليون)(?:\s+جنيه)?",
            normalized_text,
            flags=re.IGNORECASE,
        )

        if million_match:
            value = self._to_float(
                million_match.group(1)
            )

            if value is not None:
                return value * 1_000_000, currency

        price_patterns = [
            r"(?:price|السعر)\s*[:\-]?\s*"
            r"([\d,]+(?:\.\d+)?)",

            r"([\d,]+(?:\.\d+)?)\s*"
            r"(?:egp|le|l\.e\.|جنيه|جنية|ج\.م)",

            r"(?:egp|le|l\.e\.|جنيه|جنية|ج\.م)\s*"
            r"([\d,]+(?:\.\d+)?)",

            r"\$\s*([\d,]+(?:\.\d+)?)",
        ]

        for pattern in price_patterns:
            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )

            if match:
                value = self._to_float(
                    match.group(1)
                )

                if value is not None:
                    return value, currency

        return None, currency

    def _extract_area(
        self,
        text: str,
    ) -> tuple[float | None, str | None]:

        normalized_text = self._normalize_numbers(text)

        for pattern in self.AREA_PATTERNS:
            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            value = self._to_float(
                match.group(1)
            )

            if value is not None:
                return value, "sqm"

        return None, None

    def _extract_integer(
        self,
        text: str,
        patterns: list[str],
    ) -> int | None:

        normalized_text = self._normalize_numbers(text)

        for pattern in patterns:
            match = re.search(
                pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            for group in match.groups():
                if group is None:
                    continue

                try:
                    return int(group)
                except ValueError:
                    continue

        return None

    def _extract_location(
        self,
        text: str,
    ) -> str | None:

        patterns = [
            r"(?:location|loc|الموقع)\s*[:\-]\s*([^\n,]+)",
            r"(?:located\s+in|في)\s+"
            r"([A-Za-z\u0600-\u06FF][^\n,]+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                location = match.group(1).strip()

                if location:
                    return location

        return None

    def _extract_compound(
        self,
        text: str,
    ) -> str | None:

        patterns = [
            r"(?:compound|project|كمبوند|مشروع)"
            r"\s*[:\-]\s*([^\n,]+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    def _extract_developer(
        self,
        text: str,
    ) -> str | None:

        patterns = [
            r"(?:developer|developed\s+by|المطور"
            r"|المطور\s+العقاري)"
            r"\s*[:\-]\s*([^\n,]+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    def _extract_delivery(
        self,
        text: str,
    ) -> str | None:

        patterns = [
            r"(?:delivery|handover|التسليم)"
            r"\s*[:\-]?\s*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = match.group(1).strip()

                if value:
                    return value

        return None

    def _extract_amenities(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract common real-estate amenities.
        """

        amenities_map = {
            "pool": [
                "pool",
                "swimming pool",
                "حمام سباحة",
                "حمام سباحه",
            ],
            "gym": [
                "gym",
                "fitness",
                "جيم",
                "نادي صحي",
            ],
            "parking": [
                "parking",
                "garage",
                "جراج",
                "باركينج",
                "موقف سيارات",
            ],
            "security": [
                "security",
                "24/7 security",
                "امن",
                "أمن",
                "حراسة",
            ],
            "garden": [
                "garden",
                "حديقة",
                "جاردن",
            ],
            "clubhouse": [
                "clubhouse",
                "كلوب هاوس",
            ],
            "elevator": [
                "elevator",
                "lift",
                "اسانسير",
                "أسانسير",
            ],
        }

        text_lower = text.lower()

        found: list[str] = []

        for canonical_name, variants in amenities_map.items():
            for variant in variants:
                if variant.lower() in text_lower:
                    found.append(canonical_name)
                    break

        return found

    def _extract_from_patterns(
        self,
        text: str,
        pattern_groups: dict[str, list[str]],
    ) -> str | None:

        for canonical_value, patterns in pattern_groups.items():
            for pattern in patterns:
                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                ):
                    return canonical_value

        return None

    def _detect_currency(
        self,
        text: str,
    ) -> str | None:

        for pattern, currency in self.CURRENCY_PATTERNS:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return currency

        return None

    @staticmethod
    def _normalize_numbers(
        text: str,
    ) -> str:
        """
        Convert Arabic/Persian digits to Western digits.
        """

        arabic_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩",
            "0123456789",
        )

        persian_digits = str.maketrans(
            "۰۱۲۳۴۵۶۷۸۹",
            "0123456789",
        )

        text = text.translate(arabic_digits)
        text = text.translate(persian_digits)

        return text

    @staticmethod
    def _to_float(
        value: str,
    ) -> float | None:
        """
        Convert a numeric string into float.

        Handles:
            3500000
            3,500,000
            150.5
            150,5
        """

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        # Remove thousands separators.
        if "," in value and "." in value:
            # Example: 3,500,000.50
            value = value.replace(",", "")

        elif "," in value:
            parts = value.split(",")

            # 3,500,000 -> thousands separators
            if all(
                len(part) == 3
                for part in parts[1:]
            ):
                value = "".join(parts)

            # 150,5 -> decimal separator
            else:
                value = value.replace(",", ".")

        try:
            return float(value)
        except ValueError:
            logger.warning(
                "Could not convert value to float: %s",
                value,
            )
            return None

