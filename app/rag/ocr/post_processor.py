from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProcessedOCR:
    """
    Cleaned and normalized OCR output.
    Retains cleaned text and a list of specific transformations made.
    """
    text: str
    lines: list[str] = field(default_factory=list)
    corrections: list[dict[str, str]] = field(default_factory=list)


class OCRPostProcessor:
    """
    Post-process OCR text before property normalization and extraction.

    Responsibilities:
        - Clean repeated whitespace and blank lines.
        - Normalize common Arabic character variations (alif, yaa, tatweel).
        - Convert Arabic-Indic and Persian numerals to Western digits so that
          property prices, areas, phone numbers, and unit numbers can be parsed cleanly.
        - Normalize Arabic punctuation (Arabic comma, semicolon, question mark).
        - Filter obvious OCR artifacts and isolated noise without dropping real content.

    Design Principle:
        Keep normalization conservative. Never destroy currency symbols,
        unit identifiers, decimal points, or address components.
    """

    def __init__(
        self,
        normalize_arabic: bool = True,
        normalize_digits: bool = True,
        remove_noise: bool = True,
    ) -> None:
        self.normalize_arabic = normalize_arabic
        self.normalize_digits = normalize_digits
        self.remove_noise = remove_noise

        logger.info(
            "OCRPostProcessor initialized | normalize_arabic=%s | normalize_digits=%s | remove_noise=%s",
            normalize_arabic,
            normalize_digits,
            remove_noise,
        )

    def process(self, text: str) -> ProcessedOCR:
        """
        Process raw OCR text and return a clean ProcessedOCR object.
        """
        if not text or not text.strip():
            logger.debug("OCR post-processing received empty text")
            return ProcessedOCR(text="", lines=[], corrections=[])

        original_text = text
        normalized_text = self._normalize_line_breaks(text)

        raw_lines = [
            line.strip()
            for line in normalized_text.split("\n")
            if line.strip()
        ]

        processed_lines: list[str] = []
        corrections: list[dict[str, str]] = []

        for line in raw_lines:
            original_line = line
            line = self._clean_whitespace(line)

            if self.normalize_arabic:
                line = self._normalize_arabic(line)

            if self.normalize_digits:
                line = self._normalize_digits(line)

            line = self._normalize_punctuation(line)

            if self.remove_noise:
                line = self._remove_obvious_noise(line)

            line = self._clean_whitespace(line)

            if not line:
                continue

            if line != original_line:
                corrections.append({"before": original_line, "after": line})

            processed_lines.append(line)

        final_text = "\n".join(processed_lines).strip()

        logger.debug(
            "OCR post-processing completed | in_chars=%d | out_chars=%d | lines=%d | corrections=%d",
            len(original_text),
            len(final_text),
            len(processed_lines),
            len(corrections),
        )

        return ProcessedOCR(
            text=final_text,
            lines=processed_lines,
            corrections=corrections,
        )

    def process_lines(self, lines: list[str]) -> ProcessedOCR:
        """Process an explicit list of OCR recognized lines."""
        return self.process("\n".join(lines))

    @staticmethod
    def _normalize_line_breaks(text: str) -> str:
        """Standardize newline characters."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Collapse multiple horizontal spaces and tabs into a single space."""
        return re.sub(r"[ \t]+", " ", text).strip()

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        """
        Normalize common Arabic character orthographic variants.
        Normalizes:
            - Alif variants: أ, إ, آ, ٱ -> ا
            - Alif Maqsura: ى -> ي
            - Removes tatweel / kashida (ـ)
        Preserves:
            - Taa Marbuta (ة)
            - Arabic diacritics / letters
        """
        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ـ": "",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    @staticmethod
    def _normalize_digits(text: str) -> str:
        """
        Convert Eastern Arabic-Indic numerals (٠-٩) and Persian numerals (۰-۹)
        to Western Arabic digits (0-9) so that prices, phone numbers, and square meters
        are uniformly searchable.
        """
        arabic_indic_table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        persian_table = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        return text.translate(arabic_indic_table).translate(persian_table)

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        """
        Normalize typographic variations in punctuation while preserving meaning.
        """
        replacements = {
            "،": ",",
            "؛": ";",
            "؟": "?",
            "٪": "%",
            "٫": ".",  # Arabic decimal separator to standard period
            "٬": ",",  # Arabic thousands separator to comma
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return text

    @staticmethod
    def _remove_obvious_noise(text: str) -> str:
        """
        Conservatively remove unreadable OCR glyph noise.
        """
        # Lines consisting only of noise symbols like ~~~~~ or |~`_
        if re.fullmatch(r"^[|~`_^=+#\*\-]+$", text):
            return ""

        # Normalize repeated punctuation like ..... or ,,
        text = re.sub(r"([.,;:!?])\1{2,}", r"\1", text)
        return text
