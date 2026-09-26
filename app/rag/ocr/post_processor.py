from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProcessedOCR:
    """
    Cleaned OCR output.

    This is still OCR text.
    Property-specific extraction belongs to normalization.
    """

    text: str
    lines: list[str] = field(default_factory=list)
    corrections: list[dict[str, str]] = field(default_factory=list)


class OCRPostProcessor:
    """
    Post-process OCR text before property normalization.

    Responsibilities:
        - Clean whitespace
        - Normalize Arabic characters
        - Normalize common OCR punctuation
        - Normalize Arabic/Persian digits
        - Remove obvious OCR noise
        - Preserve useful property information

    Does NOT:
        - extract property fields
        - determine price
        - determine area
        - determine bedrooms
        - create embeddings
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
            "OCRPostProcessor initialized | "
            "normalize_arabic=%s | "
            "normalize_digits=%s | "
            "remove_noise=%s",
            normalize_arabic,
            normalize_digits,
            remove_noise,
        )

    def process(
        self,
        text: str,
    ) -> ProcessedOCR:
        """
        Process raw OCR text.
        """

        if not text or not text.strip():
            logger.warning(
                "OCR post-processing skipped: empty text"
            )

            return ProcessedOCR(text="")

        original_text = text

        text = self._normalize_line_breaks(text)

        lines = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        processed_lines: list[str] = []
        corrections: list[dict[str, str]] = []

        for line in lines:

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
                corrections.append(
                    {
                        "before": original_line,
                        "after": line,
                    }
                )

            processed_lines.append(line)

        processed_text = "\n".join(
            processed_lines
        ).strip()

        logger.info(
            "OCR post-processing completed | "
            "original_chars=%d | processed_chars=%d | "
            "lines=%d | corrections=%d",
            len(original_text),
            len(processed_text),
            len(processed_lines),
            len(corrections),
        )

        return ProcessedOCR(
            text=processed_text,
            lines=processed_lines,
            corrections=corrections,
        )

    def process_lines(
        self,
        lines: list[str],
    ) -> ProcessedOCR:
        """
        Process OCR lines directly.
        """

        text = "\n".join(lines)

        return self.process(text)

    @staticmethod
    def _normalize_line_breaks(
        text: str,
    ) -> str:
        """
        Normalize different newline formats.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _clean_whitespace(
        text: str,
    ) -> str:
        """
        Normalize spaces and tabs.
        """

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _normalize_arabic(
        text: str,
    ) -> str:
        """
        Normalize common Arabic character variations.

        Examples:

            أ / إ / آ → ا
            ى         → ي
            ة         → ة

        We intentionally do NOT remove tashkeel because
        aggressive Arabic normalization can sometimes
        damage useful text.
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
            text = text.replace(
                source,
                target,
            )

        return text

    @staticmethod
    def _normalize_digits(
        text: str,
    ) -> str:
        """
        Convert Arabic-Indic and Persian digits
        to standard Western digits.

        Arabic:
            ٠١٢٣٤٥٦٧٨٩

        Persian:
            ۰۱۲۳۴۵۶۷۸۹

        Example:

            ١٥٠ متر
            →
            150 متر
        """

        arabic_digits = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩",
            "0123456789",
        )

        persian_digits = str.maketrans(
            "۰۱۲۳۴۵۶۷۸۹",
            "0123456789",
        )

        text = text.translate(
            arabic_digits
        )

        text = text.translate(
            persian_digits
        )

        return text

    @staticmethod
    def _normalize_punctuation(
        text: str,
    ) -> str:
        """
        Normalize common punctuation variations.
        """

        replacements = {
            "،": ",",
            "؛": ";",
            "؟": "?",
            "٪": "%",
            "٫": ".",
            "٬": ",",
        }

        for source, target in replacements.items():
            text = text.replace(
                source,
                target,
            )

        return text

    @staticmethod
    def _remove_obvious_noise(
        text: str,
    ) -> str:
        """
        Remove extremely obvious OCR noise.

        Keep this conservative.

        We don't want to delete numbers, symbols,
        Arabic words, or property information that
        may actually be useful.
        """

        # Remove isolated repeated symbols.
        text = re.sub(
            r"^[|~`_^]+$",
            "",
            text,
        )

        # Remove repeated punctuation.
        text = re.sub(
            r"([.,;:!?])\1{2,}",
            r"\1",
            text,
        )

        return text.strip()