from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OCRDetection:
    """
    Represents one text detection returned by OCR.
    """

    text: str
    confidence: float
    bbox: list[Any] | None = None


@dataclass
class ParsedDocument:
    """
    Structured representation of OCR output.

    This is an intermediate representation.

    It does NOT:
        - normalize property fields
        - extract final price/area/etc.
        - create embeddings
        - store data in Qdrant
    """

    text: str
    detections: list[OCRDetection] = field(default_factory=list)
    language_hint: str | None = None
    average_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """
    Parse and clean OCR output into a structured document.

    Input:
        Raw OCR detections.

    Output:
        ParsedDocument.

    Pipeline position:

        Image
          ↓
        EasyOCR
          ↓
        DocumentParser
          ↓
        PostProcessor
          ↓
        PropertyNormalizer
    """

    def __init__(
        self,
        min_confidence: float = 0.30,
    ) -> None:

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0.0 and 1.0"
            )

        self.min_confidence = min_confidence

        logger.info(
            "DocumentParser initialized | min_confidence=%.2f",
            min_confidence,
        )

    def parse(
        self,
        detections: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """
        Parse OCR detections.

        Expected detection format:

        {
            "text": "Apartment for sale",
            "confidence": 0.94,
            "bbox": [...]
        }

        Returns:
            ParsedDocument
        """

        if not detections:
            logger.warning(
                "Document parsing skipped: no OCR detections"
            )

            return ParsedDocument(
                text="",
                detections=[],
                metadata=metadata or {},
            )

        parsed_detections: list[OCRDetection] = []

        for detection in detections:

            if not isinstance(detection, dict):
                logger.warning(
                    "Skipping invalid OCR detection: %s",
                    detection,
                )
                continue

            text = str(
                detection.get("text", "")
            ).strip()

            confidence = float(
                detection.get("confidence", 0.0)
            )

            bbox = detection.get("bbox")

            if not text:
                continue

            if confidence < self.min_confidence:
                logger.debug(
                    "Skipping low-confidence detection | "
                    "confidence=%.3f | text=%s",
                    confidence,
                    text,
                )
                continue

            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                continue

            parsed_detections.append(
                OCRDetection(
                    text=cleaned_text,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        document_text = self._build_document_text(
            parsed_detections
        )

        average_confidence = (
            self._calculate_average_confidence(
                parsed_detections
            )
        )

        language_hint = self._detect_language(
            document_text
        )

        parsed_document = ParsedDocument(
            text=document_text,
            detections=parsed_detections,
            language_hint=language_hint,
            average_confidence=average_confidence,
            metadata=metadata or {},
        )

        logger.info(
            "Document parsing completed | "
            "detections=%d | characters=%d | "
            "average_confidence=%.3f | language=%s",
            len(parsed_detections),
            len(document_text),
            average_confidence,
            language_hint,
        )

        return parsed_document

    def parse_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """
        Parse already extracted raw text.

        Useful when OCR output is available only as plain text.
        """

        if not text or not text.strip():
            return ParsedDocument(
                text="",
                detections=[],
                metadata=metadata or {},
            )

        cleaned_text = self._clean_text(text)

        language_hint = self._detect_language(
            cleaned_text
        )

        return ParsedDocument(
            text=cleaned_text,
            language_hint=language_hint,
            metadata=metadata or {},
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Basic OCR text cleanup.

        We intentionally keep this lightweight.

        Property-specific normalization belongs to:
            app/rag/normalization/
        """

        text = text.strip()

        # Normalize repeated whitespace.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Remove excessive blank lines.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _build_document_text(
        detections: list[OCRDetection],
    ) -> str:
        """
        Combine OCR detections into document text.

        The current implementation preserves EasyOCR's
        detection order.

        More advanced spatial ordering can be added later
        if property images require it.
        """

        lines = [
            detection.text
            for detection in detections
            if detection.text
        ]

        return "\n".join(lines).strip()

    @staticmethod
    def _calculate_average_confidence(
        detections: list[OCRDetection],
    ) -> float:

        if not detections:
            return 0.0

        total = sum(
            detection.confidence
            for detection in detections
        )

        return total / len(detections)

    @staticmethod
    def _detect_language(
        text: str,
    ) -> str | None:
        """
        Lightweight language hint detection.

        Returns:
            "ar"      Arabic
            "en"      English
            "ar_en"   Mixed Arabic + English
            None      No detectable text
        """

        if not text:
            return None

        arabic_chars = re.findall(
            r"[\u0600-\u06FF]",
            text,
        )

        english_chars = re.findall(
            r"[A-Za-z]",
            text,
        )

        has_arabic = bool(arabic_chars)
        has_english = bool(english_chars)

        if has_arabic and has_english:
            return "ar_en"

        if has_arabic:
            return "ar"

        if has_english:
            return "en"

        return None