from __future__ import annotations

import logging
import re
from typing import Any

from app.rag.ocr.post_processor import OCRPostProcessor
from app.rag.ocr.schemas import BlockResult, OCRDetection, PageResult

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parses and cleans low-level OCR detections into structured PageResult instances.

    Responsibilities:
        - Confidence threshold filtering (drop low-confidence OCR artifacts).
        - Optional text post-processing (Arabic normalization, digit conversion).
        - Aggregation of detection blocks into coherent page text.
        - Average confidence calculation across recognized blocks.
        - Language detection hint ('ar', 'en', 'ar_en').
        - Document-level final pass to clean cross-block artifacts.
    """

    def __init__(
        self,
        min_confidence: float = 0.30,
        post_processor: OCRPostProcessor | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self.min_confidence = min_confidence
        self.post_processor = post_processor or OCRPostProcessor()

        logger.info(
            "DocumentParser initialized | min_confidence=%.2f",
            min_confidence,
        )

    def parse_page(
        self,
        detections: list[OCRDetection | dict[str, Any]],
        page_number: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> PageResult:
        """
        Parse raw OCR detections for a single document page into a PageResult.

        Args:
            detections: List of OCRDetection objects or dictionaries with keys 'text', 'confidence', 'bbox'.
            page_number: 1-indexed page number.
            metadata: Optional page-level metadata.

        Returns:
            PageResult containing structured blocks, full text, and confidence.
        """
        meta = metadata or {}
        if not detections:
            return PageResult(
                page_number=page_number,
                text="",
                confidence=0.0,
                language=None,
                blocks=[],
                metadata=meta,
            )

        valid_blocks: list[BlockResult] = []

        for item in detections:
            # Handle both dataclass and dictionary formats
            if isinstance(item, OCRDetection):
                raw_text = item.text
                confidence = float(item.confidence)
                bbox = item.bbox
            elif isinstance(item, dict):
                raw_text = str(item.get("text", "")).strip()
                confidence = float(item.get("confidence", 0.0))
                bbox = item.get("bbox") or item.get("bounding_box")
            else:
                logger.warning("Skipping unknown detection type: %s", type(item))
                continue

            if not raw_text or not raw_text.strip():
                continue

            if confidence < self.min_confidence:
                logger.debug(
                    "Dropping low confidence detection | text=%s | conf=%.3f < min=%.3f",
                    raw_text[:50],
                    confidence,
                    self.min_confidence,
                )
                continue

            # Run post processing on block text
            processed = self.post_processor.process(raw_text)
            clean_text = processed.text

            if not clean_text:
                continue

            valid_blocks.append(
                BlockResult(
                    text=clean_text,
                    confidence=confidence,
                    bounding_box=bbox,
                )
            )

        # Assemble full page text
        page_lines = [block.text for block in valid_blocks if block.text]
        raw_full_text = "\n".join(page_lines).strip()

        # Document-level final pass: clean cross-block whitespace artifacts
        # without altering meaning.  This catches broken words and extra
        # whitespace that per-block processing cannot see.
        full_text = self._final_document_pass(raw_full_text)

        # Compute page confidence
        avg_confidence = (
            sum(block.confidence for block in valid_blocks) / len(valid_blocks)
            if valid_blocks
            else 0.0
        )

        language_hint = self.detect_language(full_text)

        return PageResult(
            page_number=page_number,
            text=full_text,
            confidence=avg_confidence,
            language=language_hint,
            blocks=valid_blocks,
            metadata=meta,
        )

    def parse_plain_text(
        self,
        text: str,
        page_number: int = 1,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> PageResult:
        """
        Create a PageResult from plain text (e.g. from existing parser fallback).
        """
        processed = self.post_processor.process(text)
        clean_text = processed.text
        lines = [line for line in clean_text.split("\n") if line.strip()]
        blocks = [BlockResult(text=line, confidence=confidence) for line in lines]

        return PageResult(
            page_number=page_number,
            text=clean_text,
            confidence=confidence if clean_text else 0.0,
            language=self.detect_language(clean_text),
            blocks=blocks,
            metadata=metadata or {},
        )

    def _final_document_pass(self, text: str) -> str:
        """
        Lightweight document-level post-processing pass on assembled full text.

        Handles cross-block artifacts that per-block processing cannot see:
        - Collapses excessive blank lines.
        - Strips trailing whitespace per line.
        - Normalizes line boundaries.

        Does NOT apply content-altering transformations to preserve meaning.
        """
        if not text:
            return text

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3+ consecutive newlines into double newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines).strip()

        return text

    @staticmethod
    def detect_language(text: str) -> str | None:
        """
        Detect language hint from text content.

        Returns:
            - 'ar': Arabic characters present, no English
            - 'en': English characters present, no Arabic
            - 'ar_en': Mixed Arabic and English
            - None: No recognizable linguistic characters
        """
        if not text:
            return None

        has_arabic = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))
        has_english = bool(re.search(r"[A-Za-z]", text))

        if has_arabic and has_english:
            return "ar_en"
        if has_arabic:
            return "ar"
        if has_english:
            return "en"

        return None
