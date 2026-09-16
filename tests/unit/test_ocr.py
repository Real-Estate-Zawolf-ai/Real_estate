from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.rag.ocr.base import BaseOCRProvider
from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.extractor import OCRExtractor
from app.rag.ocr.pdf_processor import PDFOCRProcessor
from app.rag.ocr.post_processor import OCRPostProcessor
from app.rag.ocr.schemas import (
    BlockResult,
    OCRDetection,
    OCRResult,
    OCRStatus,
    PageResult,
)


class MockOCRProvider(BaseOCRProvider):
    """Mock OCR provider for unit tests without needing deep-learning model downloads."""

    def __init__(self, detections: list[OCRDetection] | None = None) -> None:
        self.detections = detections or []

    def is_available(self) -> bool:
        return True

    def read_image(self, image_path: str | Path, **kwargs: Any) -> list[OCRDetection]:
        return self.detections


# =====================================================================
# 1. Schema Tests
# =====================================================================

def test_block_result_schema() -> None:
    block = BlockResult(
        text="شقة للبيع",
        confidence=0.95432,
        bounding_box=[[10, 10], [100, 10], [100, 50], [10, 50]],
    )
    d = block.to_dict()
    assert d["text"] == "شقة للبيع"
    assert d["confidence"] == 0.9543
    assert d["bounding_box"] == [[10, 10], [100, 10], [100, 50], [10, 50]]


def test_page_result_schema() -> None:
    page = PageResult(
        page_number=1,
        text="Villa for sale",
        confidence=0.92,
        language="en",
        blocks=[BlockResult(text="Villa for sale", confidence=0.92)],
    )
    d = page.to_dict()
    assert d["page_number"] == 1
    assert d["text"] == "Villa for sale"
    assert d["confidence"] == 0.92
    assert d["language"] == "en"
    assert len(d["blocks"]) == 1


def test_ocr_result_schema() -> None:
    res = OCRResult(
        source="test.jpg",
        text="Sample text",
        pages=[],
        confidence=0.88,
        status=OCRStatus.SUCCESS,
    )
    d = res.to_dict()
    assert d["source"] == "test.jpg"
    assert d["status"] == "success"
    assert d["confidence"] == 0.88


# =====================================================================
# 2. Post-Processor Tests (Arabic, English, Digits, Noise)
# =====================================================================

def test_arabic_letter_normalization() -> None:
    processor = OCRPostProcessor()
    # أ, إ, آ -> ا ; ى -> ي ; tatweel removed
    raw = "شـقة فاخرة للبيع فى أرقى كمبوند إسلامى"
    processed = processor.process(raw)
    assert "في" in processed.text
    assert "ارقي" in processed.text
    assert "اسلامي" in processed.text
    assert "شـ" not in processed.text  # tatweel removed
    assert len(processed.corrections) > 0


def test_arabic_indic_digit_normalization() -> None:
    processor = OCRPostProcessor()
    # ٠١٢٣٤٥٦٧٨٩ -> 0123456789
    raw = "السعر: ٤,٥٠٠,٠٠٠ جنيه | المساحة: ١٥٠ متر | هاتف: ٠١٠١٢٣٤٥٦٧٨"
    processed = processor.process(raw)
    assert "4,500,000" in processed.text
    assert "150" in processed.text
    assert "01012345678" in processed.text


def test_punctuation_and_noise_removal() -> None:
    processor = OCRPostProcessor()
    raw = "~~~~~\nسعر الوحدة : 2500000 ، شامل الجراج؛\n|||~~~\n"
    processed = processor.process(raw)
    assert "سعر الوحدة : 2500000 , شامل الجراج;" in processed.text
    assert "~~~~~" not in processed.text


# =====================================================================
# 3. Document Parser Tests & Language Detection
# =====================================================================

def test_language_detection() -> None:
    assert DocumentParser.detect_language("شقة فاخرة للبيع") == "ar"
    assert DocumentParser.detect_language("Luxury apartment for sale") == "en"
    assert DocumentParser.detect_language("شقة 3 bedrooms in New Cairo") == "ar_en"
    assert DocumentParser.detect_language("12345 !@#$%") is None


def test_document_parser_confidence_filter() -> None:
    parser = DocumentParser(min_confidence=0.50)
    detections = [
        OCRDetection(text="High confidence", confidence=0.90),
        OCRDetection(text="Low confidence artifact", confidence=0.25),
        OCRDetection(text="Another valid line", confidence=0.80),
    ]
    page = parser.parse_page(detections, page_number=1)
    assert "High confidence" in page.text
    assert "Low confidence artifact" not in page.text
    assert "Another valid line" in page.text
    assert len(page.blocks) == 2
    assert page.confidence == pytest.approx(0.85, 0.01)


# =====================================================================
# 4. Extractor Tests (Validation, Mock Engine, Errors)
# =====================================================================

def test_extractor_missing_file() -> None:
    extractor = OCRExtractor(provider=MockOCRProvider())
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_image("non_existent_file.jpg")


def test_extractor_unsupported_format() -> None:
    extractor = OCRExtractor(provider=MockOCRProvider())
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Unsupported image format"):
            extractor.extract_from_image(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_extractor_success_with_mock() -> None:
    mock_detections = [
        OCRDetection(
            text="مشروع زهرة التجمع",
            confidence=0.96,
            bbox=[[0, 0], [10, 0], [10, 10], [0, 10]],
        ),
        OCRDetection(
            text="Price: 5000000 EGP",
            confidence=0.94,
            bbox=[[0, 20], [10, 20], [10, 30], [0, 30]],
        ),
    ]
    extractor = OCRExtractor(provider=MockOCRProvider(mock_detections))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
        temp_path = f.name

    try:
        res = extractor.extract_from_image(temp_path)
        assert res.status == OCRStatus.SUCCESS
        assert "مشروع زهرة التجمع" in res.text
        assert "Price: 5000000 EGP" in res.text
        assert res.language == "ar_en"
        assert res.confidence > 0.90
        assert len(res.pages) == 1
        assert len(res.pages[0].blocks) == 2
        assert res.pages[0].blocks[0].bounding_box is not None
    finally:
        Path(temp_path).unlink(missing_ok=True)


# =====================================================================
# 5. Scanned PDF Processor Tests
# =====================================================================

def test_pdf_processor_missing_file() -> None:
    processor = PDFOCRProcessor(extractor=OCRExtractor(provider=MockOCRProvider()))
    with pytest.raises(FileNotFoundError):
        processor.process_pdf("missing_doc.pdf")


def test_pdf_processor_unsupported_format() -> None:
    processor = PDFOCRProcessor(extractor=OCRExtractor(provider=MockOCRProvider()))
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Expected a PDF file"):
            processor.process_pdf(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_pdf_text_vs_scanned_handling() -> None:
    """Verify embedded text parsing when PDF contains selectable text."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        pytest.skip("PyMuPDF not installed in environment")

    # Create a small in-memory PDF with text
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Unit A102 in New Cairo with 3 bedrooms and price 4500000 EGP")

    page2 = doc.new_page()
    page2.insert_text((50, 50), "Unit B205 in New Cairo with 2 bedrooms and price 3200000 EGP")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    doc.save(pdf_path)
    doc.close()

    try:
        processor = PDFOCRProcessor(
            extractor=OCRExtractor(provider=MockOCRProvider()),
            min_text_chars_per_page=20,
        )

        assert not processor.is_scanned_pdf(pdf_path)

        res = processor.process_pdf(pdf_path, force_ocr=False)
        assert res.status == OCRStatus.SUCCESS
        assert len(res.pages) == 2
        assert res.pages[0].page_number == 1
        assert "Unit A102" in res.pages[0].text
        assert res.pages[1].page_number == 2
        assert "Unit B205" in res.pages[1].text
    finally:
        Path(pdf_path).unlink(missing_ok=True)
