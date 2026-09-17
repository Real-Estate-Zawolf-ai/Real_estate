from __future__ import annotations

import io
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import Settings
from app.rag.ocr.base import BaseOCRProvider
from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.extractor import EasyOCRProvider, OCRExtractor
from app.rag.ocr.paddle_provider import PaddleOCRProvider
from app.rag.ocr.pdf_processor import PDFOCRProcessor
from app.rag.ocr.post_processor import OCRPostProcessor
from app.rag.ocr.provider_factory import create_ocr_provider
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
        self.read_image_called = 0
        self.read_image_bytes_called = 0

    def is_available(self) -> bool:
        return True

    def read_image(self, image_path: str | Path, **kwargs: Any) -> list[OCRDetection]:
        self.read_image_called += 1
        return self.detections

    def read_image_bytes(self, image_bytes: bytes, **kwargs: Any) -> list[OCRDetection]:
        self.read_image_bytes_called += 1
        return super().read_image_bytes(image_bytes, **kwargs)


def _create_dummy_image_file(path: str | Path, width: int = 100, height: int = 100) -> None:
    """Create a minimal valid PNG image file on disk using Pillow."""
    img = Image.new("RGB", (width, height), color="white")
    img.save(path, format="PNG")


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
# 2. Post-Processor Tests (Arabic, English, Digits, Noise, Flags)
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


def test_post_processor_toggle_flags() -> None:
    # Disable Arabic normalization
    p_no_ar = OCRPostProcessor(normalize_arabic=False)
    assert p_no_ar.process("أرقى فى").text == "أرقى فى"

    # Disable Digit normalization
    p_no_digits = OCRPostProcessor(normalize_digits=False)
    assert "٤,٥٠٠" in p_no_digits.process("٤,٥٠٠").text

    # Disable noise removal
    p_no_noise = OCRPostProcessor(remove_noise=False)
    assert "~~~~~" in p_no_noise.process("~~~~~").text


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


def test_document_parser_final_document_pass() -> None:
    parser = DocumentParser()
    raw_text = "Page header line   \r\n\r\n\r\n\r\nPage content line with spaces   "
    cleaned = parser._final_document_pass(raw_text)

    # Verify 3+ newlines collapsed to 2, trailing spaces stripped, and CRLF normalized
    assert "\r\n" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Page header line\n\nPage content line with spaces" in cleaned

    # Also test via parse_page
    detections = [
        OCRDetection(text="Line 1   ", confidence=0.9),
        OCRDetection(text="Line 2   ", confidence=0.9),
    ]
    page = parser.parse_page(detections)
    assert page.text == "Line 1\nLine 2"


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


def test_extractor_corrupted_image() -> None:
    extractor = OCRExtractor(provider=MockOCRProvider())
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"not a valid png file at all")
        temp_path = f.name
    try:
        res = extractor.extract_from_image(temp_path)
        assert res.status == OCRStatus.FAILED
        assert any("Corrupted image file" in err for err in res.errors)
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
        temp_path = f.name

    try:
        _create_dummy_image_file(temp_path)
        res = extractor.extract_from_image(temp_path)
        assert res.status == OCRStatus.SUCCESS
        assert "مشروع زهرة التجمع" in res.text
        assert "Price: 5000000 EGP" in res.text
        assert res.language == "ar_en"
        assert res.confidence > 0.90
        assert len(res.pages) == 1
        assert len(res.pages[0].blocks) == 2
        assert res.pages[0].blocks[0].bounding_box is not None
        assert res.metadata.get("image_format") == "PNG"
    finally:
        Path(temp_path).unlink(missing_ok=True)


# =====================================================================
# 5. Base Provider & In-Memory Bytes
# =====================================================================

def test_base_ocr_provider_read_image_bytes_default() -> None:
    provider = MockOCRProvider([OCRDetection(text="Byte Test", confidence=0.99)])
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), color="blue").save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Calling read_image_bytes should write to a temp file and call read_image
    detections = provider.read_image_bytes(raw_bytes)
    assert len(detections) == 1
    assert detections[0].text == "Byte Test"
    assert provider.read_image_called == 1


# =====================================================================
# 6. Provider Factory & Configuration Tests
# =====================================================================

def test_provider_factory_paddleocr() -> None:
    provider = create_ocr_provider(provider_name="paddleocr")
    assert isinstance(provider, PaddleOCRProvider)
    assert provider.lang == "ar"


def test_provider_factory_easyocr() -> None:
    provider = create_ocr_provider(provider_name="easyocr")
    assert isinstance(provider, EasyOCRProvider)


def test_provider_factory_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown OCR provider 'unknown'"):
        create_ocr_provider(provider_name="unknown")


def test_provider_factory_with_settings() -> None:
    settings = Settings(
        OCR_PROVIDER="paddleocr",
        OCR_GPU=True,
        OCR_MODEL_DIR="/tmp/models",
    )
    provider = create_ocr_provider(settings=settings)
    assert isinstance(provider, PaddleOCRProvider)
    assert provider.use_gpu is True
    assert provider.model_dir == "/tmp/models"


# =====================================================================
# 7. PaddleOCR Provider (Adapter & Lazy Loading)
# =====================================================================

def test_paddle_ocr_provider_adapter_standard() -> None:
    raw_paddle_output = [
        [
            [[[10.0, 10.0], [100.0, 10.0], [100.0, 40.0], [10.0, 40.0]], ("شقة للبيع", 0.985)],
            [[[10.0, 50.0], [120.0, 50.0], [120.0, 80.0], [10.0, 80.0]], ("3 Rooms New Cairo", 0.942)],
        ]
    ]
    detections = PaddleOCRProvider._adapt_results(raw_paddle_output)
    assert len(detections) == 2
    assert detections[0].text == "شقة للبيع"
    assert detections[0].confidence == pytest.approx(0.985, 0.001)
    assert detections[0].bbox == [[10, 10], [100, 10], [100, 40], [10, 40]]
    assert detections[1].text == "3 Rooms New Cairo"


def test_paddle_ocr_provider_adapter_empty() -> None:
    assert PaddleOCRProvider._adapt_results(None) == []
    assert PaddleOCRProvider._adapt_results([]) == []
    assert PaddleOCRProvider._adapt_results([[]]) == []
    assert PaddleOCRProvider._adapt_results([[None]]) == []


def test_paddle_ocr_provider_missing_dependency() -> None:
    provider = PaddleOCRProvider()
    with patch.dict("sys.modules", {"paddleocr": None}):
        assert provider.is_available() is False
        with pytest.raises(ImportError, match="paddleocr is not installed"):
            provider._get_engine()


def test_paddle_ocr_thread_safe_init() -> None:
    provider = PaddleOCRProvider()
    mock_engine = MagicMock()
    mock_paddle_cls = MagicMock(return_value=mock_engine)

    with patch.dict("sys.modules", {"paddleocr": MagicMock(PaddleOCR=mock_paddle_cls)}):
        engines = []

        def worker():
            eng = provider._get_engine()
            engines.append(eng)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Ensure single instantiation across multiple threads
        assert len(engines) == 10
        assert all(eng is mock_engine for eng in engines)
        mock_paddle_cls.assert_called_once()


# =====================================================================
# 8. Scanned PDF Processor Tests (In-Memory Rendering & Fitz Cleanup)
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


def test_pdf_processor_force_ocr_uses_read_image_bytes() -> None:
    """Verify force_ocr renders pages in memory and calls read_image_bytes."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")

    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.new_page(width=200, height=200)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    doc.save(pdf_path)
    doc.close()

    mock_provider = MockOCRProvider([OCRDetection(text="Scanned Page Content", confidence=0.95)])
    extractor = OCRExtractor(provider=mock_provider)
    processor = PDFOCRProcessor(extractor=extractor, render_dpi=72)

    try:
        res = processor.process_pdf(pdf_path, force_ocr=True)
        assert res.status == OCRStatus.SUCCESS
        assert len(res.pages) == 2
        # Verify read_image_bytes was called twice (once per page)
        assert mock_provider.read_image_bytes_called == 2
        assert "Scanned Page Content" in res.pages[0].text
    finally:
        Path(pdf_path).unlink(missing_ok=True)
