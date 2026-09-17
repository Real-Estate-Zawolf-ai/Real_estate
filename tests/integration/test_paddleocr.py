from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.rag.ocr.extractor import OCRExtractor
from app.rag.ocr.paddle_provider import PaddleOCRProvider
from app.rag.ocr.schemas import OCRStatus


def test_paddleocr_mocked_end_to_end_pipeline() -> None:
    """
    Simulated integration test verifying the full pipeline with PaddleOCRProvider:
    Image File -> Pillow Validation -> PaddleOCR Engine Execution -> Adapter -> PostProcessor -> Final Document.
    """
    provider = PaddleOCRProvider(lang="ar", use_angle_cls=True)

    # Mock internal PaddleOCR engine
    mock_engine = MagicMock()
    mock_raw_output = [
        [
            [[[10, 10], [200, 10], [200, 40], [10, 40]], ("كمبوند زايد ريزيدنس", 0.982)],
            [[[10, 50], [150, 50], [150, 80], [10, 80]], ("السعر: 4,000,000 ج.م", 0.945)],
            [[[10, 90], [180, 90], [180, 120], [10, 120]], ("Phone: 01099887766", 0.961)],
        ]
    ]
    mock_engine.ocr.return_value = mock_raw_output
    provider._engine = mock_engine

    extractor = OCRExtractor(provider=provider)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name

    try:
        # Create valid test image
        img = Image.new("RGB", (300, 200), color="white")
        img.save(temp_path, format="PNG")

        res = extractor.extract_from_image(temp_path)

        assert res.status == OCRStatus.SUCCESS
        assert len(res.pages) == 1
        assert len(res.pages[0].blocks) == 3

        # Verify Arabic normalization and digit handling
        assert "كمبوند زايد ريزيدنس" in res.text
        assert "السعر: 4,000,000" in res.text
        assert "01099887766" in res.text
        assert res.confidence > 0.95
        assert res.language == "ar_en"
        assert res.pages[0].blocks[0].bounding_box == [[10, 10], [200, 10], [200, 40], [10, 40]]

        # Verify mock engine called with cls=True
        mock_engine.ocr.assert_called_once_with(str(Path(temp_path)), cls=True)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_paddleocr_mocked_in_memory_bytes_pipeline() -> None:
    """Verify that read_image_bytes interacts with the engine correctly."""
    provider = PaddleOCRProvider(lang="ar")
    mock_engine = MagicMock()
    mock_raw_output = [
        [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], ("شقة للإيجار", 0.99)],
        ]
    ]
    mock_engine.ocr.return_value = mock_raw_output
    provider._engine = mock_engine

    buf = io.BytesIO()
    Image.new("RGB", (100, 50), color="white").save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    detections = provider.read_image_bytes(raw_bytes)
    assert len(detections) == 1
    assert detections[0].text == "شقة للإيجار"
    assert detections[0].confidence == pytest.approx(0.99)
    assert mock_engine.ocr.called


@pytest.mark.skipif(not PaddleOCRProvider().is_available(), reason="paddleocr not installed in environment")
def test_paddleocr_live_execution() -> None:
    """Live test executed only when paddleocr and models are present in the environment."""
    provider = PaddleOCRProvider(lang="ar", use_gpu=False)
    assert provider.is_available()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name

    try:
        img = Image.new("RGB", (200, 60), color="white")
        img.save(temp_path, format="PNG")

        extractor = OCRExtractor(provider=provider)
        res = extractor.extract_from_image(temp_path)
        assert res.status in (OCRStatus.SUCCESS, OCRStatus.EMPTY)
    finally:
        Path(temp_path).unlink(missing_ok=True)
