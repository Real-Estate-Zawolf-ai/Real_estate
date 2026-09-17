from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.rag.ocr.schemas import OCRDetection


class BaseOCRProvider(ABC):
    """
    Abstract base class for OCR engines (e.g., EasyOCR, PaddleOCR, Tesseract, Cloud).
    Allows swapping or mocking the OCR engine cleanly without changing
    the consumer business logic.
    """

    @abstractmethod
    def read_image(
        self,
        image_path: str | Path,
        **kwargs: Any,
    ) -> list[OCRDetection]:
        """
        Extract text detections from an image path.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            List of OCRDetection instances containing text, confidence, and bbox.
        """
        pass

    def read_image_bytes(
        self,
        buffer: bytes,
        **kwargs: Any,
    ) -> list[OCRDetection]:
        """
        Extract text detections from in-memory image bytes.

        The default implementation writes to a temporary file and delegates
        to ``read_image()``.  Providers that support in-memory processing
        (e.g. via numpy arrays) should override this for better performance.

        Args:
            buffer: Raw image bytes (e.g. PNG or JPEG encoded).

        Returns:
            List of OCRDetection instances containing text, confidence, and bbox.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(buffer)
            tmp_path = Path(tmp.name)

        try:
            return self.read_image(tmp_path, **kwargs)
        finally:
            tmp_path.unlink(missing_ok=True)

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the OCR backend dependencies and models are available."""
        pass

