from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.rag.ocr.schemas import OCRDetection


class BaseOCRProvider(ABC):
    """
    Abstract base class for OCR engines (e.g., EasyOCR, Tesseract, Cloud).
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

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the OCR backend dependencies and models are available."""
        pass
