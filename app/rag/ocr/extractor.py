from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.rag.ocr.base import BaseOCRProvider
from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.schemas import (
    OCRDetection,
    OCRResult,
    OCRStatus,
    PageResult,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


class EasyOCRProvider(BaseOCRProvider):
    """
    Default OCR provider using EasyOCR for Arabic and English recognition.
    Loads the underlying reader lazily to conserve startup resources.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        model_storage_directory: str | None = None,
    ) -> None:
        self.languages = languages or ["ar", "en"]
        self.gpu = gpu
        self.model_storage_directory = model_storage_directory
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise ImportError(
                    "easyocr is not installed. Please install it using 'pip install easyocr'."
                ) from exc

            logger.info(
                "Initializing EasyOCR reader | languages=%s | gpu=%s",
                self.languages,
                self.gpu,
            )
            kwargs: dict[str, Any] = {"gpu": self.gpu}
            if self.model_storage_directory:
                kwargs["model_storage_directory"] = self.model_storage_directory

            self._reader = easyocr.Reader(self.languages, **kwargs)
        return self._reader

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def read_image(
        self,
        image_path: str | Path,
        **kwargs: Any,
    ) -> list[OCRDetection]:
        reader = self._get_reader()
        path_str = str(image_path)

        logger.debug("Running EasyOCR readtext on: %s", path_str)
        # detail=1 returns list of (bbox, text, confidence)
        results = reader.readtext(path_str, detail=1, paragraph=False)

        detections: list[OCRDetection] = []
        for item in results:
            if len(item) == 3:
                bbox, text, conf = item
                if text and str(text).strip():
                    detections.append(
                        OCRDetection(
                            text=str(text).strip(),
                            confidence=float(conf),
                            bbox=bbox,
                        )
                    )
        return detections


class OCRExtractor:
    """
    High-level OCR extraction service.

    Supported formats:
        - JPG / JPEG
        - PNG
        - WEBP
        - BMP
        - TIFF / TIF

    Features:
        - Pluggable OCR provider (EasyOCR by default, mockable for tests).
        - Confidence score tracking.
        - Bounding box extraction.
        - Structured OCRResult with language hint and status.
        - Robust error and edge-case handling.
    """

    def __init__(
        self,
        provider: BaseOCRProvider | None = None,
        languages: list[str] | None = None,
        gpu: bool = False,
        min_confidence: float = 0.30,
        model_storage_directory: str | None = None,
    ) -> None:
        self.languages = languages or ["ar", "en"]
        self.gpu = gpu
        self.provider = provider or EasyOCRProvider(
            languages=self.languages,
            gpu=self.gpu,
            model_storage_directory=model_storage_directory,
        )
        self.parser = DocumentParser(min_confidence=min_confidence)

        logger.info(
            "OCRExtractor initialized | provider=%s | languages=%s | gpu=%s | min_conf=%.2f",
            self.provider.__class__.__name__,
            self.languages,
            self.gpu,
            min_confidence,
        )

    def extract_from_image(
        self,
        image_path: str | Path,
        extra_metadata: dict[str, Any] | None = None,
    ) -> OCRResult:
        """
        Extract structured OCR result from an image file.

        Args:
            image_path: Path to the image file.
            extra_metadata: Optional custom metadata to associate with result.

        Returns:
            OCRResult object with extracted text, blocks, confidence, and status.
        """
        path = Path(image_path)
        metadata = dict(extra_metadata or {})
        metadata["file_path"] = str(path)
        metadata["file_name"] = path.name

        # 1. Validation
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        # 2. Corrupted file check (if Pillow is installed)
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
                metadata["image_format"] = img.format
                metadata["image_size"] = img.size
        except ImportError:
            pass
        except Exception as exc:
            logger.error("Image integrity verification failed for %s: %s", path, exc)
            return OCRResult(
                source=str(path),
                text="",
                pages=[],
                status=OCRStatus.FAILED,
                errors=[f"Corrupted image file: {exc}"],
                metadata=metadata,
            )

        # 3. Perform OCR
        try:
            detections = self.provider.read_image(path)
        except Exception as exc:
            logger.exception("OCR provider failure on image: %s", path)
            return OCRResult(
                source=str(path),
                text="",
                pages=[],
                status=OCRStatus.FAILED,
                errors=[f"OCR engine execution error: {exc}"],
                metadata=metadata,
            )

        # 4. Parse Detections
        page_res: PageResult = self.parser.parse_page(
            detections=detections,
            page_number=1,
            metadata={"source_file": path.name},
        )

        status = OCRStatus.SUCCESS if page_res.text else OCRStatus.EMPTY

        return OCRResult(
            source=str(path),
            text=page_res.text,
            pages=[page_res],
            language=page_res.language,
            confidence=page_res.confidence,
            status=status,
            metadata=metadata,
        )

    def extract_text(self, image_path: str | Path) -> str:
        """
        Convenience method to retrieve plain text directly from an image.
        """
        result = self.extract_from_image(image_path)
        return result.text

    def extract_with_confidence(
        self,
        image_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Extract detections as dictionaries with text, confidence, and bbox.
        """
        result = self.extract_from_image(image_path)
        if not result.pages:
            return []
        return [b.to_dict() for b in result.pages[0].blocks]
