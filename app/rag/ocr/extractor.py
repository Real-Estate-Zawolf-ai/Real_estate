from __future__ import annotations

import logging
from pathlib import Path

import easyocr

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


class OCRExtractor:
    """
    OCR extractor using EasyOCR.

    Supported languages:
        - Arabic
        - English

    Responsibilities:
        - Load EasyOCR reader.
        - Validate image files.
        - Extract text from images.
        - Return raw OCR text.

    This class does NOT:
        - normalize property fields
        - extract structured property attributes
        - chunk text
        - create embeddings
        - store data in Qdrant
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        model_storage_directory: str | None = None,
    ) -> None:

        self.languages = languages or ["ar", "en"]
        self.gpu = gpu

        logger.info(
            "Loading OCR model | languages=%s | gpu=%s",
            self.languages,
            self.gpu,
        )

        reader_kwargs = {
            "gpu": self.gpu,
        }

        if model_storage_directory:
            reader_kwargs["model_storage_directory"] = (
                model_storage_directory
            )

        self.reader = easyocr.Reader(
            self.languages,
            **reader_kwargs,
        )

        logger.info(
            "OCR model loaded successfully | languages=%s | gpu=%s",
            self.languages,
            self.gpu,
        )

    def extract_from_image(
        self,
        image_path: str | Path,
    ) -> str:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        logger.info(
            "Starting OCR extraction | file=%s",
            path,
        )

        try:
            results = self.reader.readtext(
                str(path),
                detail=1,
                paragraph=False,
            )

            extracted_lines: list[str] = []

            for result in results:
                if len(result) != 3:
                    continue

                _, text, confidence = result

                text = text.strip()

                if not text:
                    continue

                logger.debug(
                    "OCR detection | text=%s | confidence=%.3f",
                    text,
                    confidence,
                )

                extracted_lines.append(text)

            extracted_text = "\n".join(extracted_lines).strip()

            if not extracted_text:
                logger.warning(
                    "OCR returned empty text | file=%s",
                    path,
                )
                return ""

            logger.info(
                "OCR extraction completed | file=%s | "
                "detections=%d | characters=%d",
                path,
                len(extracted_lines),
                len(extracted_text),
            )

            return extracted_text

        except Exception:
            logger.exception(
                "OCR extraction failed | file=%s",
                path,
            )
            raise

    def extract_with_confidence(
        self,
        image_path: str | Path,
    ) -> list[dict]:
        """
        Extract OCR results together with confidence scores.

        Returns:
            [
                {
                    "text": "...",
                    "confidence": 0.95,
                    "bbox": [...]
                }
            ]
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        logger.info(
            "Starting detailed OCR extraction | file=%s",
            path,
        )

        results = self.reader.readtext(
            str(path),
            detail=1,
            paragraph=False,
        )

        detections = []

        for result in results:
            if len(result) != 3:
                continue

            bbox, text, confidence = result

            text = text.strip()

            if not text:
                continue

            detections.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox,
                }
            )

        logger.info(
            "Detailed OCR completed | file=%s | detections=%d",
            path,
            len(detections),
        )

        return detections