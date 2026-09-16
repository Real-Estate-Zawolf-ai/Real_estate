from __future__ import annotations

from app.rag.ocr.base import BaseOCRProvider
from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.extractor import EasyOCRProvider, OCRExtractor
from app.rag.ocr.pdf_processor import PDFOCRProcessor
from app.rag.ocr.post_processor import OCRPostProcessor, ProcessedOCR
from app.rag.ocr.schemas import (
    BlockResult,
    OCRDetection,
    OCRResult,
    OCRStatus,
    PageResult,
)

__all__ = [
    "BaseOCRProvider",
    "EasyOCRProvider",
    "OCRExtractor",
    "DocumentParser",
    "OCRPostProcessor",
    "ProcessedOCR",
    "PDFOCRProcessor",
    "OCRResult",
    "PageResult",
    "BlockResult",
    "OCRDetection",
    "OCRStatus",
]
