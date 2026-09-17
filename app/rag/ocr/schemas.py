from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OCRStatus(str, Enum):
    """Status of the OCR extraction process."""
    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass
class BlockResult:
    """
    Represents a recognized text segment or line with its confidence
    and spatial bounding box.
    """
    text: str
    confidence: float
    bounding_box: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box,
        }


@dataclass
class PageResult:
    """
    Represents the OCR results of a single document page or image.
    """
    page_number: int
    text: str
    confidence: float = 0.0
    language: str | None = None
    blocks: list[BlockResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "language": self.language,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
        }


@dataclass
class OCRResult:
    """
    Complete structured output of an OCR extraction job across all pages.
    """
    source: str
    text: str
    pages: list[PageResult] = field(default_factory=list)
    language: str | None = None
    confidence: float = 0.0
    status: OCRStatus = OCRStatus.SUCCESS
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "text": self.text,
            "pages": [p.to_dict() for p in self.pages],
            "language": self.language,
            "confidence": round(self.confidence, 4),
            "status": self.status.value if isinstance(self.status, OCRStatus) else str(self.status),
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class OCRDetection:
    """
    Low-level raw detection item from an underlying OCR engine.
    """
    text: str
    confidence: float
    bbox: list[Any] | None = None
