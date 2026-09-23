"""
floorplan_parser.py

Parses floor plans / unit sketches for *structural* information:
room layout, room count, approximate room boundaries, and overall
shape — complementary to OCR, which pulls labels/numbers (e.g. "3.2m",
"Bedroom 1") from the same image.

Per the project's AI design principles, this module only extracts
structured signals — it does not decide property attributes on its
own. Its output should be combined with OCR text and reconciled by
`app/rag/normalization/` before it's treated as trusted property data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np

from .image_processor import preprocess_image, ImagePreprocessConfig


@dataclass
class RoomRegion:
    """A detected room-like region within the floor plan image."""
    label: Optional[str] = None          # filled in later by OCR/normalization, if available
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h in pixels
    area_px: float = 0.0
    approx_shape: str = "unknown"        # "rectangular" | "irregular"


@dataclass
class FloorplanResult:
    source_file: str
    room_count_estimate: int = 0
    regions: List[RoomRegion] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)  # width, height
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


def _detect_room_regions(gray: np.ndarray) -> List[RoomRegion]:
    """Heuristic wall/room segmentation: threshold + contour detection.

    This is a lightweight geometric baseline suitable for clean, mostly
    line-based floor plans. For noisy/hand-drawn sketches or complex
    architectural drawings, replace this with a trained segmentation
    model (e.g. a U-Net style room-layout model) behind the same
    interface — the rest of the pipeline doesn't need to change.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = gray.shape[0] * gray.shape[1]
    regions: List[RoomRegion] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out noise and the full-page outer contour.
        if area < image_area * 0.01 or area > image_area * 0.9:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        shape = "rectangular" if len(approx) == 4 else "irregular"

        regions.append(RoomRegion(
            bounding_box=(x, y, w, h),
            area_px=area,
            approx_shape=shape,
        ))

    return regions


def parse_floorplan(
    source: Union[str, Path, bytes, np.ndarray],
    source_file: Optional[str] = None,
    preprocess_config: Optional[ImagePreprocessConfig] = None,
) -> FloorplanResult:
    """Extract structural layout information from a floor plan / unit
    sketch image.
    """
    # Floor plans benefit from deskew + grayscale before edge detection.
    config = preprocess_config or ImagePreprocessConfig(
        deskew=True, denoise=True, to_grayscale=True,
    )
    img = preprocess_image(source, config)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions = _detect_room_regions(gray)

    notes = []
    if not regions:
        notes.append(
            "No room regions detected — image may be low quality, "
            "hand-drawn, or not a standard floor plan layout."
        )

    confidence = min(1.0, len(regions) / 6.0) if regions else 0.0

    return FloorplanResult(
        source_file=source_file or (str(source) if isinstance(source, (str, Path)) else "in_memory"),
        room_count_estimate=len(regions),
        regions=regions,
        image_size=(gray.shape[1], gray.shape[0]),
        confidence=confidence,
        notes=notes,
    )