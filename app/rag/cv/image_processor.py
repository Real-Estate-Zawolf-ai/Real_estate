"""
image_processor.py

Shared, low-level image preprocessing used by both `vision_extractor`
(property photos) and `floorplan_parser` (floor plans / unit sketches)
before either is fed to OCR or a CV model.

Kept dependency-light: OpenCV + Pillow only. No ML models here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImagePreprocessConfig:
    """Tunable preprocessing options.

    max_dimension: downscale so the longer side never exceeds this
        (keeps CV/OCR calls fast and cost-predictable).
    denoise: apply light denoising — helps scanned/low-quality uploads.
    deskew: attempt automatic rotation correction (useful for scanned
        contracts / floor plans photographed at an angle).
    to_grayscale: convert to grayscale (usually only wanted before OCR,
        NOT before a vision model that needs color/layout cues).
    """
    max_dimension: int = 2000
    denoise: bool = True
    deskew: bool = True
    to_grayscale: bool = False


def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Load an image from a path, raw bytes, or an existing array into a
    BGR numpy array (OpenCV convention).
    """
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at path: {source}")
        return img

    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return img

    raise TypeError(f"Unsupported image source type: {type(source)}")


def _resize_if_needed(img: np.ndarray, max_dimension: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return img
    scale = max_dimension / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Estimate skew angle from the largest text/line contours and rotate
    to correct it. Conservative: only rotates for small angles typical of
    scanned documents / photographed floor plans, not general photos.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return img  # not enough signal to estimate skew reliably

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5 or abs(angle) > 15:
        return img  # ignore negligible or implausible angles

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(
    source: Union[str, Path, bytes, np.ndarray],
    config: Optional[ImagePreprocessConfig] = None,
) -> np.ndarray:
    """Run the standard preprocessing pipeline and return a BGR numpy array
    ready for OCR or a vision model.
    """
    config = config or ImagePreprocessConfig()
    img = load_image(source)
    img = _resize_if_needed(img, config.max_dimension)

    if config.deskew:
        img = _deskew(img)
    if config.denoise:
        img = _denoise(img)
    if config.to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def to_pil(img: np.ndarray) -> Image.Image:
    """Convert a BGR (or grayscale) numpy array to a PIL Image (RGB),
    for handing off to libraries/APIs that expect PIL/RGB input.
    """
    if img.ndim == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)