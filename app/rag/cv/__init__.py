"""
Computer Vision module for the AI Real Estate Assistant.

Responsible for extracting *visual* information from property images,
floor plans, and unit sketches — as opposed to `app/rag/ocr/`, which
handles text extraction from the same kinds of documents.

Typical flow:
    Image / Floor Plan
        -> image_processor.preprocess_image()
        -> vision_extractor.extract_visual_features()  (general property photos)
           OR
        -> floorplan_parser.parse_floorplan()          (floor plans / unit sketches)
        -> combined with OCR output during normalization
"""

from .image_processor import preprocess_image, ImagePreprocessConfig
from .vision_extractor import extract_visual_features, VisualFeatures
from .floorplan_parser import parse_floorplan, FloorplanResult

__all__ = [
    "preprocess_image",
    "ImagePreprocessConfig",
    "extract_visual_features",
    "VisualFeatures",
    "parse_floorplan",
    "FloorplanResult",
]