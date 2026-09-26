"""
vision_extractor.py

Extracts *visual* information from general property photos (interiors,
exteriors, amenities) — room type, condition/quality cues, and a
description usable for embeddings / search.

This does NOT read text (that's `app/rag/ocr/`) and does NOT parse
structural floor-plan layouts (that's `floorplan_parser.py`).

Backend: fully open-source, runs locally via `transformers`.
  - Captioning: Salesforce BLIP (image-to-text)
  - Attributes (room_type / condition / style / amenities): CLIP
    zero-shot classification against candidate label lists — no
    training needed, just a labeled vocabulary you control.

Models load once (lazily, on first call) and are cached in the module.
Swap MODEL_NAMES below to try other open checkpoints (e.g. a LLaVA
variant for richer captions) without touching the rest of the file.

Install:
    pip install transformers torch pillow
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from .image_processor import preprocess_image, to_pil, ImagePreprocessConfig


@dataclass
class VisualFeatures:
    """Normalized output of visual extraction for one property image."""
    source_file: str
    room_type: Optional[str] = None          # e.g. "living_room", "kitchen", "bedroom", "exterior"
    condition: Optional[str] = None          # e.g. "new", "good", "needs_renovation"
    style_tags: List[str] = field(default_factory=list)   # e.g. ["modern", "furnished"]
    detected_amenities: List[str] = field(default_factory=list)  # e.g. ["pool", "balcony"]
    caption: Optional[str] = None            # natural-language description, embeddable
    confidence: float = 0.0
    raw_model_output: Optional[dict] = None  # kept for traceability / debugging


MODEL_NAMES = {
    "caption": "Salesforce/blip-image-captioning-base",
    "clip": "openai/clip-vit-base-patch32",
}

# Candidate label vocabularies for CLIP zero-shot classification.
# Edit these to match your actual property taxonomy / CRM fields.
ROOM_TYPE_LABELS = [
    "living room", "bedroom", "kitchen", "bathroom",
    "dining room", "balcony", "building exterior", "swimming pool area",
]
CONDITION_LABELS = ["new / like new", "good condition", "needs renovation"]
STYLE_LABELS = ["modern", "classic", "minimalist", "luxurious", "furnished", "unfurnished"]
AMENITY_LABELS = [
    "swimming pool", "garden", "balcony", "parking garage",
    "elevator", "gym", "security gate", "sea view",
]

# CLIP similarity threshold above which a style/amenity tag is kept
# (style_tags and amenities can have multiple hits; room_type/condition
# always take the single best match).
_TAG_THRESHOLD = 0.22

_models: Dict[str, object] = {}


def _get_models():
    """Lazily load and cache the BLIP + CLIP models/processors."""
    if _models:
        return _models

    import torch
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        CLIPProcessor, CLIPModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    blip_processor = BlipProcessor.from_pretrained(MODEL_NAMES["caption"])
    blip_model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAMES["caption"]
    ).to(device)

    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAMES["clip"])
    clip_model = CLIPModel.from_pretrained(MODEL_NAMES["clip"]).to(device)

    _models.update({
        "device": device,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "clip_processor": clip_processor,
        "clip_model": clip_model,
    })
    return _models


def _caption_image(pil_image) -> str:
    import torch
    m = _get_models()
    inputs = m["blip_processor"](pil_image, return_tensors="pt").to(m["device"])
    with torch.no_grad():
        out = m["blip_model"].generate(**inputs, max_new_tokens=40)
    return m["blip_processor"].decode(out[0], skip_special_tokens=True)


def _clip_scores(pil_image, labels: List[str]) -> List[float]:
    """Return CLIP cosine-similarity-based probabilities for each label,
    normalized via softmax over this specific label set (a fresh
    zero-shot classification per attribute group).
    """
    import torch
    m = _get_models()
    inputs = m["clip_processor"](
        text=labels, images=pil_image, return_tensors="pt", padding=True
    ).to(m["device"])
    with torch.no_grad():
        outputs = m["clip_model"](**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    return probs.tolist()


def _call_vision_model(pil_image) -> dict:
    """Runs BLIP captioning + CLIP zero-shot attribute classification and
    returns the normalized dict shape expected by `extract_visual_features`.
    """
    caption = _caption_image(pil_image)

    room_scores = _clip_scores(pil_image, ROOM_TYPE_LABELS)
    condition_scores = _clip_scores(pil_image, CONDITION_LABELS)
    style_scores = _clip_scores(pil_image, STYLE_LABELS)
    amenity_scores = _clip_scores(pil_image, AMENITY_LABELS)

    room_idx = int(np.argmax(room_scores))
    condition_idx = int(np.argmax(condition_scores))

    style_tags = [
        label for label, score in zip(STYLE_LABELS, style_scores)
        if score >= _TAG_THRESHOLD
    ]
    detected_amenities = [
        label for label, score in zip(AMENITY_LABELS, amenity_scores)
        if score >= _TAG_THRESHOLD
    ]

    # Overall confidence: how peaked the room-type distribution is —
    # a reasonable single scalar signal for how "sure" this pass was.
    confidence = float(room_scores[room_idx])

    return {
        "room_type": ROOM_TYPE_LABELS[room_idx],
        "condition": CONDITION_LABELS[condition_idx],
        "style_tags": style_tags,
        "detected_amenities": detected_amenities,
        "caption": caption,
        "confidence": confidence,
        "_scores": {
            "room_type": dict(zip(ROOM_TYPE_LABELS, room_scores)),
            "condition": dict(zip(CONDITION_LABELS, condition_scores)),
        },
    }


def extract_visual_features(
    source: Union[str, Path, bytes, np.ndarray],
    source_file: Optional[str] = None,
    preprocess_config: Optional[ImagePreprocessConfig] = None,
) -> VisualFeatures:
    """Run preprocessing + the vision model on a single property image and
    return normalized VisualFeatures for downstream normalization/indexing.
    """
    # Property photos should generally keep color + not be deskewed
    # aggressively — override defaults sensibly for this use case.
    config = preprocess_config or ImagePreprocessConfig(
        deskew=False, to_grayscale=False,
    )
    img = preprocess_image(source, config)
    pil_img = to_pil(img)

    result = _call_vision_model(pil_img)

    return VisualFeatures(
        source_file=source_file or (str(source) if isinstance(source, (str, Path)) else "in_memory"),
        room_type=result.get("room_type"),
        condition=result.get("condition"),
        style_tags=result.get("style_tags", []),
        detected_amenities=result.get("detected_amenities", []),
        caption=result.get("caption"),
        confidence=float(result.get("confidence", 0.0)),
        raw_model_output=result,
    )


def extract_visual_features_batch(
    sources: List[Union[str, Path, bytes, np.ndarray]],
) -> List[VisualFeatures]:
    """Convenience batch wrapper. Swap for true batched inference later
    if your vision backend supports it — kept simple/sequential for now
    to match the MVP's "deterministic, traceable" design principle.
    """
    return [extract_visual_features(s) for s in sources]