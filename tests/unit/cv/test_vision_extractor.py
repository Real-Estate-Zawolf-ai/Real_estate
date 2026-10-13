# Needs the offline ingestion extras (requirements-ingest.txt); skipped on the lean API install.
import pytest as _pytest
_pytest.importorskip("cv2")

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.rag.cv.vision_extractor import extract_visual_features, VisualFeatures

@pytest.fixture
def mock_image():
    return np.full((100, 100, 3), 255, dtype=np.uint8)

def test_extract_visual_features_logic(mock_image):
    """
    Test the orchestration logic of extract_visual_features by mocking
    the internal vision model calls.
    """
    mock_result = {
        "room_type": "living room",
        "condition": "good condition",
        "style_tags": ["modern", "furnished"],
        "detected_amenities": ["balcony"],
        "caption": "A modern living room with a balcony",
        "confidence": 0.85,
        "_scores": {
            "room_type": {"living room": 0.85, "kitchen": 0.15},
            "condition": {"good condition": 0.7, "new": 0.3},
        },
    }

    # We patch _call_vision_model to avoid loading real weights
    with patch("app.rag.cv.vision_extractor._call_vision_model", return_value=mock_result):
        features = extract_visual_features(mock_image, source_file="test.jpg")

        assert isinstance(features, VisualFeatures)
        assert features.source_file == "test.jpg"
        assert features.room_type == "living room"
        assert features.condition == "good condition"
        assert "modern" in features.style_tags
        assert "balcony" in features.detected_amenities
        assert features.caption == "A modern living room with a balcony"
        assert features.confidence == 0.85
        assert features.raw_model_output == mock_result

def test_extract_visual_features_in_memory(mock_image):
    """Test that when no filename is provided, it defaults to 'in_memory'."""
    mock_result = {
        "room_type": "bedroom",
        "condition": "new",
        "style_tags": [],
        "detected_amenities": [],
        "caption": "A new bedroom",
        "confidence": 0.9,
    }

    with patch("app.rag.cv.vision_extractor._call_vision_model", return_value=mock_result):
        features = extract_visual_features(mock_image)
        assert features.source_file == "in_memory"

def test_extract_visual_features_batch(mock_image):
    """Test the batch wrapper."""
    mock_result = {
        "room_type": "kitchen",
        "condition": "good condition",
        "style_tags": [],
        "detected_amenities": [],
        "caption": "A kitchen",
        "confidence": 0.7,
    }

    with patch("app.rag.cv.vision_extractor._call_vision_model", return_value=mock_result):
        from app.rag.cv.vision_extractor import extract_visual_features_batch
        results = extract_visual_features_batch([mock_image, mock_image])

        assert len(results) == 2
        assert results[0].room_type == "kitchen"
        assert results[1].room_type == "kitchen"
