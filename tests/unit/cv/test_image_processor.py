# Needs the offline ingestion extras (requirements-ingest.txt); skipped on the lean API install.
import pytest as _pytest
_pytest.importorskip("cv2")
_pytest.importorskip("PIL")

import pytest
import numpy as np
import cv2
from PIL import Image
from app.rag.cv.image_processor import (
    load_image,
    preprocess_image,
    to_pil,
    ImagePreprocessConfig,
)

@pytest.fixture
def sample_image_array():
    """Create a simple 100x100 BGR image."""
    return np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)

@pytest.fixture
def sample_image_path(tmp_path, sample_image_array):
    """Save a sample image to a temporary file."""
    path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(path), sample_image_array)
    return str(path)

def test_load_image_array(sample_image_array):
    img = load_image(sample_image_array)
    assert np.array_equal(img, sample_image_array)

def test_load_image_path(sample_image_path):
    img = load_image(sample_image_path)
    assert img is not None
    assert img.shape == (100, 100, 3)

def test_load_image_bytes(sample_image_array):
    _, buffer = cv2.imencode(".jpg", sample_image_array)
    bytes_data = buffer.tobytes()
    img = load_image(bytes_data)
    assert img is not None
    # Note: JPEG compression might change pixels slightly, but shape should match
    assert img.shape == (100, 100, 3)

def test_load_image_invalid_path():
    with pytest.raises(ValueError, match="Could not read image at path"):
        load_image("non_existent_file.jpg")

def test_preprocess_image_grayscale(sample_image_array):
    config = ImagePreprocessConfig(to_grayscale=True, deskew=False, denoise=False)
    img = preprocess_image(sample_image_array, config)
    assert img.ndim == 2  # Grayscale should be 2D

def test_preprocess_image_resize(sample_image_array):
    # Create a larger image
    large_img = np.full((3000, 1000, 3), (255, 0, 0), dtype=np.uint8)
    config = ImagePreprocessConfig(max_dimension=1000, deskew=False, denoise=False)
    img = preprocess_image(large_img, config)
    assert max(img.shape[:2]) <= 1000

def test_to_pil_rgb(sample_image_array):
    pil_img = to_pil(sample_image_array)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "RGB"

def test_to_pil_grayscale():
    gray_img = np.full((100, 100), 128, dtype=np.uint8)
    pil_img = to_pil(gray_img)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.mode == "L"
