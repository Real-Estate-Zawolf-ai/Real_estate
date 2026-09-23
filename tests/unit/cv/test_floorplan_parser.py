import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from app.rag.cv.floorplan_parser import parse_floorplan, FloorplanResult

def create_synthetic_floorplan(num_rooms=3):
    """
    Creates a white image with black rectangles representing rooms.
    """
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    # Room sizes and positions
    rooms = [
        (50, 50, 80, 80),
        (200, 50, 80, 120),
        (300, 200, 80, 80),
    ][:num_rooms]

    for (x, y, w, h) in rooms:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    return img

def test_parse_floorplan_basic():
    img = create_synthetic_floorplan(num_rooms=3)
    # We use bytes to test the load_image integration within parse_floorplan
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    result = parse_floorplan(img_bytes)

    assert isinstance(result, FloorplanResult)
    assert result.room_count_estimate == 3
    assert len(result.regions) == 3
    assert result.confidence > 0

def test_parse_floorplan_empty():
    # Pure white image
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    result = parse_floorplan(img_bytes)

    assert result.room_count_estimate == 0
    assert "No room regions detected" in result.notes[0]
    assert result.confidence == 0.0

def test_parse_floorplan_large_noise():
    # Image with one giant rectangle (too large to be a room)
    img = np.full((500, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (490, 490), (0, 0, 0), -1)
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    result = parse_floorplan(img_bytes)

    # Should filter out the outer contour
    assert result.room_count_estimate == 0
