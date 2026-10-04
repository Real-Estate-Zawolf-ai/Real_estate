import pytest

from app.rag.retrieval.hybrid_search import _min_max_normalize


def test_min_max_normalize_preserves_order():
    assert _min_max_normalize([2.0, 4.0, 6.0]) == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_normalize_handles_equal_values():
    assert _min_max_normalize([3.0, 3.0]) == [1.0, 1.0]
