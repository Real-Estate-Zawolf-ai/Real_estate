"""Test listings must never be passed off as real inventory.

Listings ingested from a test/demo file are hidden from the public catalogue, from search and from
the assistant unless SHOW_DEMO_LISTINGS is switched on - and then they carry ``is_demo`` so the
website can label them. Local file paths from ingestion never reach visitors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag.retrieval.hybrid_search import SearchResult
from app.workflows import rag

DEMO_PATH = "data\\raw\\properties\\test_properties_small.csv"  # exactly what the Windows ingest stored
REAL_PATH = "data/raw/properties/march_inventory.xlsx"


def point(pid, path):
    return SimpleNamespace(
        id=pid,
        payload={
            "property_id": pid,
            "source_path": path,
            "file_name": path.replace("\\", "/").split("/")[-1],
            "file_extension": ".csv",
            "source_type": "spreadsheet",
            "location": "New Cairo",
            "price": 5_000_000,
        },
    )


class FakeQdrant:
    """Serves ``points`` in pages of ``limit`` like Qdrant's scroll."""

    def __init__(self, points):
        self.points = points
        self.calls = 0

    def scroll(self, collection_name, limit, offset=None, **_):
        self.calls += 1
        start = offset or 0
        end = start + limit
        return self.points[start:end], (end if end < len(self.points) else None)

    def close(self):
        pass


@pytest.fixture
def qdrant(monkeypatch):
    def install(points):
        fake = FakeQdrant(points)
        monkeypatch.setattr(rag, "create_client", lambda: fake)
        return fake

    return install


@pytest.mark.parametrize(
    "source, expected",
    [
        (DEMO_PATH, True),
        ("data/raw/properties/test_properties.csv", True),
        ("C:\\x\\Demo-Listings.xlsx", True),
        ("sample_units.pdf", True),
        (REAL_PATH, False),
        ("data/test/real_listings.csv", False),  # a folder called 'test' does not make the file a test file
        (None, False),
        ("", False),
    ],
)
def test_demo_files_are_recognised_by_file_name_only(source, expected, settings_env):
    settings_env()
    assert rag.is_demo_listing(source) is expected


def test_demo_listings_are_hidden_by_default(qdrant, settings_env):
    settings_env()
    qdrant([point("1", DEMO_PATH), point("2", REAL_PATH), point("3", DEMO_PATH)])

    assert [p["property_id"] for p in rag.list_indexed_properties()] == ["2"]


def test_demo_mode_shows_them_labelled_and_leaves_real_listings_unlabelled(qdrant, settings_env):
    settings_env(SHOW_DEMO_LISTINGS="true")
    qdrant([point("1", DEMO_PATH), point("2", REAL_PATH)])

    listed = {p["property_id"]: p for p in rag.list_indexed_properties()}
    assert listed["1"]["is_demo"] is True
    assert "is_demo" not in listed["2"]


def test_the_catalogue_never_exposes_local_file_paths(qdrant, settings_env):
    settings_env(SHOW_DEMO_LISTINGS="true")
    qdrant([point("1", DEMO_PATH), point("2", REAL_PATH)])

    for record in rag.list_indexed_properties():
        assert not {"source_path", "file_name", "file_extension", "source_type"} & record.keys()
        assert "\\" not in repr(record)


def test_the_catalogue_pages_through_qdrant_instead_of_stopping_at_one_page(qdrant, settings_env):
    settings_env()
    fake = qdrant([point(str(n), REAL_PATH) for n in range(250)])

    assert len(rag.list_indexed_properties(limit=100)) == 250
    assert fake.calls == 3


def test_the_catalogue_is_capped(qdrant, settings_env, monkeypatch):
    settings_env()
    monkeypatch.setattr(rag, "CATALOGUE_MAX_PROPERTIES", 120)
    qdrant([point(str(n), REAL_PATH) for n in range(400)])

    assert len(rag.list_indexed_properties(limit=100)) == 120


def result(pid, source):
    return SearchResult(
        property_id=pid,
        chunk_id=f"c{pid}",
        text="t",
        metadata={"file_name": source.replace("\\", "/").split("/")[-1], "source_path": source},
        combined_score=1.0,
        source_file=source,
    )


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(rag, "create_client", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(rag, "rerank", lambda query, candidates, top_k: candidates[:top_k])

    def run(*results):
        monkeypatch.setattr(rag, "hybrid_search", lambda **_: list(results))
        return rag.search_index("villa")

    return run


def test_search_and_the_assistant_never_return_demo_listings_by_default(search, settings_env):
    settings_env()
    assert [r.property_id for r in search(result("1", DEMO_PATH), result("2", REAL_PATH))] == ["2"]


def test_search_marks_demo_listings_in_demo_mode(search, settings_env):
    settings_env(SHOW_DEMO_LISTINGS="true")
    found = {r.property_id: r for r in search(result("1", DEMO_PATH), result("2", REAL_PATH))}
    assert found["1"].metadata["is_demo"] is True
    assert "is_demo" not in found["2"].metadata


def test_candidate_payloads_sent_to_the_browser_have_no_local_paths():
    payload = rag._candidate_dict(result("1", DEMO_PATH))
    assert "source_path" not in payload["metadata"]
    assert payload["source_file"] == "test_properties_small.csv"
