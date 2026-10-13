# Needs the offline ingestion extras (requirements-ingest.txt); skipped on the lean API install.
import pytest as _pytest
_pytest.importorskip("pypdf")
_pytest.importorskip("pandas")

from pathlib import Path

from app.rag.ingestion.pipeline import (
    detect_file_type,
    is_supported,
    ingest_file,
)


def test_detect_file_type():
    assert detect_file_type("property.pdf") == "pdf"
    assert detect_file_type("property.docx") == "office"
    assert detect_file_type("property.pptx") == "office"
    assert detect_file_type("property.txt") == "text"
    assert detect_file_type("property.md") == "text"
    assert detect_file_type("properties.csv") == "spreadsheet"
    assert detect_file_type("properties.xlsx") == "spreadsheet"
    assert detect_file_type("property.jpg") == "image"


def test_is_supported():
    assert is_supported("property.pdf")
    assert is_supported("property.docx")
    assert is_supported("property.txt")
    assert is_supported("properties.xlsx")
    assert is_supported("property.jpg")

    assert not is_supported("property.exe")


def test_ingest_text_file(tmp_path: Path):
    file_path = tmp_path / "property.txt"

    file_path.write_text(
        """
        Property: Mountain View
        Location: New Cairo
        Type: Apartment
        Bedrooms: 3
        Bathrooms: 2
        Area: 180 sqm
        Price: 5,000,000 EGP

        The apartment has a garden view and is located
        near the main entrance of the compound.
        """,
        encoding="utf-8",
    )

    result = ingest_file(
        str(file_path),
        property_id="PROP-001",
    )

    assert result["status"] == "success"
    assert result["file_type"] == "text"
    assert result["chunks_count"] > 0

    chunks = result["chunks"]

    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert "id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk

    assert first_chunk["metadata"]["property_id"] == "PROP-001"
    assert first_chunk["metadata"]["source"] == "property.txt"


def test_ingest_empty_text_file(tmp_path: Path):
    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    result = ingest_file(str(file_path))

    assert result["status"] == "empty"


def test_ingest_unsupported_file(tmp_path: Path):
    file_path = tmp_path / "property.exe"

    file_path.write_text(
        "fake content",
        encoding="utf-8",
    )

    result = ingest_file(str(file_path))

    assert result["status"] == "unsupported"
