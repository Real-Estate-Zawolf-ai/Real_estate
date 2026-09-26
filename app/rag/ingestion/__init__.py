"""
RAG Ingestion — public API.

Re-exports all loaders, the chunker, and metadata helpers so consumers
can import from ``app.rag.ingestion`` without reaching into submodules.
"""

from app.rag.ingestion.pdf_loader import parse_pdf
from app.rag.ingestion.office_loader import parse_office
from app.rag.ingestion.text_loader import parse_text
from app.rag.ingestion.spreadsheet_loader import parse_spreadsheet
from app.rag.ingestion.image_loader import load_image
from app.rag.ingestion.chunker import chunk_text
from app.rag.ingestion.metadata import build_metadata, add_chunk_metadata

__all__ = [
    "parse_pdf",
    "parse_office",
    "parse_text",
    "parse_spreadsheet",
    "load_image",
    "chunk_text",
    "build_metadata",
    "add_chunk_metadata",
]