from pathlib import Path
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.

    Falls back to pdfplumber for pages where pypdf
    fails to extract meaningful text.
    """

    logger.info("Starting PDF parsing: %s", file_path)

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        logger.info(
            "PDF loaded | filename=%s | total_pages=%d",
            file_path,
            total_pages,
        )

        page_texts: list[str | None] = [None] * total_pages
        blank_pages: list[int] = []

        # Primary extraction using pypdf
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                page_texts[i] = text
            else:
                blank_pages.append(i + 1)

        # Fallback extraction using pdfplumber
        if blank_pages:
            logger.info(
                "Some pages returned no text from pypdf | blank_pages=%s",
                blank_pages,
            )

            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    for page_num in blank_pages:
                        page = pdf.pages[page_num - 1]
                        fallback_text = page.extract_text() or ""

                        if fallback_text.strip():
                            page_texts[page_num - 1] = fallback_text

            except Exception:
                logger.exception(
                    "pdfplumber fallback failed | filename=%s",
                    file_path,
                )

        # Preserve original page order
        full_text = "\n\n".join(
            text.strip()
            for text in page_texts
            if text and text.strip()
        )

        if not full_text:
            logger.warning(
                "No text extracted; PDF may be image-based | filename=%s",
                file_path,
            )
        else:
            logger.info(
                "PDF text extracted successfully | filename=%s | characters=%d",
                file_path,
                len(full_text),
            )

        return full_text

    except Exception:
        logger.exception(
            "PDF parsing failed | filename=%s",
            file_path,
        )
        raise