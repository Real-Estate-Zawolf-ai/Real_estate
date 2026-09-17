from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.ocr.extractor import OCRExtractor
from app.rag.ocr.schemas import OCRResult, OCRStatus, PageResult

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)


class PDFOCRProcessor:
    """
    Processor for handling PDF documents requiring OCR.

    Capabilities:
        - Scanned vs. text-based PDF detection (inspects selectable text density).
        - Renders scanned/image PDF pages to high-resolution images (using PyMuPDF / fitz).
        - Runs OCR page-by-page.
        - Preserves page numbers, block coordinates, and aggregate confidence.
        - Avoids expensive OCR rendering if the PDF is already text-rich (unless force_ocr=True).
        - Uses in-memory rendering when the provider supports ``read_image_bytes()``.
    """

    def __init__(
        self,
        extractor: OCRExtractor | None = None,
        min_text_chars_per_page: int = 40,
        render_dpi: int = 300,
        settings: Settings | None = None,
    ) -> None:
        # Wire settings if provided; explicit kwargs override.
        if settings is not None:
            min_text_chars_per_page = (
                min_text_chars_per_page
                if min_text_chars_per_page != 40
                else settings.OCR_MIN_TEXT_CHARS_PER_PAGE
            )
            render_dpi = (
                render_dpi if render_dpi != 300 else settings.OCR_RENDER_DPI
            )

        self.extractor = extractor or OCRExtractor(settings=settings)
        self.min_text_chars_per_page = min_text_chars_per_page
        self.render_dpi = render_dpi

        logger.info(
            "PDFOCRProcessor initialized | min_text_chars_per_page=%d | render_dpi=%d",
            min_text_chars_per_page,
            render_dpi,
        )

    def is_scanned_pdf(self, pdf_path: str | Path) -> bool:
        """
        Determine if a PDF is predominantly scanned (lacks embedded text).
        Returns True if the average character count per page is below the threshold.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            import fitz  # PyMuPDF
            with fitz.open(str(path)) as doc:
                total_pages = len(doc)
                if total_pages == 0:
                    return True

                total_chars = 0
                for page in doc:
                    text = page.get_text() or ""
                    total_chars += len(text.strip())

                avg_chars = total_chars / total_pages
                return avg_chars < self.min_text_chars_per_page
        except ImportError:
            # Fallback to pypdf if fitz not installed
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                total_pages = len(reader.pages)
                if total_pages == 0:
                    return True
                total_chars = sum(len((p.extract_text() or "").strip()) for p in reader.pages)
                return (total_chars / total_pages) < self.min_text_chars_per_page
            except Exception as exc:
                logger.warning("Error checking if PDF is scanned, defaulting to True: %s", exc)
                return True
        except Exception as exc:
            logger.warning("Error inspecting PDF with PyMuPDF: %s", exc)
            return True

    def process_pdf(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
        extra_metadata: dict[str, Any] | None = None,
    ) -> OCRResult:
        """
        Process a PDF document, executing OCR on pages that require it.

        Args:
            pdf_path: Path to the PDF file.
            force_ocr: If True, executes OCR even if selectable text is present.
            extra_metadata: Additional metadata to attach to the OCRResult.

        Returns:
            OCRResult containing per-page results and aggregated document text.
        """
        path = Path(pdf_path)
        metadata = dict(extra_metadata or {})
        metadata["file_path"] = str(path)
        metadata["file_name"] = path.name

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: '{path.suffix}'")

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF page rendering and OCR. "
                "Please ensure PyMuPDF is installed."
            ) from exc

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", path, exc)
            return OCRResult(
                source=str(path),
                text="",
                pages=[],
                status=OCRStatus.FAILED,
                errors=[f"Corrupted or unreadable PDF: {exc}"],
                metadata=metadata,
            )

        # Use context manager to guarantee resource cleanup.
        with doc:
            total_pages = len(doc)
            metadata["total_pages"] = total_pages

            if total_pages == 0:
                return OCRResult(
                    source=str(path),
                    text="",
                    pages=[],
                    status=OCRStatus.EMPTY,
                    metadata=metadata,
                )

            page_results: list[PageResult] = []
            errors: list[str] = []

            # Zoom matrix for high-quality rendering (default 300 DPI: 300/72 ≈ 4.166)
            scale = self.render_dpi / 72.0
            matrix = fitz.Matrix(scale, scale)

            for page_index in range(total_pages):
                page_num = page_index + 1
                page = doc[page_index]
                embedded_text = (page.get_text() or "").strip()

                # If the page already has rich embedded text and force_ocr is False, use text parser
                if not force_ocr and len(embedded_text) >= self.min_text_chars_per_page:
                    logger.debug(
                        "PDF page %d has selectable text (%d chars). Using embedded text.",
                        page_num,
                        len(embedded_text),
                    )
                    parsed_page = self.extractor.parser.parse_plain_text(
                        embedded_text,
                        page_number=page_num,
                        confidence=1.0,
                        metadata={"source_page": page_num, "method": "embedded_text"},
                    )
                    page_results.append(parsed_page)
                    continue

                # Render page to image and perform OCR
                logger.info("Rendering PDF page %d for OCR extraction", page_num)
                try:
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                    # Use in-memory bytes to avoid unnecessary disk I/O.
                    img_bytes = pixmap.tobytes("png")
                    detections = self.extractor.provider.read_image_bytes(img_bytes)

                    # Build page result from raw detections via the parser
                    rendered_page_res = self.extractor.parser.parse_page(
                        detections=detections,
                        page_number=page_num,
                        metadata={"source_page": page_num, "method": "scanned_ocr"},
                    )
                    page_results.append(rendered_page_res)

                except Exception as exc:
                    logger.exception("Failed to render/OCR PDF page %d: %s", page_num, exc)
                    errors.append(f"Page {page_num} OCR failed: {exc}")
                    page_results.append(
                        PageResult(
                            page_number=page_num,
                            text="",
                            confidence=0.0,
                            metadata={"error": str(exc)},
                        )
                    )

        # Combine all page texts
        all_texts = [p.text for p in page_results if p.text]
        full_text = "\n\n".join(all_texts).strip()

        # Aggregate confidence across pages that yielded text
        pages_with_confidence = [p.confidence for p in page_results if p.confidence > 0.0]
        avg_doc_confidence = (
            sum(pages_with_confidence) / len(pages_with_confidence)
            if pages_with_confidence
            else 0.0
        )

        detected_lang = self.extractor.parser.detect_language(full_text)

        status = OCRStatus.SUCCESS
        if errors and not full_text:
            status = OCRStatus.FAILED
        elif errors and full_text:
            status = OCRStatus.PARTIAL
        elif not full_text:
            status = OCRStatus.EMPTY

        return OCRResult(
            source=str(path),
            text=full_text,
            pages=page_results,
            language=detected_lang,
            confidence=avg_doc_confidence,
            status=status,
            errors=errors,
            metadata=metadata,
        )
