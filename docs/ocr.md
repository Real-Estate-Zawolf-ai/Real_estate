# OCR & Document Processing Module

## Overview

The OCR module (`app/rag/ocr/`) provides text extraction and parsing capabilities for real-estate documents, supporting:
- **Arabic and English** text recognition simultaneously.
- Real-estate images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`).
- Scanned and hybrid PDFs (automatic scanned page detection, 300 DPI image rendering via PyMuPDF, and page-by-page OCR).
- Bounding box coordinates and confidence score tracking per line/block.
- Conservative text post-processing (Arabic orthographic normalization, Arabic-Indic to Western numeral translation for property prices and phone numbers).

---

## Architecture & Components

```
                      Document / Image
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
    Image File                             PDF Document
        │                                       │
        │                                       ▼
        │                              PDFOCRProcessor
        │                      (Check if scanned vs text-rich)
        │                                       │
        │                          ┌────────────┴────────────┐
        │                          ▼                         ▼
        │                     Text-Rich PDF             Scanned PDF
        │                  (Embedded Extractor)      (PyMuPDF 300 DPI Render)
        │                          │                         │
        └──────────────────────────┼─────────────────────────┘
                                   ▼
                             OCRExtractor
                     (BaseOCRProvider / EasyOCR)
                                   │
                                   ▼
                             DocumentParser
                     (Confidence filter, language detection)
                                   │
                                   ▼
                           OCRPostProcessor
               (Arabic & numeral normalization, noise cleanup)
                                   │
                                   ▼
                               OCRResult
                   (Pages, Blocks, Confidence, Bboxes)
```

### Module Files

| File | Purpose |
|---|---|
| `app/rag/ocr/schemas.py` | Structured dataclasses (`OCRResult`, `PageResult`, `BlockResult`, `OCRDetection`, `OCRStatus`). |
| `app/rag/ocr/base.py` | Abstract `BaseOCRProvider` interface for pluggable OCR backends. |
| `app/rag/ocr/extractor.py` | `OCRExtractor` with `EasyOCRProvider` (GPU/CPU support, lazy model initialization). |
| `app/rag/ocr/post_processor.py` | `OCRPostProcessor` for Arabic normalization and Arabic-Indic digit translation. |
| `app/rag/ocr/document_parser.py` | `DocumentParser` for confidence filtering and language hint detection (`ar`, `en`, `ar_en`). |
| `app/rag/ocr/pdf_processor.py` | `PDFOCRProcessor` for scanned PDF detection and page rendering. |
| `app/rag/ocr/__init__.py` | Clean public exports for the module. |

---

## Output Data Schema

Every OCR extraction returns an `OCRResult`:

```python
OCRResult:
    source: str                    # Path to file
    text: str                      # Aggregated document text
    pages: list[PageResult]        # 1-indexed per-page results
        page_number: int
        text: str
        confidence: float          # Average confidence score for page
        language: str | None       # 'ar', 'en', 'ar_en'
        blocks: list[BlockResult]  # Bounding box & confidence per text segment
            text: str
            confidence: float
            bounding_box: list     # Coordinates [[x1,y1], [x2,y2], ...]
    language: str | None           # Detected document-level language
    confidence: float              # Overall document confidence
    status: OCRStatus              # 'success' | 'partial' | 'empty' | 'failed'
    errors: list[str]              # Actionable error messages
    metadata: dict                 # Source file, dimensions, etc.
```

---

## Usage Examples

### 1. Image OCR

```python
from app.rag.ocr import OCRExtractor

extractor = OCRExtractor(languages=["ar", "en"], gpu=False)
result = extractor.extract_from_image("data/raw/images/property_ad.jpg")

print(f"Status: {result.status}")
print(f"Detected Language: {result.language}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Extracted Text:\n{result.text}")
```

### 2. Scanned / Hybrid PDF Processing

```python
from app.rag.ocr import PDFOCRProcessor

processor = PDFOCRProcessor()
result = processor.process_pdf("data/raw/documents/contract_scan.pdf")

for page in result.pages:
    print(f"--- Page {page.page_number} (Conf: {page.confidence:.2f}) ---")
    print(page.text)
```

### 3. Integrating with Ingestion Pipeline

The ingestion pipeline can conditionally delegate to OCR:

```python
from app.rag.ocr import OCRExtractor, PDFOCRProcessor

if file_type == "image":
    ocr_result = OCRExtractor().extract_from_image(file_path)
    text = ocr_result.text

elif file_type == "pdf":
    pdf_processor = PDFOCRProcessor()
    if pdf_processor.is_scanned_pdf(file_path):
        ocr_result = pdf_processor.process_pdf(file_path)
        text = ocr_result.text
    else:
        text = extract_embedded_pdf_text(file_path)
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OCR_LANGUAGES` | `["ar", "en"]` | Languages recognized by EasyOCR |
| `OCR_GPU` | `false` | Enable CUDA GPU acceleration |
| `OCR_MIN_CONFIDENCE` | `0.30` | Minimum confidence threshold to keep a block |
| `OCR_MODEL_DIR` | `null` | Custom local directory for EasyOCR model weights |
| `OCR_RENDER_DPI` | `300` | DPI for rendering PDF pages to images |
| `OCR_MIN_TEXT_CHARS_PER_PAGE` | `40` | Threshold below which a PDF is treated as scanned |

---

## Running Tests

Run the OCR unit tests using `pytest`:

```bash
pytest tests/unit/test_ocr.py -v
```
