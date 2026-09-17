# OCR & Document Processing Module

## Overview

The OCR module (`app/rag/ocr/`) provides text extraction and parsing capabilities for real-estate documents, supporting:
- **Arabic and English** text recognition simultaneously.
- **Pluggable Provider Architecture**: Choose between **PaddleOCR** (recommended default) and **EasyOCR**, or add custom engines (Tesseract, Cloud OCR).
- Real-estate images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`).
- Scanned and hybrid PDFs (automatic scanned page detection, 300 DPI in-memory image rendering via PyMuPDF without temp file disk I/O, and page-by-page OCR).
- Bounding box coordinates and confidence score tracking per line/block.
- Conservative text post-processing (Arabic orthographic normalization, Arabic-Indic to Western numeral translation for property prices and phone numbers).
- Final document-level pass to clean cross-block artifacts and whitespace.

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
        │                  (Embedded Extractor)      (PyMuPDF 300 DPI In-Memory)
        │                          │                         │
        └──────────────────────────┼─────────────────────────┘
                                   ▼
                              OCRExtractor
                                   │
               ┌───────────────────┴───────────────────┐
               ▼                                       ▼
        PaddleOCRProvider                      EasyOCRProvider
     (Default, Arabic+English)               (Alternative backend)
               └───────────────────┬───────────────────┘
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
| `app/rag/ocr/base.py` | Abstract `BaseOCRProvider` interface (`read_image`, `read_image_bytes`, `is_available`). |
| `app/rag/ocr/paddle_provider.py` | `PaddleOCRProvider` with result adapter, thread-safe lazy init, and in-memory byte support. |
| `app/rag/ocr/provider_factory.py` | `create_ocr_provider()` factory resolving providers from Settings or arguments. |
| `app/rag/ocr/extractor.py` | `OCRExtractor` image coordinator and `EasyOCRProvider` implementation. |
| `app/rag/ocr/post_processor.py` | `OCRPostProcessor` for Arabic normalization and Arabic-Indic digit translation. |
| `app/rag/ocr/document_parser.py` | `DocumentParser` for confidence filtering, language detection, and document-level cleanup. |
| `app/rag/ocr/pdf_processor.py` | `PDFOCRProcessor` for scanned PDF detection and in-memory page rendering. |
| `app/rag/ocr/__init__.py` | Clean public exports for the module. |

---

## Providers Comparison

| Feature | PaddleOCR (Recommended) | EasyOCR |
|---|---|---|
| **Arabic Script Accuracy** | High (Robust text detection & recognition on cursive Arabic) | Moderate |
| **Mixed Arabic + English** | Seamless (`lang='ar'` includes Latin & numerals) | Supported (`languages=['ar', 'en']`) |
| **In-Memory Rendering** | Supported (`read_image_bytes` via NumPy/Pillow) | Supported (via temp file or bytes) |
| **Initialization** | Thread-safe lazy loading | Thread-safe lazy loading |
| **Default in Project** | Yes (`OCR_PROVIDER=paddleocr`) | Fallback option |

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

### 1. Image OCR with Default Factory

```python
from app.rag.ocr import OCRExtractor, create_ocr_provider

# Automatically resolves provider from config (PaddleOCR by default)
provider = create_ocr_provider()
extractor = OCRExtractor(provider=provider)

result = extractor.extract_from_image("data/raw/images/property_ad.jpg")

print(f"Status: {result.status}")
print(f"Detected Language: {result.language}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Extracted Text:\n{result.text}")
```

### 2. Explicitly Selecting a Provider

```python
from app.rag.ocr import OCRExtractor, PaddleOCRProvider

# Explicit PaddleOCR instantiation
provider = PaddleOCRProvider(lang="ar", use_gpu=False)
extractor = OCRExtractor(provider=provider)
result = extractor.extract_from_image("data/raw/images/contract_preview.png")
```

### 3. Scanned / Hybrid PDF Processing

```python
from app.rag.ocr import PDFOCRProcessor

processor = PDFOCRProcessor()
result = processor.process_pdf("data/raw/documents/contract_scan.pdf")

for page in result.pages:
    print(f"--- Page {page.page_number} (Conf: {page.confidence:.2f}) ---")
    print(page.text)
```

### 4. In-Memory Bytes Processing

```python
from app.rag.ocr import create_ocr_provider

provider = create_ocr_provider()
with open("page_render.png", "rb") as f:
    img_bytes = f.read()

detections = provider.read_image_bytes(img_bytes)
for det in detections:
    print(det.text, det.confidence)
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OCR_PROVIDER` | `paddleocr` | Active OCR backend (`paddleocr` or `easyocr`) |
| `OCR_LANGUAGES` | `["ar", "en"]` | Languages list for EasyOCR provider |
| `OCR_GPU` | `false` | Enable CUDA GPU acceleration |
| `OCR_MIN_CONFIDENCE` | `0.30` | Minimum confidence threshold to keep a block |
| `OCR_MODEL_DIR` | `null` | Custom directory for OCR model weights |
| `OCR_RENDER_DPI` | `300` | DPI for rendering PDF pages to images |
| `OCR_MIN_TEXT_CHARS_PER_PAGE` | `40` | Threshold below which a PDF is treated as scanned |

---

## Running Tests

Run unit and integration tests using `pytest`:

```bash
# Unit tests
python -m pytest tests/unit/test_ocr.py -v

# Integration tests
python -m pytest tests/integration/test_paddleocr.py -v

# All tests
python -m pytest tests/ -v
```
