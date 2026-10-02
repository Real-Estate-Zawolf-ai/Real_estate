from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.post_processor import OCRPostProcessor


def test_document_parser_filters_low_confidence_detections():
    parsed = DocumentParser(min_confidence=0.5).parse(
        [
            {"text": "Apartment for sale", "confidence": 0.9},
            {"text": "noise", "confidence": 0.1},
        ]
    )
    assert parsed.text == "Apartment for sale"
    assert parsed.average_confidence == 0.9


def test_post_processor_normalizes_arabic_digits():
    processed = OCRPostProcessor().process("السعر: ٣,٥٠٠,٠٠٠")
    assert "3,500,000" in processed.text


def test_empty_ocr_input_is_safe():
    assert OCRPostProcessor().process(" ").text == ""
