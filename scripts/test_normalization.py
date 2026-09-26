from app.rag.ocr.document_parser import DocumentParser
from app.rag.ocr.post_processor import OCRPostProcessor
from app.rag.normalization.property_normalizer import PropertyNormalizer
from app.rag.normalization.field_mapper import FieldMapper


def main() -> None:
    print("=" * 80)
    print("REAL ESTATE NORMALIZATION PIPELINE TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Simulate OCR output
    # ---------------------------------------------------------
    raw_ocr_text = """
    شقة للبيع في التجمع الخامس
    ٣ غرف نوم
    ٢ حمام
    المساحة: ١٥٠ متر مربع
    السعر: ٣,٥٠٠,٠٠٠ جنيه مصري
    المطور: Mountain View
    تشطيب كامل
    """

    print("\n[1] RAW OCR TEXT")
    print("-" * 80)
    print(raw_ocr_text)

    # ---------------------------------------------------------
    # 2. Document Parser
    # ---------------------------------------------------------
    parser = DocumentParser()

    parsed = parser.parse_text(raw_ocr_text)

    print("\n[2] DOCUMENT PARSER")
    print("-" * 80)
    print(f"Language hint: {parsed.language_hint}")
    print(f"Average confidence: {parsed.average_confidence}")
    print(f"Text:\n{parsed.text}")

    # ---------------------------------------------------------
    # 3. OCR Post Processor
    # ---------------------------------------------------------
    post_processor = OCRPostProcessor()

    processed = post_processor.process(parsed.text)

    print("\n[3] OCR POST PROCESSOR")
    print("-" * 80)
    print(f"Processed text:\n{processed.text}")
    print(f"Corrections: {processed.corrections}")

    # ---------------------------------------------------------
    # 4. Property Normalizer
    # ---------------------------------------------------------
    normalizer = PropertyNormalizer()

    normalized_property = normalizer.normalize(processed.text)

    print("\n[4] PROPERTY NORMALIZER")
    print("-" * 80)
    print(normalized_property)

    # ---------------------------------------------------------
    # 5. Field Mapper
    # ---------------------------------------------------------
    mapper = FieldMapper()

    # Convert normalized property to dictionary
    property_dict = {
        "property_type": normalized_property.property_type,
        "transaction_type": normalized_property.transaction_type,
        "price": normalized_property.price,
        "currency": normalized_property.currency,
        "area": normalized_property.area,
        "area_unit": normalized_property.area_unit,
        "bedrooms": normalized_property.bedrooms,
        "bathrooms": normalized_property.bathrooms,
        "location": normalized_property.location,
        "compound": normalized_property.compound,
        "developer": normalized_property.developer,
        "finishing": normalized_property.finishing,
        "delivery": normalized_property.delivery,
        "amenities": normalized_property.amenities,
    }

    mapped = mapper.map_fields(property_dict)

    print("\n[5] FIELD MAPPER")
    print("-" * 80)
    print("Mapped fields:")
    for key, value in mapped.mapped.items():
        print(f"  {key}: {value}")

    print("\nUnmapped fields:")
    for key, value in mapped.unmapped.items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # 6. Final Result
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL NORMALIZED PROPERTY")
    print("=" * 80)

    for key, value in property_dict.items():
        print(f"{key:20} = {value}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()