from pathlib import Path

from app.rag.ocr.extractor import OCRExtractor


def main() -> None:

    image_path = Path(
        "/Users/mm/Desktop/RealEstateProject/Real_estate/data/raw/images/imagetest.jpg"
    )

    print("=" * 80)
    print("REAL ESTATE OCR TEST")
    print("=" * 80)

    if not image_path.exists():
        print(f"\nImage not found:")
        print(image_path)

        print("\nPut a test image here:")
        print("data/raw/images/test_property.jpg")

        return

    print(f"\nImage: {image_path}")

    print("\nLoading EasyOCR...")
    print("Languages: Arabic + English")
    print("Device: CPU")

    extractor = OCRExtractor(
        languages=["ar", "en"],
        gpu=False,
    )

    print("\nEasyOCR loaded successfully.")

    print("\nRunning OCR...")

    text = extractor.extract_from_image(
        image_path
    )

    print("\n" + "=" * 80)
    print("OCR RESULT")
    print("=" * 80)

    if text:
        print(text)
    else:
        print("[No text detected]")

    print("\n" + "=" * 80)
    print(f"Characters extracted: {len(text)}")
    print("=" * 80)


if __name__ == "__main__":
    main()