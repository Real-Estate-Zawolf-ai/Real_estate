from app.rag.ingestion.pipeline import ingest_file


file_path = "/Users/mm/Desktop/RealEstateProject/Real_estate/data/raw/documents/ Requirements.docx"

result = ingest_file(file_path)

print(f"\nStatus: {result['status']}")
print(f"File type: {result['file_type']}")
print(f"Total chunks: {result['chunks_count']}")

for i, chunk in enumerate(result["chunks"], start=1):
    print("\n" + "=" * 80)
    print(f"CHUNK {i}")
    print(f"Length: {len(chunk['text'])} characters")
    print(f"ID: {chunk['id']}")
    print("-" * 80)
    print(chunk["text"])

print("\n" + "=" * 80)