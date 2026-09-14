from app.rag.embeddings import EmbeddingModel


def main() -> None:
    embedder = EmbeddingModel()

    texts = [
        "شقة للبيع في التجمع الخامس بمساحة 150 متر",
        "Apartment for sale in New Cairo with 3 bedrooms",
        "شقة للبيع في القاهرة الجديدة",
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Expected dimension: {embedder.dimension}")

    print("\nFirst 10 values:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()