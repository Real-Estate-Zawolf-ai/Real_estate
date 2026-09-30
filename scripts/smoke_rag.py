"""
scripts/smoke_rag.py

End-to-end smoke test for the real RAG pipeline (no mocks):

    Metadata Filter -> Hybrid Search (Qdrant) -> Rerank -> Groq Generation

Run from the repository root so that .env is picked up:

    python scripts/smoke_rag.py --query "عايز شقة 3 غرف في التجمع"
    python scripts/smoke_rag.py \
        --query "3 bedroom apartment in New Cairo" \
        --requirements '{"location": "New Cairo", "min_bedrooms": 3, "max_price": 5000000}'

If the connection to Qdrant Cloud is reset during the TLS handshake, run with:

    OPENSSL_CONF=$PWD/openssl_qdrant.cnf python scripts/smoke_rag.py ...

Requires in .env:
    GROQ_API_KEY
    QDRANT_URL (or QDRANT_CLUSTER_ENDPOINT)
    QDRANT_API_KEY
and a populated Qdrant collection (see scripts/ingest.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.helpers.config import get_settings  # noqa: E402
from app.rag.generation import generate_response  # noqa: E402
from app.rag.retrieval.hybrid_search import SearchResult, hybrid_search  # noqa: E402
from app.rag.retrieval.metadata_filter import build_metadata_filter  # noqa: E402
from app.rag.retrieval.reranker import rerank  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real end-to-end RAG smoke test.")
    parser.add_argument("--query", required=True, help="Customer query text.")
    parser.add_argument(
        "--requirements",
        default="{}",
        help='JSON object of structured requirements, e.g. \'{"location": "New Cairo"}\'.',
    )
    parser.add_argument("--collection", default=None, help="Qdrant collection name.")
    parser.add_argument("--top-n", type=int, default=10, help="Hybrid candidates before rerank.")
    parser.add_argument("--top-k", type=int, default=3, help="Final results after rerank.")
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder.")
    parser.add_argument("--no-generate", action="store_true", help="Stop after retrieval.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args()


def _build_qdrant_client(settings):
    from qdrant_client import QdrantClient

    url = settings.QDRANT_URL or settings.QDRANT_CLUSTER_ENDPOINT

    if not url:
        raise RuntimeError("Set QDRANT_URL (or QDRANT_CLUSTER_ENDPOINT) in .env.")

    return QdrantClient(url=url, api_key=settings.QDRANT_API_KEY, timeout=30)


def _print_results(title: str, results: list[SearchResult]) -> None:
    print(f"\n=== {title} ({len(results)}) ===")

    if not results:
        print("(none)")
        return

    for rank, result in enumerate(results, start=1):
        snippet = (result.text or "").replace("\n", " ")[:90]
        print(
            f"{rank}. property_id={result.property_id} | chunk_id={result.chunk_id} | "
            f"source={result.source_file}\n"
            f"   semantic={result.semantic_score:.4f} | lexical={result.lexical_score:.4f} | "
            f"combined={result.combined_score:.4f}\n"
            f"   text: {snippet}"
        )


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    collection = args.collection or settings.QDRANT_COLLECTION

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------

    if not args.no_generate and not settings.GROQ_API_KEY:
        print("FAIL: GROQ_API_KEY is not set in .env (or use --no-generate).")
        return 1

    try:
        requirements = json.loads(args.requirements)
        if not isinstance(requirements, dict):
            raise ValueError("requirements must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL: invalid --requirements: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Qdrant connectivity (separate from collection existence)
    # ------------------------------------------------------------------

    try:
        client = _build_qdrant_client(settings)
        existing = [item.name for item in client.get_collections().collections]
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot reach Qdrant: {exc!r}")
        print(
            "If this is 'Connection reset by peer' during the TLS handshake, run with:\n"
            "  OPENSSL_CONF=$PWD/openssl_qdrant.cnf python scripts/smoke_rag.py ..."
        )
        return 1

    print(f"Qdrant reachable. Collections: {existing}")

    if collection not in existing:
        print(
            f"FAIL: collection '{collection}' does not exist on this cluster. "
            "Run scripts/ingest.py to create and populate it."
        )
        return 1

    points_count = getattr(client.get_collection(collection), "points_count", 0) or 0
    print(f"Collection '{collection}': {points_count} points")

    if points_count == 0:
        print("FAIL: collection is empty. Run scripts/ingest.py first.")
        return 1

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    metadata_filter = build_metadata_filter(requirements)
    print(f"Query: {args.query}")
    print(f"Filter: {metadata_filter.describe()}")

    try:
        candidates = hybrid_search(
            query=args.query,
            requirements_filter=metadata_filter,
            collection_name=collection,
            qdrant_client=client,
            top_k=args.top_n,
        )
    except AttributeError as exc:
        print(
            f"FAIL: {exc!r}\n"
            "Your qdrant-client version may not have QdrantClient.search(). "
            "Check `pip show qdrant-client`; newer versions use query_points()."
        )
        return 1

    _print_results("Hybrid candidates", candidates)

    if not candidates and not metadata_filter.is_empty():
        # Diagnose: is the filter too strict, or do the payload keys not match?
        unfiltered = hybrid_search(
            query=args.query,
            requirements_filter=None,
            collection_name=collection,
            qdrant_client=client,
            top_k=3,
        )
        print("\nFilter returned nothing. Unfiltered top results:")
        _print_results("Unfiltered (diagnostic)", unfiltered)

        if unfiltered:
            print(f"Payload keys seen: {sorted(unfiltered[0].metadata.keys())}")
            print(
                "If your filter fields (e.g. 'location', 'price') are missing from these "
                "keys, or nested under 'metadata', the Qdrant filter keys need adjusting."
            )

    if candidates:
        print(f"\nMetadata keys on top candidate: {sorted(candidates[0].metadata.keys())}")

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    final_results = candidates

    if candidates and not args.no_rerank:
        final_results = rerank(args.query, candidates, top_k=args.top_k)
        _print_results("Reranked", final_results)
    else:
        final_results = candidates[: args.top_k]

    if args.no_generate:
        return 0

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    generation = generate_response(
    customer_query=args.query,
    candidates=final_results,
    requirements=requirements,
)

    print("\n=== Answer ===")
    print(generation.answer_text)
    print("\n=== Traceability ===")
    print(f"model:                  {generation.model}")
    print(f"grounding_property_ids: {generation.grounding_property_ids}")
    print(f"cited_property_ids:     {generation.cited_property_ids}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())