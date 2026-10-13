"""Validate the deployment configuration. Never prints secret values.

    python scripts/check_env.py              # configuration only (used by start.sh)
    python scripts/check_env.py --connect    # also test every provider credential

Exit status 1 when APP_ENV=production and the configuration is unusable, so a
bad deploy stops at startup instead of failing on the first customer. With
``--connect`` any failing connection also exits 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.helpers.config import get_settings  # noqa: E402
from app.observability.logging import redact  # noqa: E402


def _flag(value: object) -> str:
    return "set" if value else "MISSING"


def show_configuration() -> list[str]:
    settings = get_settings()
    print(f"APP_ENV={settings.APP_ENV}  LOG_FORMAT={settings.LOG_FORMAT}")
    rows = [
        ("DATABASE_URL / POSTGRES_*", _flag(settings.DATABASE_URL or settings.POSTGRES_HOST)),
        ("SESSION_SECRET", _flag(settings.SESSION_SECRET) + (f" ({len(settings.SESSION_SECRET)} chars)" if settings.SESSION_SECRET else "")),
        ("ADMIN_API_KEY", _flag(settings.ADMIN_API_KEY) + (f" ({len(settings.ADMIN_API_KEY)} chars)" if settings.ADMIN_API_KEY else "")),
        ("CORS_ALLOWED_ORIGINS", ", ".join(settings.cors_origins) or "MISSING"),
        ("GROQ_API_KEY", _flag(settings.GROQ_API_KEY)),
        ("GEMINI_API_KEY", _flag(settings.GEMINI_API_KEY)),
        ("QDRANT_URL", settings.qdrant_endpoint),
        ("QDRANT_API_KEY", _flag(settings.QDRANT_API_KEY)),
        ("QDRANT_COLLECTION", settings.QDRANT_COLLECTION),
        ("EMBEDDING", f"{settings.EMBEDDING_BACKEND}:{settings.EMBEDDING_MODEL} dim={settings.EMBEDDING_DIMENSION}"),
        ("RERANKER_BACKEND", settings.RERANKER_BACKEND),
        ("WhatsApp (4 values)", "configured" if settings.whatsapp_configured and settings.WHATSAPP_VERIFY_TOKEN else "not configured"),
        ("Staff alert channels", ", ".join(settings.staff_channels) or "NONE (alerts only via admin API)"),
        ("TRUSTED_PROXY_HOPS", str(settings.TRUSTED_PROXY_HOPS)),
    ]
    for name, value in rows:
        print(f"  {name:28s} {value}")
    return settings.production_problems()


def run_connection_checks() -> bool:
    settings = get_settings()
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    print("Connection checks:")
    try:
        from app.database.migrations import schema_status
        from app.database.session import check_database, is_postgres

        connected, error = check_database()
        report("database", connected, error or ("postgresql" if is_postgres() else "sqlite"))
        if connected:
            up_to_date, detail = schema_status()
            report("migrations", up_to_date, detail)
    except Exception as error:
        report("database", False, redact(type(error).__name__))

    try:
        from app.database.vector_db import describe_collection

        vector = describe_collection()
        report("qdrant collection", vector["ok"], vector.get("error") or f"{vector.get('points', '?')} points, dim {vector.get('actual_dimension')}")
    except Exception as error:
        report("qdrant collection", False, type(error).__name__)

    if settings.EMBEDDING_BACKEND == "gemini" and settings.GEMINI_API_KEY:
        try:
            from app.rag.embeddings import EmbeddingModel

            vector = EmbeddingModel().embed_query("شقة في التجمع الخامس")
            report("gemini embeddings", len(vector) == settings.EMBEDDING_DIMENSION, f"{len(vector)}-dimensional")
        except Exception as error:
            report("gemini embeddings", False, redact(str(error))[:120])

    if settings.GROQ_API_KEY:
        try:
            from groq import Groq

            models = [model.id for model in Groq(api_key=settings.GROQ_API_KEY).models.list().data]
            wanted = settings.GROQ_MODEL
            report("groq", wanted in models, f"{len(models)} models; {wanted} {'available' if wanted in models else 'NOT available'}")
        except Exception as error:
            report("groq", False, redact(str(error))[:120])

    if settings.whatsapp_configured:
        try:
            import httpx

            response = httpx.get(
                f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}/{settings.WHATSAPP_PHONE_NUMBER_ID}",
                params={"fields": "display_phone_number,verified_name"},
                headers={"Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}"},
                timeout=15,
            )
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            report("whatsapp token", response.status_code == 200, body.get("display_phone_number") or f"HTTP {response.status_code}")
        except Exception as error:
            report("whatsapp token", False, type(error).__name__)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", action="store_true", help="also test each provider credential")
    args = parser.parse_args()

    settings = get_settings()
    problems = show_configuration()
    for warning in settings.production_warnings():
        print(f"  warning: {warning}")

    failed = False
    if problems:
        print(f"\n{len(problems)} configuration problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        failed = settings.is_production
        if not settings.is_production:
            print("(not fatal: APP_ENV is not 'production')")
    if args.connect:
        print()
        failed = (not run_connection_checks()) or failed
    if failed:
        sys.exit(1)
    print("\nConfiguration OK" if not problems else "")


if __name__ == "__main__":
    main()
