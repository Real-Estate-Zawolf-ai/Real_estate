"""HTTP-level protections: headers, CORS, docs/metrics exposure, rate limits, error hygiene."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.integration.test_api_channels import chat, client, voice_token  # noqa: F401  (fixtures)

KEY = {"X-Admin-Key": "test-admin-key-0123456789abcdef"}
ROOT = Path(__file__).resolve().parents[2]


def test_every_response_has_security_headers_and_a_request_id(client):
    response = client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert len(response.headers["x-request-id"]) == 12
    assert "strict-transport-security" not in response.headers  # HTTPS-only header: production only


def test_hsts_is_sent_in_production(client, settings_env):
    settings_env(APP_ENV="production")
    assert "max-age=31536000" in client.get("/health").headers["strict-transport-security"]


def test_unhandled_errors_return_a_generic_json_500_not_a_traceback():
    @app.get("/__boom", include_in_schema=False)
    async def boom():
        raise RuntimeError("secret internal detail: db password is hunter2")

    try:
        response = TestClient(app, raise_server_exceptions=False).get("/__boom")
    finally:
        app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", "") != "/__boom"]
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "hunter2" not in response.text and "Traceback" not in response.text


def test_cors_allows_only_configured_origins():
    """CORS is fixed when the app is created, so check it in a process configured like production."""
    code = """
import json
from fastapi.testclient import TestClient
from app.main import app
c = TestClient(app)
def pre(origin, headers="content-type"):
    r = c.options("/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": headers})
    return {k: v for k, v in r.headers.items() if k.startswith("access-control")}
print(json.dumps({"ok": pre("https://app.example.com"), "evil": pre("https://evil.example"), "admin": pre("https://app.example.com", "x-admin-key")}))
"""
    env = {
        "CORS_ALLOWED_ORIGINS": "https://app.example.com", "DATABASE_URL": "sqlite:///cors-test.db",
        "PATH": "", "SYSTEMROOT": "C:\Windows",
    }
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    result = __import__("json").loads(out.stdout.strip().splitlines()[-1])
    assert result["ok"]["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-credentials" not in result["ok"]  # no cookies, ever
    assert "access-control-allow-origin" not in result["evil"]
    assert "x-admin-key" not in result["admin"].get("access-control-allow-headers", "").lower()  # browsers never need it


def test_metrics_need_the_admin_key_unless_made_public(client, settings_env):
    settings_env(METRICS_PUBLIC="false")
    assert client.get("/metrics/").status_code == 401
    assert client.get("/metrics/", headers={"X-Admin-Key": "wrong"}).status_code == 401
    body = client.get("/metrics/", headers=KEY)
    assert body.status_code == 200 and "real_estate_requests_total" in body.text
    settings_env(METRICS_PUBLIC="true")
    assert client.get("/metrics/").status_code == 200


def test_api_docs_are_not_served_in_production():
    code = (
        "from app.main import app; "
        "print(app.docs_url, app.redoc_url, app.openapi_url)"
    )
    env = {"APP_ENV": "production", "DATABASE_URL": "sqlite:///x.db", "PATH": "", "SYSTEMROOT": "C:\\Windows"}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "None None None", out.stderr[-500:]


def test_api_docs_are_available_in_development(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"]


def test_chat_flood_gets_429_with_retry_after_and_other_clients_are_unaffected(client, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_CHAT_PER_MINUTE="3", TRUSTED_PROXY_HOPS="1")
    codes = [client.post("/chat", json={"query": "hello"}, headers={"X-Forwarded-For": "203.0.113.1"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    blocked = client.post("/chat", json={"query": "hello"}, headers={"X-Forwarded-For": "203.0.113.1"})
    assert int(blocked.headers["retry-after"]) >= 1
    assert client.post("/chat", json={"query": "hello"}, headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200


def test_one_session_cannot_flood_even_from_many_addresses(client, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_CHAT_SESSION_PER_MINUTE="2", RATE_LIMIT_CHAT_PER_MINUTE="100", TRUSTED_PROXY_HOPS="1")
    session = chat(client, "hello", headers_unused=None) if False else client.post("/chat", json={"query": "hello"}).json()["session_id"]
    codes = [client.post("/chat", json={"query": "hi", "session_id": session}, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
             for i in range(1, 5)]
    assert codes[:1] == [200] and 429 in codes


def test_creating_sessions_is_limited_per_client(client, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_SESSION_CREATE_PER_HOUR="2", RATE_LIMIT_CHAT_PER_MINUTE="100")
    codes = [client.post("/chat", json={"query": "hello"}).status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]  # a bot cannot mint unlimited conversations


def test_search_endpoints_are_rate_limited(client, settings_env, stub_rag, monkeypatch):
    import app.api.routes.properties as properties

    monkeypatch.setattr(properties, "list_indexed_properties", lambda: [])
    monkeypatch.setattr(properties, "search_index", lambda *a, **k: [])
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_SEARCH_PER_MINUTE="2")
    assert [client.get("/properties").status_code for _ in range(3)] == [200, 200, 429]
    assert [client.post("/properties/search", json={"query": "villa"}).status_code for _ in range(2)] == [429, 429]  # shared budget
    assert client.post("/rag/search", json={"query": "villa"}).status_code == 429


def test_the_catalogue_is_cached_so_visitors_do_not_hit_qdrant_each_time(client, monkeypatch):
    import app.api.routes.properties as properties

    calls = []
    monkeypatch.setattr(properties, "list_indexed_properties", lambda: calls.append(1) or [{"property_id": "1"}])
    for _ in range(5):
        assert client.get("/properties").json() == [{"property_id": "1"}]
    assert len(calls) == 1


def test_provider_failures_surface_as_503_without_internals(client, monkeypatch):
    import app.main as main

    def broken(*args, **kwargs):
        raise ConnectionError("qdrant at 10.0.0.5 refused: api-key=SECRET123")

    monkeypatch.setattr(main, "answer_query", broken)
    response = client.post("/rag/search", json={"query": "villa"})
    assert response.status_code == 503 and "SECRET123" not in response.text and "10.0.0.5" not in response.text


def test_chat_input_limits(client):
    assert client.post("/chat", json={"query": "x" * 2001}).status_code == 422
    assert client.post("/chat", json={"query": "hi", "language": "fr"}).status_code == 422
    assert client.post("/chat", json={"query": "hi", "clear_requirements": ["not_a_key"]}).status_code == 422
    assert client.post("/chat", json={"query": "hi", "session_id": "x" * 101}).status_code == 422
    assert client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_an_expired_session_starts_a_fresh_conversation(client, settings_env, stub_rag):
    first = client.post("/chat", json={"query": "villa in Sheikh Zayed"}).json()
    settings_env(SESSION_MAX_AGE_DAYS="0")  # everything issued before "now" is too old
    import time

    time.sleep(1.1)
    later = client.post("/chat", json={"query": "hello", "session_id": first["session_id"]}).json()
    assert later["session_id"] != first["session_id"] and later["requirements"] == {}


def test_a_valid_session_gets_a_refreshed_id_once_it_is_a_day_old(client, monkeypatch, stub_rag):
    from app.api import sessions

    first = client.post("/chat", json={"query": "hello"}).json()["session_id"]
    raw = sessions.verify_session_id(first)
    aged = sessions.issue_session_id(raw, now=__import__("time").time() - 2 * 86400)
    renewed = client.post("/chat", json={"query": "hello again", "session_id": aged}).json()["session_id"]
    assert renewed != aged and sessions.verify_session_id(renewed) == raw  # same conversation, new expiry


def test_voice_token_is_rate_limited_and_origin_checked(client, settings_env):
    assert client.post("/voice/token", json={}, headers={"origin": "https://evil.example"}).status_code == 403
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_VOICE_TOKEN_PER_MINUTE="2")
    codes = [client.post("/voice/token", json={}, headers={"origin": "http://testserver"}).status_code for _ in range(4)]
    assert codes == [200, 200, 429, 429]


def test_oversized_request_bodies_are_rejected_by_validation(client):
    assert client.post("/api/leads", json={"name": "a" * 500, "phone": "01012345678"}).status_code == 422
    assert client.post("/api/leads", json={"name": "a", "phone": "01012345678", "message": "m" * 3000}).status_code == 422
