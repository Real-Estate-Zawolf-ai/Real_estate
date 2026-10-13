"""Post-deploy smoke test for the running API. Safe to run against production.

    python scripts/smoke_prod.py --api https://aqar-api.onrender.com --origin https://aqar.pages.dev

Optional environment (never printed): ADMIN_API_KEY (checks the staff API),
WHATSAPP_VERIFY_TOKEN (checks Meta's webhook handshake).

It reads data and sends one harmless chat message ("hello"). It never books,
never sends a WhatsApp message and never changes configuration. A free Render
instance may need ~50 seconds to wake, so the first request waits for it.
Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    print(f"  [{'OK ' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def wake(client: httpx.Client, api: str) -> bool:
    """A sleeping free instance answers the first request slowly (or 502/503 while booting)."""
    deadline = time.monotonic() + 120
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            if client.get(f"{api}/health", timeout=30).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        print(f"  ... waiting for the API to wake (attempt {attempt})", flush=True)
        time.sleep(5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", required=True, help="API origin, e.g. https://aqar-api.onrender.com")
    parser.add_argument("--origin", required=True, help="the website origin allowed by CORS_ALLOWED_ORIGINS")
    parser.add_argument("--expect-indexed", action="store_true", help="also require /health/ready to be fully ready (index populated)")
    args = parser.parse_args()
    api = args.api.rstrip("/")
    admin_key = os.environ.get("ADMIN_API_KEY")
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN")

    with httpx.Client(follow_redirects=False) as client:
        print(f"API: {api}")
        awake = wake(client, api)
        check("API is up", awake)
        if not awake:
            return 1
        check("served over HTTPS", api.startswith("https://"), "WebSockets and the microphone need a secure origin")

        ready = client.get(f"{api}/health/ready", timeout=60)
        body = ready.json() if ready.headers.get("content-type", "").startswith("application/json") else {}
        checks = body.get("checks", {})
        for name in ("database", "migrations", "conversation_memory", "vector_search", "providers"):
            if name in checks:
                detail = checks[name].get("error") or checks[name].get("detail") or ""
                severe = name != "vector_search" or args.expect_indexed
                check(f"readiness: {name}", checks[name].get("ok", False) or not severe, detail or ("ok" if checks[name].get("ok") else "not ready"))
        check("readiness overall", ready.status_code == 200 or not args.expect_indexed, f"HTTP {ready.status_code}")

        for path in ("/docs", "/openapi.json", "/redoc"):
            check(f"{path} is not exposed", client.get(f"{api}{path}", timeout=30).status_code == 404)
        check("/metrics requires credentials", client.get(f"{api}/metrics/", timeout=30).status_code in (401, 403, 503))

        headers = client.get(f"{api}/health", timeout=30).headers
        check("security headers present", headers.get("x-content-type-options") == "nosniff" and "strict-transport-security" in headers)

        preflight = client.options(f"{api}/chat", headers={"Origin": args.origin, "Access-Control-Request-Method": "POST",
                                                              "Access-Control-Request-Headers": "content-type"}, timeout=30)
        check("CORS allows the website origin", preflight.headers.get("access-control-allow-origin") == args.origin)
        evil = client.options(f"{api}/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}, timeout=30)
        check("CORS refuses other origins", "access-control-allow-origin" not in evil.headers)

        chat = client.post(f"{api}/chat", json={"query": "hello", "language": "en"}, headers={"Origin": args.origin}, timeout=90)
        check("chat answers", chat.status_code == 200 and bool(chat.json().get("response")), f"HTTP {chat.status_code}")
        if chat.status_code == 200:
            session_id = chat.json()["session_id"]
            token = client.post(f"{api}/voice/token", json={"session_id": session_id, "language": "en"}, headers={"Origin": args.origin}, timeout=30)
            check("voice token issued (or voice cleanly disabled)", token.status_code in (200, 503), f"HTTP {token.status_code}")
            forged = session_id[:-1] + ("0" if session_id[-1] != "0" else "1")
            again = client.post(f"{api}/chat", json={"query": "hello", "session_id": forged}, timeout=60).json()
            check("a forged session id is not honoured", again.get("session_id") != forged)

        check("staff API rejects anonymous calls", client.get(f"{api}/admin/bookings", timeout=30).status_code in (401, 503))
        check("staff API rejects a wrong key", client.get(f"{api}/admin/bookings", headers={"X-Admin-Key": "definitely-wrong"}, timeout=30).status_code in (401, 503))
        if admin_key:
            good = client.get(f"{api}/admin/bookings", headers={"X-Admin-Key": admin_key}, timeout=30)
            check("staff API accepts the real key", good.status_code == 200, f"HTTP {good.status_code}")
            who = client.get(f"{api}/admin/diagnostics/client-ip", headers={"X-Admin-Key": admin_key}, timeout=30).json()
            check("rate limiter sees your real address (not the proxy's)", who.get("resolved") not in (None, who.get("peer")),
                  f"resolved={who.get('resolved')} peer={who.get('peer')} hops={who.get('trusted_proxy_hops')}")
        else:
            print("  (set ADMIN_API_KEY to also check the staff API and client-IP detection)")

        if verify_token:
            handshake = client.get(f"{api}/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": verify_token, "hub.challenge": "smoke-123"}, timeout=30)
            check("WhatsApp webhook handshake", handshake.status_code == 200 and handshake.text == "smoke-123", f"HTTP {handshake.status_code}")
        wrong = client.get(f"{api}/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"}, timeout=30)
        check("WhatsApp webhook refuses a wrong verify token", wrong.status_code == 403)
        unsigned = client.post(f"{api}/whatsapp/webhook", content=b"{}", headers={"Content-Type": "application/json"}, timeout=30)
        check("WhatsApp webhook refuses unsigned events", unsigned.status_code == 401)

    failed = [name for name, passed, _ in RESULTS if not passed]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
