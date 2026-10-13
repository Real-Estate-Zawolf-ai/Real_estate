"""Drive the website in a real Chrome and check the things a unit test cannot.

    pip install playwright          # uses the Chrome already on the machine (no download)
    python scripts/browser_smoke.py --site https://aqar.pages.dev [--expect-listings 3]

Checks, in order:
  * the home and listing pages render with no JavaScript errors and no Content-Security-Policy violations
  * live listings load cross-origin from the API (CORS + API_BASE_URL are right)
  * markup hidden in listing data and in the URL is shown as text and never runs
  * the built-in sample inventory is not shown to visitors
  * the chat widget opens, sends a message and shows the assistant's reply
  * the microphone button is present (a real call still needs a human and a microphone)

Exit status 1 if any check fails. Nothing here sends WhatsApp messages or books anything.
"""

from __future__ import annotations

import argparse
import sys
from urllib.parse import quote

CHECKS: list[tuple[str, bool, str]] = []
SAMPLE_IDS = {"mivida-courtyard", "zayed-villa", "north-coast", "district-office"}  # the built-in sample listings


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    print(f"  [{'OK ' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def guarded(name: str, action) -> None:
    """Run one section; a timeout or missing element is a failed check, not a crash."""
    try:
        action()
    except Exception as error:  # playwright TimeoutError / Error
        check(name, False, f"{type(error).__name__}: {str(error).splitlines()[0][:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True, help="website origin, e.g. https://aqar.pages.dev")
    parser.add_argument("--channel", default="chrome", help="installed browser channel (chrome | msedge)")
    parser.add_argument("--expect-listings", type=int, default=0, help="minimum live listings expected on the listing page")
    parser.add_argument("--expect-xss-listing", action="store_true", help="a hostile test listing (markup in its fields) is indexed")
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()
    site = args.site.rstrip("/")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("playwright is not installed: pip install playwright")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        context = browser.new_context(permissions=[], viewport={"width": 1280, "height": 900})
        page = context.new_page()
        problems: list[str] = []
        page.on("pageerror", lambda error: problems.append(f"js error: {error}"))
        page.on("console", lambda m: problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        page.add_init_script("window.__pwned = 0;")

        def home() -> None:
            print(f"Home page: {site}/")
            page.goto(f"{site}/", wait_until="networkidle")
            check("home page renders", page.locator("main").count() > 0, page.title())
            check("no sample-inventory banner", page.locator(".dev-note").count() == 0)

        def listing() -> None:
            print("Listing page")
            page.goto(f"{site}/#/properties", wait_until="networkidle")
            page.wait_for_timeout(800)
            cards = page.locator(".card").count()
            check("live listings load from the API", cards >= args.expect_listings, f"{cards} card(s)")
            hrefs = page.eval_on_selector_all('a[href^="#/properties/"]', "els => els.map(e => e.getAttribute('href'))")
            linked = {href.rsplit("/", 1)[-1] for href in hrefs}
            check("no invented sample listings are shown", not (linked & SAMPLE_IDS), f"sample ids linked: {sorted(linked & SAMPLE_IDS)}")
            body = page.inner_text("body")
            check("a missing price reads 'Price on request', never 'EGP 0'", "EGP 0" not in body and "EGP 0" not in body)
            if args.expect_xss_listing:
                check("markup in listing data did not execute", page.evaluate("window.__pwned") == 0, f"__pwned={page.evaluate('window.__pwned')}")
                check("hostile listing is displayed as text", "Evil Villa" in body)

        def url_markup() -> None:
            print("URL-borne markup")
            page.goto(f"{site}/#/properties?location={quote('<img src=x onerror=window.__pwned=3>')}")
            page.reload(wait_until="networkidle")
            page.wait_for_timeout(500)
            check("markup in the URL did not execute", page.evaluate("window.__pwned") == 0, f"__pwned={page.evaluate('window.__pwned')}")
            check("URL value is shown as plain text", "<img src=x" in page.inner_text(".filters", timeout=5000))

        def chat() -> None:
            print("Chat")
            page.goto(f"{site}/", wait_until="networkidle")
            page.locator(".cb-launcher").click()
            page.wait_for_selector("#cb-panel:not([hidden])", timeout=10000)
            textarea = page.locator("#cb-panel textarea")
            textarea.fill("hello")
            textarea.press("Enter")
            # Free-tier hosts need up to a minute to wake: wait for the reply, not a fixed time.
            page.wait_for_function(
                "document.querySelectorAll('#cb-panel .cb-bubble').length >= 3 || document.querySelector('.cb-bubble--error')",
                timeout=90000,
            )
            page.wait_for_timeout(500)
            bubbles = page.locator("#cb-panel .cb-bubble").all_inner_texts()
            check("assistant replied", len(bubbles) >= 3 and page.locator(".cb-bubble--error").count() == 0, f"{len(bubbles)} bubbles")
            check("microphone button present", page.locator(".cb-voice-btn").count() > 0)
            stored = page.evaluate("JSON.stringify(sessionStorage) + JSON.stringify(localStorage)")
            check("no provider key reaches the browser", not any(marker in stored for marker in ("AIza", "gsk_", "EAA", "sk-")))

        guarded("home page", home)
        guarded("listing page", listing)
        guarded("URL-borne markup", url_markup)
        if not args.skip_chat:
            guarded("chat", chat)

        csp = [p for p in problems if "Content Security Policy" in p or "Content-Security-Policy" in p]
        check("no Content-Security-Policy violations", not csp, "; ".join(csp)[:1200])
        other = [p for p in problems if p not in csp and "favicon" not in p.lower()]
        check("no JavaScript errors on any page", not other, "; ".join(other)[:600])
        browser.close()

    failed = [name for name, passed, _ in CHECKS if not passed]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
