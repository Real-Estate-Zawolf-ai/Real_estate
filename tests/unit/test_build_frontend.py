"""The static-site build for Cloudflare Pages / Vercel (scripts/build_frontend.py)."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_frontend", ROOT / "scripts" / "build_frontend.py")
build_frontend = importlib.util.module_from_spec(spec)
sys.modules["build_frontend"] = build_frontend
spec.loader.exec_module(build_frontend)

API = "https://aqar-api.onrender.com"


@pytest.fixture
def built(tmp_path):
    out = tmp_path / "dist"
    config = build_frontend.build(out, {"API_BASE_URL": API})
    return out, config


def test_files_are_laid_out_the_way_index_html_expects(built):
    out, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")
    for reference in re.findall(r'(?:src|href)="(/assets/[^"]+)"', html):
        assert (out / reference.lstrip("/")).exists(), reference
    assert (out / "assets" / "chat" / "voice-worklet.js").exists()  # loaded relative to voice.js
    assert (out / "_headers").exists()
    assert not (out / "assets" / "index.html").exists()


def test_runtime_config_points_the_site_at_the_api_and_hides_sample_listings(built):
    out, config = built
    js = (out / "assets" / "runtime-config.js").read_text(encoding="utf-8")
    parsed = json.loads(js.split("window.APP_CONFIG = ", 1)[1].rstrip().rstrip(";"))
    assert parsed == config
    assert parsed["API_BASE_URL"] == API
    assert parsed["SHOW_PREVIEW_INVENTORY"] is False  # production never shows invented listings
    assert parsed["CHAT_DEFAULT_LANG"] == "ar" and parsed["VOICE_ENABLED"] is True


def test_the_csp_allows_only_self_and_the_api(built):
    out, _ = built
    csp = re.search(r"Content-Security-Policy: (.+)", (out / "_headers").read_text(encoding="utf-8")).group(1)
    assert "script-src 'self'" in csp and "'unsafe-eval'" not in csp
    assert "script-src 'self' 'unsafe-inline'" not in csp  # inline scripts stay blocked: the XSS safety net
    assert f"connect-src 'self' {API} wss://aqar-api.onrender.com" in csp
    assert "object-src 'none'" in csp and "frame-ancestors 'none'" in csp and "base-uri 'none'" in csp
    # The brand fonts load from Google Fonts: allowed precisely, not via a blanket https: for styles/fonts.
    assert "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com" in csp
    assert "font-src 'self' data: https://fonts.gstatic.com" in csp
    assert "worklet-src" not in csp  # not a directive Chrome implements: it only logs an error
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'http-equiv="Content-Security-Policy"' in html  # also delivered as <meta> for hosts that ignore _headers


def test_headers_file_has_the_hardening_headers_and_short_cache_rules(built):
    headers = (built[0] / "_headers").read_text(encoding="utf-8")
    for expected in ("X-Content-Type-Options: nosniff", "X-Frame-Options: DENY", "Strict-Transport-Security:", "microphone=(self)",
                     "/assets/runtime-config.js\n  Cache-Control: no-cache"):
        assert expected in headers


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "required"),
        ("", "required"),
        ("not a url", "absolute"),
        ("ftp://api.example.com", "absolute"),
        ("http://api.example.com", "https"),
        ("https://api.example.com/v1", "origin only"),
        ("https://api.example.com/?x=1", "origin only"),
    ],
)
def test_a_wrong_api_url_fails_the_build_instead_of_shipping_a_broken_site(tmp_path, value, message):
    with pytest.raises(build_frontend.BuildError, match=message):
        build_frontend.build(tmp_path / "dist", {"API_BASE_URL": value} if value is not None else {})


def test_local_http_api_is_allowed_only_for_localhost(tmp_path):
    build_frontend.build(tmp_path / "a", {"API_BASE_URL": "http://localhost:8000"})
    with pytest.raises(build_frontend.BuildError):
        build_frontend.build(tmp_path / "b", {"API_BASE_URL": "http://192.168.1.5:8000"})
    build_frontend.build(tmp_path / "c", {"API_BASE_URL": "http://192.168.1.5:8000"}, allow_insecure=True)


def test_trailing_slash_is_normalised(tmp_path):
    assert build_frontend.build(tmp_path / "d", {"API_BASE_URL": API + "/"})["API_BASE_URL"] == API


def test_other_options_are_validated(tmp_path):
    with pytest.raises(build_frontend.BuildError):
        build_frontend.build(tmp_path / "e", {"API_BASE_URL": API, "CHAT_DEFAULT_LANG": "fr"})
    with pytest.raises(build_frontend.BuildError):
        build_frontend.build(tmp_path / "f", {"API_BASE_URL": API, "VOICE_ENABLED": "maybe"})
    config = build_frontend.build(tmp_path / "g", {"API_BASE_URL": API, "VOICE_ENABLED": "false", "CHAT_DEFAULT_LANG": "en",
                                                   "REQUEST_TIMEOUT_MS": "90000", "SHOW_PREVIEW_INVENTORY": "true"})
    assert (config["VOICE_ENABLED"], config["CHAT_DEFAULT_LANG"], config["REQUEST_TIMEOUT_MS"], config["SHOW_PREVIEW_INVENTORY"]) == (False, "en", 90000, True)


def test_config_values_cannot_break_out_of_the_script(tmp_path):
    js = build_frontend.runtime_config_js({"API_BASE_URL": 'https://x.example"</script><script>alert(1)//'})
    assert "</script>" not in js  # JSON-escaped, so a hostile value cannot inject markup


def test_a_rebuild_replaces_the_previous_output(tmp_path):
    out = tmp_path / "dist"
    build_frontend.build(out, {"API_BASE_URL": API})
    (out / "stale.txt").write_text("old")
    build_frontend.build(out, {"API_BASE_URL": API})
    assert not (out / "stale.txt").exists()


def test_the_site_source_never_contains_credentials_or_the_real_api_url():
    leaks = []
    pattern = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[A-Za-z0-9]{10,}|EAA[A-Za-z0-9]{20,}|onrender\.com|neon\.tech|qdrant\.io)")
    for path in (ROOT / "app" / "frontend").rglob("*"):
        if path.is_file() and path.suffix in {".js", ".html", ".css"}:
            leaks += [f"{path.name}: {m}" for m in pattern.findall(path.read_text(encoding="utf-8"))]
    assert leaks == []
