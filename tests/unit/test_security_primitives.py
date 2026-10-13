"""Rate limiting, client identification, session/token signing and log redaction."""

import json
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import ratelimit
from app.api.ratelimit import SlidingWindowLimiter, client_ip
from app.api import sessions
from app.observability.logging import JsonFormatter, RedactingFilter, redact


def request(peer="10.1.1.1", forwarded=None):
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer), url=SimpleNamespace(path="/x"))


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def test_window_allows_up_to_the_limit_then_reports_when_to_retry():
    limiter = SlidingWindowLimiter()
    assert [limiter.hit("k", 3, 60) for _ in range(3)] == [0.0, 0.0, 0.0]
    retry = limiter.hit("k", 3, 60)
    assert 0 < retry <= 60
    assert limiter.hit("other", 3, 60) == 0.0  # keys are independent


def test_window_slides_instead_of_resetting_in_blocks(monkeypatch):
    limiter = SlidingWindowLimiter()
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    assert limiter.hit("k", 2, 10) == 0 and limiter.hit("k", 2, 10) == 0
    now[0] += 9
    assert limiter.hit("k", 2, 10) > 0  # still full
    now[0] += 1.5
    assert limiter.hit("k", 2, 10) == 0  # the first hit aged out


def test_memory_is_bounded_when_an_attacker_rotates_identities(monkeypatch):
    limiter = SlidingWindowLimiter(max_keys=50)
    for index in range(500):
        limiter.hit(f"ip-{index}", 5, 60)
    assert len(limiter._hits) <= 50 + 1


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------


def test_without_a_trusted_proxy_forwarded_headers_are_ignored(settings_env):
    settings_env(TRUSTED_PROXY_HOPS="0")
    assert client_ip(request(peer="10.1.1.1", forwarded="6.6.6.6")) == "10.1.1.1"


def test_behind_one_proxy_the_rightmost_entry_is_the_client_and_forged_ones_are_ignored(settings_env):
    settings_env(TRUSTED_PROXY_HOPS="1")
    honest = client_ip(request(forwarded="203.0.113.5"))
    spoofed = client_ip(request(forwarded="1.1.1.1, 2.2.2.2, 203.0.113.5"))
    assert honest == spoofed == "203.0.113.5"  # prepending fake hops buys an attacker nothing


def test_two_trusted_proxies_pick_the_entry_before_the_last(settings_env):
    settings_env(TRUSTED_PROXY_HOPS="2")
    assert client_ip(request(forwarded="9.9.9.9, 203.0.113.5, 198.51.100.1")) == "203.0.113.5"


def test_a_short_header_falls_back_safely_and_garbage_cannot_blow_up(settings_env):
    settings_env(TRUSTED_PROXY_HOPS="3")
    assert client_ip(request(forwarded="203.0.113.5")) == "203.0.113.5"
    assert client_ip(request(forwarded="x" * 500)) == "x" * 64  # truncated, never unbounded
    assert client_ip(SimpleNamespace(headers={}, client=None)) == "unknown"


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


def test_exceeding_a_limit_raises_429_with_retry_after(settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_SEARCH_PER_MINUTE="2")
    ratelimit.limit_search(request())
    ratelimit.limit_search(request())
    with pytest.raises(HTTPException) as error:
        ratelimit.limit_search(request())
    assert error.value.status_code == 429
    assert int(error.value.headers["Retry-After"]) >= 1
    ratelimit.limit_search(request(peer="10.9.9.9"))  # another client is unaffected


def test_disabled_limiting_never_blocks(settings_env):
    settings_env(RATE_LIMIT_ENABLED="false", RATE_LIMIT_SEARCH_PER_MINUTE="1")
    for _ in range(20):
        ratelimit.limit_search(request())


def test_chat_is_limited_per_ip_and_per_session(settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_CHAT_PER_MINUTE="100", RATE_LIMIT_CHAT_SESSION_PER_MINUTE="2")
    ratelimit.limit_chat(request(), "session-a")
    ratelimit.limit_chat(request(), "session-a")
    with pytest.raises(HTTPException):
        ratelimit.limit_chat(request(), "session-a")  # one chatty session
    ratelimit.limit_chat(request(), "session-b")  # a different session from the same IP is fine


def test_voice_tokens_are_limited_per_ip_and_per_session(settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_VOICE_TOKEN_PER_MINUTE="2")
    ratelimit.limit_voice_token(request(), "s1")
    ratelimit.limit_voice_token(request(), "s1")
    with pytest.raises(HTTPException):
        ratelimit.limit_voice_token(request(), "s1")


def test_repeated_wrong_admin_keys_lock_the_client_out_even_for_the_right_key(settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_ADMIN_FAILURES_PER_15MIN="3", RATE_LIMIT_ADMIN_PER_MINUTE="1000")
    attacker = request(peer="10.6.6.6")
    for _ in range(3):
        ratelimit.limit_admin(attacker)
        ratelimit.record_admin_failure(attacker)
    with pytest.raises(HTTPException) as error:
        ratelimit.limit_admin(attacker)
    assert error.value.status_code == 429
    ratelimit.limit_admin(request(peer="10.7.7.7"))  # the operator on another address still works


def test_whatsapp_flood_is_dropped_per_sender(settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_WHATSAPP_SENDER_PER_MINUTE="3")
    assert [ratelimit.allow_whatsapp_sender("2010") for _ in range(5)] == [True, True, True, False, False]
    assert ratelimit.allow_whatsapp_sender("2011") is True


# ---------------------------------------------------------------------------
# Sessions and voice tokens
# ---------------------------------------------------------------------------


def test_session_ids_verify_and_survive_resigning():
    session_id = sessions.issue_session_id()
    raw = sessions.verify_session_id(session_id)
    assert raw and sessions.website_thread(raw) == f"web:{raw}"
    refreshed = sessions.issue_session_id(raw)
    assert sessions.verify_session_id(refreshed) == raw  # same conversation, fresh timestamp


@pytest.mark.parametrize("mutate", [
    lambda raw, stamp, sig: f"{raw}x.{stamp}.{sig}",          # different conversation, old signature
    lambda raw, stamp, sig: f"{raw}.{stamp}0.{sig}",          # forged timestamp
    lambda raw, stamp, sig: f"{raw}.{stamp}.{sig[:-1]}{'1' if sig[-1] == '0' else '0'}",  # corrupted signature (always differs)
    lambda raw, stamp, sig: f"{raw}.{stamp}",                 # old two-part format
    lambda raw, stamp, sig: f"{raw}.{stamp}.{sig}.extra",
    lambda raw, stamp, sig: "",
])
def test_tampered_session_ids_are_rejected(mutate):
    raw, stamp, sig = sessions.issue_session_id().split(".")
    assert sessions.verify_session_id(mutate(raw, stamp, sig)) is None


def test_session_ids_expire_and_report_when_to_refresh(settings_env):
    settings_env(SESSION_MAX_AGE_DAYS="30")
    long_ago = time.time() - 31 * 86400
    assert sessions.verify_session_id(sessions.issue_session_id(now=long_ago)) is None
    old_but_valid = sessions.issue_session_id(now=time.time() - 3 * 86400)
    raw, age = sessions.inspect_session_id(old_but_valid)
    assert raw and sessions.needs_refresh(age)
    assert not sessions.needs_refresh(sessions.inspect_session_id(sessions.issue_session_id())[1])
    assert sessions.verify_session_id(sessions.issue_session_id(now=time.time() + 3600)) is None  # from the future


def test_a_different_server_secret_invalidates_every_session(settings_env):
    session_id = sessions.issue_session_id()
    settings_env(SESSION_SECRET="another-secret-0123456789abcdef0123456789")
    sessions.reset_secret_cache()
    assert sessions.verify_session_id(session_id) is None


def test_voice_tokens_expire_and_only_work_once():
    token = sessions.issue_voice_token("raw-session", "ar")
    claims = sessions.consume_voice_token(token)
    assert claims and claims["sid"] == "raw-session" and claims["lang"] == "ar"
    assert sessions.consume_voice_token(token) is None  # replay


def test_expired_and_forged_voice_tokens_are_rejected(settings_env):
    settings_env(VOICE_TOKEN_TTL_SECONDS="-5")
    assert sessions.consume_voice_token(sessions.issue_voice_token("s", "en")) is None
    settings_env(VOICE_TOKEN_TTL_SECONDS="60")
    token = sessions.issue_voice_token("s", "en")
    body, signature = token.split(".")
    assert sessions.verify_voice_token(f"{body}.{'0' * len(signature)}") is None
    assert sessions.verify_voice_token(None) is None
    assert sessions.verify_voice_token("x" * 600) is None


def test_voice_token_language_is_restricted():
    claims = sessions.verify_voice_token(sessions.issue_voice_token("s", "fr"))
    assert claims["lang"] == "en"


def test_origin_check_allows_listed_origins_only(settings_env):
    settings_env(CORS_ALLOWED_ORIGINS="https://app.example.com,https://www.example.com/")
    assert sessions.origin_allowed("https://app.example.com")
    assert sessions.origin_allowed("https://www.example.com/")
    assert not sessions.origin_allowed("https://evil.example")
    assert not sessions.origin_allowed(None)  # browsers always send Origin on WebSocket handshakes
    settings_env(CORS_ALLOWED_ORIGINS="*")
    assert sessions.origin_allowed(None)


# ---------------------------------------------------------------------------
# Log redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leak", [
    "GET /website/voice?token=eyJzaWQiOiJhYmMifQ.0123456789abcdef",
    "Authorization: Bearer EAAGm0PX4ZCpsBAKzXyzABCDEFGHIJKLMN",
    "x-admin-key: super-secret-admin-key-value",
    "POST https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/sendMessage",
    "key gsk_AbCdEfGhIjKlMnOpQrStUvWx failed",
    "using AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY for embeddings",
    "connecting postgresql://user:hunter2@ep-x.neon.tech/db",
    '{"api_key": "abc123secretvalue"}',
    "password=hunter2&user=bob",
])
def test_credentials_are_scrubbed_from_log_text(leak):
    scrubbed = redact(leak)
    for secret in ("eyJzaWQi", "EAAGm0PX", "super-secret", "AAHdqTcv", "AbCdEfGh", "AIzaSyD", "hunter2", "abc123secretvalue"):
        assert secret not in scrubbed


def test_redaction_leaves_normal_messages_alone():
    message = "Booking 8f3a created for property 501189949 at 2026-10-20 16:00"
    assert redact(message) == message


def test_the_logging_filter_scrubs_message_arguments_and_exception_text():
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "call failed: %s", ("token=abc123",), None)
    assert RedactingFilter().filter(record) and "abc123" not in record.getMessage()
    try:
        raise RuntimeError("bad url https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/x")
    except RuntimeError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    RedactingFilter().filter(record)
    assert "AAHdqTcv" not in JsonFormatter().format(record)


def test_json_logs_are_one_parseable_object_per_line():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    line = JsonFormatter().format(record)
    assert "\n" not in line
    assert json.loads(line)["message"] == "hello world"


def test_uvicorn_websocket_lines_do_not_leak_the_voice_token(capsys):
    from app.observability import configure_logging

    configure_logging(level="INFO", fmt="json")
    # uvicorn logs the upgrade request, query string included, on its own logger.
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" 403', "10.0.0.1:5000", "/website/voice?token=eyJzaWQiOiJzZWNyZXQifQ.abcdef"
    )
    output = capsys.readouterr().out
    assert "eyJzaWQi" not in output
    assert "/website/voice?token=[redacted]" in output
