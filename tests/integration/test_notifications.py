"""Staff alerts: durable outbox, fan-out to channels, retries, and no credential leaks."""

from __future__ import annotations

import httpx
import pytest

from app.database import models as db
from app.database.session import session_scope
from app.notifications import staff
from app.notifications.staff import notify_staff, retry_pending

TELEGRAM_TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"


class FakeHttp:
    """Stands in for httpx.post and records every call."""

    def __init__(self):
        self.calls: list[dict] = []
        self.fail: set[str] = set()  # substrings of URLs that should fail
        self.status = 500

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if any(part in url for part in self.fail):
            request = httpx.Request("POST", url)
            raise httpx.HTTPStatusError(f"Server error for url '{url}'", request=request, response=httpx.Response(self.status, request=request))
        return httpx.Response(200, request=httpx.Request("POST", url))


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(httpx, "post", fake)
    return fake


def outbox():
    with session_scope() as session:
        return [
            {"kind": row.kind, "status": row.status, "attempts": row.attempts, "error": row.last_error, "message": row.message}
            for row in session.query(db.StaffNotification).order_by(db.StaffNotification.id)
        ]


def test_an_alert_is_stored_even_when_no_channel_is_configured(http):
    assert notify_staff("booking_requested", "New viewing request", ref_type="booking", ref_id="b1") is not None
    assert outbox() == [{"kind": "booking_requested", "status": "no_channel", "attempts": 1, "error": None, "message": "New viewing request"}]
    assert http.calls == []  # nothing is sent anywhere, yet nothing is lost


def test_alerts_reach_every_configured_channel(http, settings_env):
    settings_env(STAFF_WEBHOOK_URL="https://hooks.example.com/T/B/X", TELEGRAM_BOT_TOKEN=TELEGRAM_TOKEN, TELEGRAM_CHAT_ID="-100123")
    notify_staff("handoff_requested", "Customer wants a human")

    urls = [call["url"] for call in http.calls]
    assert "https://hooks.example.com/T/B/X" in urls
    assert f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" in urls
    webhook = next(call for call in http.calls if "hooks.example.com" in call["url"])
    assert webhook["json"] == {"text": "Customer wants a human", "content": "Customer wants a human"}  # Slack and Discord
    assert outbox()[0]["status"] == "sent"


def test_one_working_channel_is_enough_and_the_failure_is_recorded(http, settings_env):
    settings_env(STAFF_WEBHOOK_URL="https://hooks.example.com/ok", TELEGRAM_BOT_TOKEN=TELEGRAM_TOKEN, TELEGRAM_CHAT_ID="1")
    http.fail = {"api.telegram.org"}
    notify_staff("booking_requested", "msg")
    row = outbox()[0]
    assert row["status"] == "sent"
    assert "telegram: HTTP 500" in row["error"]


def test_error_text_never_contains_the_bot_token(http, settings_env, caplog):
    settings_env(TELEGRAM_BOT_TOKEN=TELEGRAM_TOKEN, TELEGRAM_CHAT_ID="1")
    http.fail = {"api.telegram.org"}
    with caplog.at_level("WARNING"):
        notify_staff("booking_requested", "msg")
    row = outbox()[0]
    assert row["status"] == "failed"
    assert TELEGRAM_TOKEN not in (row["error"] or "")  # the httpx message embeds the URL; only the status is kept
    assert TELEGRAM_TOKEN not in caplog.text


def test_failed_alerts_are_retried_and_eventually_delivered(http, settings_env):
    settings_env(STAFF_WEBHOOK_URL="https://hooks.example.com/flaky")
    http.fail = {"hooks.example.com"}
    notify_staff("booking_requested", "msg")
    assert outbox()[0]["status"] == "failed" and outbox()[0]["attempts"] == 1

    http.fail = set()  # the provider recovers
    assert retry_pending() == 1
    assert outbox()[0]["status"] == "sent"
    assert retry_pending() == 0  # delivered alerts are not sent twice


def test_alerts_stored_before_a_channel_existed_are_delivered_once_one_is_added(http, settings_env):
    notify_staff("handoff_requested", "early alert")
    assert outbox()[0]["status"] == "no_channel"
    settings_env(STAFF_WEBHOOK_URL="https://hooks.example.com/new")
    assert retry_pending() == 1
    assert outbox()[0]["status"] == "sent"


def test_a_permanently_failing_alert_is_abandoned_after_the_attempt_limit(http, settings_env):
    settings_env(STAFF_WEBHOOK_URL="https://hooks.example.com/dead")
    http.fail = {"hooks.example.com"}
    notify_staff("booking_requested", "msg")
    for _ in range(staff.MAX_ATTEMPTS + 2):
        retry_pending()
    row = outbox()[0]
    assert row["status"] == "abandoned" and row["attempts"] == staff.MAX_ATTEMPTS
    calls = len(http.calls)
    retry_pending()
    assert len(http.calls) == calls  # no more attempts


def test_email_channel_uses_starttls_and_login(http, settings_env, monkeypatch):
    settings_env(SMTP_HOST="smtp.example.com", SMTP_FROM="bot@example.com", STAFF_EMAIL_TO="sales@example.com",
                 SMTP_USERNAME="user", SMTP_PASSWORD="pw")
    events = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            events.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            events.append(("starttls",))

        def login(self, user, password):
            events.append(("login", user))

        def send_message(self, message):
            events.append(("send", message["To"], message["Subject"]))

    monkeypatch.setattr(staff.smtplib, "SMTP", FakeSMTP)
    notify_staff("booking_requested", "New viewing request\nProperty 1")
    assert events == [("connect", "smtp.example.com", 587), ("starttls",), ("login", "user"), ("send", "sales@example.com", "New viewing request")]
    assert outbox()[0]["status"] == "sent"


def test_a_database_failure_never_propagates_into_the_customer_request(http, monkeypatch):
    def broken():
        raise RuntimeError("database is down")

    monkeypatch.setattr(staff, "session_scope", broken)
    assert notify_staff("booking_requested", "msg") is None  # swallowed, logged


def test_long_messages_are_truncated(http):
    notify_staff("booking_requested", "x" * 10_000)
    assert len(outbox()[0]["message"]) == staff.MAX_MESSAGE_CHARS
