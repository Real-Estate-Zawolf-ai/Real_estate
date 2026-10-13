"""Meta WhatsApp Cloud API webhook: signatures, duplicates, retries, floods, odd payloads.

Meta itself is simulated (signed payloads in Meta's documented shape); the
FastAPI app, the agent graph and PostgreSQL/SQLite storage are the real code.
These tests prove our side of the contract, not that Meta accepts our token.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

import app.api.routes.webhooks as webhooks
from app.database import models as db
from app.database.session import session_scope
from tests.integration.test_api_channels import client  # noqa: F401  (fixture)

SECRET = "app-secret"
PHONE_ID = "1234567890"


@pytest.fixture
def wa(client, stub_rag, settings_env, monkeypatch):
    settings_env(WHATSAPP_APP_SECRET=SECRET, WHATSAPP_VERIFY_TOKEN="verify-me", WHATSAPP_PHONE_NUMBER_ID=PHONE_ID,
                 WHATSAPP_ACCESS_TOKEN="token")
    sent: list[tuple[str, str]] = []

    async def fake_send(recipient, text):
        sent.append((recipient, text))

    monkeypatch.setattr(webhooks, "send_text_message", fake_send)

    class Meta:
        def payload(self, *messages, phone_number_id=PHONE_ID, extra=None):
            value = {"messaging_product": "whatsapp", "metadata": {"phone_number_id": phone_number_id}, "messages": list(messages)}
            value.update(extra or {})
            return json.dumps({"object": "whatsapp_business_account", "entry": [{"id": "1", "changes": [{"field": "messages", "value": value}]}]}).encode()

        def post(self, body, secret=SECRET, signature=None):
            sig = signature or "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            return client.post("/whatsapp/webhook", content=body, headers={"X-Hub-Signature-256": sig, "Content-Type": "application/json"})

        def text(self, message_id, body, sender="201000000001"):
            return {"id": message_id, "from": sender, "timestamp": "1760000000", "type": "text", "text": {"body": body}}

    meta = Meta()
    meta.sent = sent
    meta.client = client
    return meta


def claims():
    with session_scope() as session:
        return {(row.kind, row.key): row.status for row in session.query(db.ReplayGuard)}


# ---------------------------------------------------------------------------
# Handshake and authenticity
# ---------------------------------------------------------------------------


def test_verification_handshake_echoes_the_challenge_only_for_the_right_token(wa):
    ok = wa.client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "98765"})
    assert (ok.status_code, ok.text) == (200, "98765")
    assert ok.headers["content-type"].startswith("text/plain")
    for params in (
        {"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "1"},
        {"hub.mode": "unsubscribe", "hub.verify_token": "verify-me", "hub.challenge": "1"},
        {"hub.mode": "subscribe", "hub.verify_token": "verify-me"},
        {},
    ):
        assert wa.client.get("/whatsapp/webhook", params=params).status_code == 403


def test_handshake_is_refused_when_no_verify_token_is_configured(wa, settings_env):
    settings_env(WHATSAPP_VERIFY_TOKEN="")
    assert wa.client.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"}).status_code == 403


@pytest.mark.parametrize("signature", [None, "", "sha256=", "sha256=deadbeef", "md5=abc", "SHA256=" + "0" * 64])
def test_bad_or_missing_signatures_are_rejected_without_running_the_agents(wa, signature):
    body = wa.payload(wa.text("wamid.a", "villa in Sheikh Zayed"))
    response = wa.client.post("/whatsapp/webhook", content=body, headers={"Content-Type": "application/json",
                                                                          **({"X-Hub-Signature-256": signature} if signature is not None else {})})
    assert response.status_code == 401
    assert wa.sent == [] and claims() == {}


def test_a_payload_signed_with_the_wrong_secret_is_rejected(wa):
    assert wa.post(wa.payload(wa.text("wamid.b", "hi")), secret="not-the-secret").status_code == 401


def test_a_tampered_body_fails_the_signature(wa):
    body = wa.payload(wa.text("wamid.c", "hi"))
    signature = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert wa.post(body.replace(b'"hi"', b'"ho"'), signature=signature).status_code == 401


def test_without_an_app_secret_every_event_is_rejected(wa, settings_env):
    settings_env(WHATSAPP_APP_SECRET="")
    assert wa.post(wa.payload(wa.text("wamid.d", "hi"))).status_code == 401  # fails closed


def test_malformed_json_with_a_valid_signature_is_a_400(wa):
    assert wa.post(b"{ not json").status_code == 400


# ---------------------------------------------------------------------------
# Duplicates and retries
# ---------------------------------------------------------------------------


def test_meta_redelivering_a_message_is_answered_once(wa):
    body = wa.payload(wa.text("wamid.dup", "villa in Sheikh Zayed"))
    assert [wa.post(body).status_code for _ in range(3)] == [200, 200, 200]
    assert len(wa.sent) == 1
    assert claims() == {("whatsapp", "wamid.dup"): "done"}


def test_the_same_message_twice_in_one_delivery_is_answered_once(wa):
    message = wa.text("wamid.twin", "villa in Sheikh Zayed")
    wa.post(wa.payload(message, message))
    assert len(wa.sent) == 1


def test_a_batch_of_different_messages_is_each_answered(wa):
    wa.post(wa.payload(wa.text("wamid.1", "villa in Sheikh Zayed"), wa.text("wamid.2", "apartment in New Cairo", sender="201000000002")))
    assert sorted(recipient for recipient, _ in wa.sent) == ["201000000001", "201000000002"]


def test_a_failed_message_is_retried_on_redelivery_and_the_customer_is_told_meanwhile(wa, monkeypatch):
    real = webhooks.run_whatsapp_turn
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("provider timed out")
        return real(*args, **kwargs)

    monkeypatch.setattr(webhooks, "run_whatsapp_turn", flaky)
    body = wa.payload(wa.text("wamid.retry", "villa in Sheikh Zayed"))

    wa.post(body)
    assert claims()[("whatsapp", "wamid.retry")] == "failed"
    assert len(wa.sent) == 1 and "try again" in wa.sent[0][1].lower()  # the customer is not left in silence

    wa.post(body)  # Meta redelivers
    assert claims()[("whatsapp", "wamid.retry")] == "done"
    assert len(wa.sent) == 2 and "501189949" in wa.sent[1][1]  # now they get the real answer


def test_a_message_abandoned_by_a_crashed_worker_can_be_picked_up_again(wa):
    from datetime import datetime, timedelta, timezone

    from app.database import replay

    assert replay.claim("whatsapp", "wamid.crash", ttl_seconds=600)  # a worker claimed it, then died
    body = wa.payload(wa.text("wamid.crash", "villa in Sheikh Zayed"))
    wa.post(body)
    assert wa.sent == []  # too soon: it may still be in progress elsewhere
    with session_scope() as session:
        session.query(db.ReplayGuard).update({"created_at": datetime.now(timezone.utc) - timedelta(minutes=10)})
    wa.post(body)
    assert len(wa.sent) == 1


# ---------------------------------------------------------------------------
# Routing, floods and unusual content
# ---------------------------------------------------------------------------


def test_events_for_another_phone_number_are_ignored(wa):
    wa.post(wa.payload(wa.text("wamid.other", "hi"), phone_number_id="999"))
    assert wa.sent == [] and claims() == {}


def test_delivery_receipts_and_reactions_are_ignored(wa):
    receipt = json.dumps({"entry": [{"changes": [{"value": {"metadata": {"phone_number_id": PHONE_ID},
                                                            "statuses": [{"id": "wamid.x", "status": "delivered"}]}}]}]}).encode()
    assert wa.post(receipt).status_code == 200
    reaction = {"id": "wamid.r", "from": "201000000001", "type": "reaction", "reaction": {"emoji": "👍"}}
    assert wa.post(wa.payload(reaction)).status_code == 200
    assert wa.sent == []


def test_unreadable_media_gets_a_polite_bilingual_notice(wa):
    image = {"id": "wamid.img", "from": "201000000001", "type": "image", "image": {"id": "media-1"}}
    wa.post(wa.payload(image))
    (recipient, text), = wa.sent
    assert recipient == "201000000001" and "voice" in text and "رسائل" in text


def test_a_sender_flooding_messages_is_throttled_without_an_error(wa, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_WHATSAPP_SENDER_PER_MINUTE="3", RATE_LIMIT_WEBHOOK_PER_MINUTE="1000")
    messages = [wa.text(f"wamid.f{i}", "villa in Sheikh Zayed") for i in range(8)]
    assert wa.post(wa.payload(*messages)).status_code == 200  # Meta must always get a 200 or it retries
    assert len(wa.sent) == 3
    assert wa.post(wa.payload(wa.text("wamid.other", "hello", sender="201000000009"))).status_code == 200
    assert len(wa.sent) == 4  # another customer is unaffected


def test_voice_notes_are_transcribed_and_answered_as_text(wa, monkeypatch):
    async def fake_download(media_id):
        return b"ogg-bytes"

    monkeypatch.setattr(webhooks, "download_media", fake_download)
    monkeypatch.setattr(webhooks, "transcribe_audio", lambda audio: "villa in Sheikh Zayed")
    audio = {"id": "wamid.voice", "from": "201000000001", "type": "audio", "audio": {"id": "media-9", "mime_type": "audio/ogg"}}
    wa.post(wa.payload(audio))
    assert len(wa.sent) == 1 and "501189949" in wa.sent[0][1]
    with session_scope() as session:
        assert session.query(db.Interaction).filter_by(channel="whatsapp:voice_note").count() == 1


def test_a_voice_note_that_cannot_be_transcribed_fails_gracefully(wa, monkeypatch):
    async def fake_download(media_id):
        return b"ogg"

    def broken(audio):
        raise RuntimeError("GROQ_API_KEY is not configured")

    monkeypatch.setattr(webhooks, "download_media", fake_download)
    monkeypatch.setattr(webhooks, "transcribe_audio", broken)
    audio = {"id": "wamid.v2", "from": "201000000001", "type": "audio", "audio": {"id": "m"}}
    assert wa.post(wa.payload(audio)).status_code == 200
    assert len(wa.sent) == 1 and "try again" in wa.sent[0][1].lower()
    assert claims()[("whatsapp", "wamid.v2")] == "failed"


def test_the_conversation_is_keyed_by_the_senders_number_and_remembered(wa):
    wa.post(wa.payload(wa.text("wamid.m1", "villa in Sheikh Zayed")))
    wa.post(wa.payload(wa.text("wamid.m2", "tell me about the second one")))
    assert wa.sent[1][1].startswith("Details for 501189810")  # "the second one" resolved from the first turn's results
    with session_scope() as session:
        customer = session.query(db.Customer).filter_by(external_id="whatsapp:201000000001").one()
        assert customer.phone == "+201000000001" and customer.phone_verified is True


def test_a_customer_message_is_stored_in_the_crm(wa):
    wa.post(wa.payload(wa.text("wamid.crm", "villa in Sheikh Zayed, my budget is 9 million, for investment")))
    with session_scope() as session:
        lead = session.query(db.Lead).one()
        assert lead.source == "whatsapp" and lead.budget_max == 9_000_000 and lead.intent == "investment" and lead.score > 0
        assert session.query(db.Interaction).count() == 1


# ---------------------------------------------------------------------------
# open-wa gateway
# ---------------------------------------------------------------------------


def test_gateway_requires_its_secret_and_dedupes(wa, settings_env):
    settings_env(WHATSAPP_GATEWAY_SECRET="gw-secret")
    body = {"message_id": "true_201000000002@c.us_A", "sender": "201000000002@c.us", "text": "hello"}
    assert wa.client.post("/internal/whatsapp/message", json=body).status_code == 401
    assert wa.client.post("/internal/whatsapp/message", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    good = {"Authorization": "Bearer gw-secret"}
    assert wa.client.post("/internal/whatsapp/message", json=body, headers=good).json()["reply"]
    assert wa.client.post("/internal/whatsapp/message", json=body, headers=good).json() == {"reply": None, "duplicate": True}


def test_gateway_is_disabled_without_a_secret(wa, settings_env):
    settings_env(WHATSAPP_GATEWAY_SECRET="")
    body = {"message_id": "m", "sender": "201000000002@c.us", "text": "hello"}
    assert wa.client.post("/internal/whatsapp/message", json=body, headers={"Authorization": "Bearer "}).status_code == 503


def test_gateway_failure_leaves_the_message_retryable(wa, settings_env, monkeypatch):
    settings_env(WHATSAPP_GATEWAY_SECRET="gw-secret")
    real = webhooks.run_whatsapp_turn
    state = {"fail": True}

    def flaky(*args, **kwargs):
        if state["fail"]:
            raise TimeoutError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(webhooks, "run_whatsapp_turn", flaky)
    body = {"message_id": "gw-1", "sender": "201000000002@c.us", "text": "hello"}
    headers = {"Authorization": "Bearer gw-secret"}
    failed = wa.client.post("/internal/whatsapp/message", json=body, headers=headers)
    assert failed.status_code == 500 and failed.json() == {"detail": "Internal server error"}
    assert claims()[("whatsapp_gateway", "gw-1")] == "failed"
    state["fail"] = False  # the gateway retries; the message is not stuck as a "duplicate"
    assert wa.client.post("/internal/whatsapp/message", json=body, headers=headers).json()["reply"]
