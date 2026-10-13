"""Persistence and concurrency against a real PostgreSQL server.

Skipped unless ``TEST_DATABASE_URL`` is set, e.g.::

    docker run -d --name aqar-test-pg -e POSTGRES_USER=aqar -e POSTGRES_PASSWORD=aqar_test_pw \\
        -e POSTGRES_DB=aqar_test -p 127.0.0.1:55432:5432 postgres:16-alpine
    TEST_DATABASE_URL=postgresql://aqar:aqar_test_pw@127.0.0.1:55432/aqar_test pytest -m postgres

These tests wipe the ``public`` schema of that database, so they refuse to run
unless its name contains "test".
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.postgres

TEST_URL = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def pg(monkeypatch, settings_env):
    """Point the app at a freshly migrated PostgreSQL database."""
    if not TEST_URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    if "test" not in (make_url(TEST_URL).database or ""):
        pytest.skip("refusing to wipe a database whose name does not contain 'test'")

    from alembic import command
    from alembic.config import Config

    from app.agents.checkpointer import close_checkpointer
    from app.agents.service import reset_conversation_service
    from app.database.session import reset_engine

    def wipe_and_migrate() -> None:
        settings_env(DATABASE_URL=TEST_URL)
        close_checkpointer()
        reset_conversation_service()
        reset_engine()
        engine = create_engine(make_url(TEST_URL).set(drivername="postgresql+psycopg"), isolation_level="AUTOCOMMIT")
        with engine.connect() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")

    wipe_and_migrate()
    yield SimpleRestart(settings_env, close_checkpointer, reset_conversation_service, reset_engine)
    close_checkpointer()
    reset_conversation_service()
    reset_engine()


class SimpleRestart:
    """Simulates a process restart: drops every pool, cache and in-memory singleton."""

    def __init__(self, settings_env, close_checkpointer, reset_conversation_service, reset_engine):
        self._steps = (close_checkpointer, reset_conversation_service, reset_engine)
        self._settings_env = settings_env

    def __call__(self) -> None:
        for step in self._steps:
            step()
        self._settings_env()  # re-read settings, as a new process would


def phone(n: int) -> str:
    return f"+2010123456{n:02d}"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_migrations_apply_roll_back_and_match_the_models(pg):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    from app.database.migrations import schema_status
    from app.database.models import Base
    from app.database.session import get_engine

    assert schema_status()[0] is True

    external = {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
    with get_engine().connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "include_object": lambda o, n, t, r, c: not (t == "table" and n in external)},
        )
        assert compare_metadata(context, Base.metadata) == []  # the models and the migrations agree

    config = Config(str(ROOT / "alembic.ini"))
    command.downgrade(config, "base")
    assert schema_status()[0] is False  # code is ahead of the database: detected, not silently run
    command.upgrade(config, "head")
    assert schema_status()[0] is True


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------


def test_conversation_and_crm_survive_a_restart(pg, stub_rag):
    from app.agents.checkpointer import setup_checkpointer
    from app.agents.service import ConversationService
    from app.agents.tools.crm import crm_service

    setup_checkpointer()
    thread = f"web:{uuid4().hex}"
    first = ConversationService().run_turn(
        thread, "villa in Sheikh Zayed, my budget is 9 million, for investment", channel="website", customer_id=thread
    )
    assert first.requirements["district"] == "Sheikh Zayed"

    pg()  # --- restart: new pools, no in-memory state ---
    setup_checkpointer()

    revived = ConversationService()
    state = revived.get_state(thread)
    assert state["requirements"]["district"] == "Sheikh Zayed"
    assert state["qualification"]["purpose"] == "investment"
    assert len(state["messages"]) == 2  # the customer's message and the reply

    follow_up = revived.run_turn(thread, "with 4 bedrooms or more", channel="website", customer_id=thread)
    assert follow_up.requirements["district"] == "Sheikh Zayed"  # remembered across the restart
    assert follow_up.requirements["min_bedrooms"] == 4

    # CRM rows were written to PostgreSQL, not process memory.
    assert crm_service.get_customer(thread)["id"] == thread
    assert crm_service.get_lead(thread)["status"] in {"new", "qualifying"}
    from app.database import models as db
    from app.database.session import session_scope

    with session_scope() as session:
        assert session.query(db.Interaction).count() == 2


def test_two_workers_never_run_turns_of_one_conversation_at_once(pg, stub_rag):
    """Both 'workers' share one conversation. Without the cross-worker lock the
    second turn would read stale state and one customer message would be lost."""
    from app.agents.checkpointer import setup_checkpointer
    from app.agents.service import ConversationService

    setup_checkpointer()
    original = stub_rag.answer_query

    def slow_answer(*args, **kwargs):
        time.sleep(0.4)  # widen the race window
        return original(*args, **kwargs)

    import app.workflows.rag as rag

    rag.answer_query = slow_answer
    thread = f"web:{uuid4().hex}"
    workers = [ConversationService(), ConversationService()]  # separate in-process locks
    errors: list[Exception] = []

    def run(worker, text):
        try:
            worker.run_turn(thread, text, channel="website", customer_id=thread)
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(workers[0], "villa in Sheikh Zayed")),
        threading.Thread(target=run, args=(workers[1], "apartment in New Cairo")),
    ]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    elapsed = time.monotonic() - started

    assert errors == []
    messages = workers[0].get_state(thread)["messages"]
    assert [m.type for m in messages].count("human") == 2  # nothing lost
    assert len(messages) == 4
    assert elapsed >= 0.75  # the two slow turns ran one after the other


def test_checkpoint_storage_stays_bounded(pg, stub_rag):
    from app.agents.checkpointer import setup_checkpointer
    from app.agents.service import ConversationService
    from app.database.session import get_engine

    setup_checkpointer()
    service = ConversationService()
    thread = f"web:{uuid4().hex}"
    for message in ("villa in Sheikh Zayed", "make it 4 bedrooms", "under 12 million", "actually New Cairo", "the first one", "payment plan?"):
        service.run_turn(thread, message, channel="website", customer_id=thread)

    with get_engine().connect() as connection:
        counts = {
            table: connection.execute(text(f"SELECT count(*) FROM {table} WHERE thread_id = :t"), {"t": thread}).scalar()
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")
        }
    assert counts["checkpoints"] == 1  # only the latest state; subgraph scratch checkpoints are gone
    assert counts["checkpoint_blobs"] <= 40
    assert service.get_state(thread)["requirements"]["district"] == "New Cairo"  # pruning lost nothing
    assert service.run_turn(thread, "thanks", channel="website", customer_id=thread).response


def test_stale_conversations_can_be_purged(pg, stub_rag):
    from app.agents.checkpointer import purge_stale_threads, setup_checkpointer
    from app.agents.service import ConversationService

    setup_checkpointer()
    service = ConversationService()
    thread = f"web:{uuid4().hex}"
    service.run_turn(thread, "hello", channel="website", customer_id=thread)
    assert purge_stale_threads(older_than_days=1) == 0  # active conversations are kept
    assert purge_stale_threads(older_than_days=-1) == 1  # everything older than "tomorrow"
    assert service.get_state(thread) == {}


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------


def _verified_customers(count: int):
    from app.crm.customers import CreateCustomer, CustomerService
    from app.database import models as db
    from app.database.session import session_scope

    customers = []
    for n in range(count):
        customer = CustomerService().create(CreateCustomer(full_name=f"Customer {n}", phone_number=phone(n)))
        with session_scope() as session:
            session.query(db.Customer).filter_by(public_id=customer.id).update({"phone_verified": True})
        customers.append(customer)
    return customers


def test_concurrent_requests_cannot_double_book_a_slot(pg):
    from app.scheduling.availability import TimeSlot
    from app.scheduling.booking import BookingConflictError, BookingService, CreateBooking

    customers = _verified_customers(8)
    start = (datetime.now(timezone.utc) + timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
    service = BookingService()
    outcomes: list[str] = []
    barrier = threading.Barrier(len(customers))

    def attempt(index: int) -> None:
        barrier.wait()
        try:
            service.create(
                CreateBooking(
                    property_id="501189949",
                    customer_id=customers[index].id,
                    slot=TimeSlot(starts_at=start, ends_at=start + timedelta(hours=1)),
                    contact_phone=phone(index),
                )
            )
            outcomes.append("booked")
        except BookingConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(len(customers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert sorted(outcomes) == ["booked"] + ["conflict"] * (len(customers) - 1)


def test_database_itself_rejects_overlapping_bookings(pg):
    """Even code that skips the service's checks cannot create an overlap."""
    from sqlalchemy.exc import IntegrityError

    from app.database import models as db
    from app.database.session import session_scope

    customer = _verified_customers(1)[0]
    start = datetime.now(timezone.utc) + timedelta(days=3)

    def insert(offset_minutes: int, status: str = "pending_confirmation") -> None:
        with session_scope() as session:
            pk = session.query(db.Customer.id).filter_by(public_id=customer.id).scalar()
            session.add(db.Booking(
                property_id="P1", customer_id=pk, starts_at=start + timedelta(minutes=offset_minutes),
                ends_at=start + timedelta(minutes=offset_minutes + 60), status=status, contact_phone=phone(0),
            ))

    insert(0)
    with pytest.raises(IntegrityError):
        insert(30)  # overlaps the first
    insert(60)  # back-to-back is fine
    insert(30, status="cancelled")  # cancelled bookings do not block


def test_expired_hold_releases_the_slot_in_the_database(pg):
    from app.scheduling.availability import TimeSlot
    from app.scheduling.booking import BookingService, CreateBooking

    first, second = _verified_customers(2)
    start = (datetime.now(timezone.utc) + timedelta(days=2)).replace(minute=0, second=0, microsecond=0)

    def request(customer, n):
        return CreateBooking(
            property_id="P9", customer_id=customer.id, contact_phone=phone(n),
            slot=TimeSlot(starts_at=start, ends_at=start + timedelta(hours=1)),
        )

    service = BookingService()
    service.create(request(first, 0))
    later = datetime.now(timezone.utc) + timedelta(hours=30)  # past the 24 h default hold
    assert service.create(request(second, 1), now=later) is not None


# ---------------------------------------------------------------------------
# Replay guard (webhook ids, voice tokens)
# ---------------------------------------------------------------------------


def test_replay_claim_is_atomic_across_connections(pg):
    from app.database import replay

    key = f"wamid.{uuid4().hex}"
    results: list[bool] = []
    barrier = threading.Barrier(12)

    def attempt() -> None:
        barrier.wait()
        results.append(replay.claim("whatsapp", key, ttl_seconds=600))

    threads = [threading.Thread(target=attempt) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert results.count(True) == 1  # exactly one worker processes the message


def test_failed_or_abandoned_claims_can_be_retaken(pg):
    from app.database import models as db
    from app.database import replay
    from app.database.session import session_scope

    assert replay.claim("whatsapp", "m1", ttl_seconds=600)
    assert not replay.claim("whatsapp", "m1", ttl_seconds=600, stale_after_seconds=300)  # still being processed
    replay.settle("whatsapp", "m1", ok=False)
    assert replay.claim("whatsapp", "m1", ttl_seconds=600)  # the provider's retry gets through
    replay.settle("whatsapp", "m1", ok=True)
    assert not replay.claim("whatsapp", "m1", ttl_seconds=600)  # done: duplicates stay blocked

    assert replay.claim("whatsapp", "m2", ttl_seconds=600)
    with session_scope() as session:  # its worker died long ago
        session.query(db.ReplayGuard).filter_by(key="m2").update({"created_at": datetime.now(timezone.utc) - timedelta(hours=1)})
    assert replay.claim("whatsapp", "m2", ttl_seconds=600, stale_after_seconds=300)


def test_voice_token_cannot_be_replayed_on_another_worker(pg):
    from app.api.sessions import consume_voice_token, issue_voice_token

    token = issue_voice_token(uuid4().hex, "ar")
    assert consume_voice_token(token) is not None
    pg()  # a different worker/process has no memory of the first use
    assert consume_voice_token(token) is None


def test_whatsapp_dedup_survives_restart(pg, stub_rag, settings_env, monkeypatch):
    """A duplicate delivery after a restart must not answer the customer twice."""
    import hashlib
    import hmac
    import json

    from fastapi.testclient import TestClient

    import app.api.routes.webhooks as webhooks
    from app.main import app

    settings_env(WHATSAPP_APP_SECRET="app-secret")
    sent: list[str] = []

    async def fake_send(recipient, message):
        sent.append(message)

    monkeypatch.setattr(webhooks, "send_text_message", fake_send)
    body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
        {"id": "wamid.restart", "from": "201000000009", "type": "text", "text": {"body": "villa in Sheikh Zayed"}}]}}]}]}).encode()
    headers = {"X-Hub-Signature-256": "sha256=" + hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()}

    from app.agents.checkpointer import setup_checkpointer

    setup_checkpointer()  # what the app lifespan does on startup
    assert TestClient(app).post("/whatsapp/webhook", content=body, headers=headers).status_code == 200
    pg()
    settings_env(WHATSAPP_APP_SECRET="app-secret")
    setup_checkpointer()
    assert TestClient(app).post("/whatsapp/webhook", content=body, headers=headers).status_code == 200
    assert len(sent) == 1
