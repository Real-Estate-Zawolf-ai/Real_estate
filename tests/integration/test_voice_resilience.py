"""Gemini Live resilience: reconnects, interruptions and tool-call cancellation.

The Live provider is faked (see conftest); the voice bridge, the agent graph and
the database are the real code.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from tests.conftest import (
    live_audio,
    live_event,
    live_go_away,
    live_tool_call,
    live_tool_cancellation,
)
from tests.integration.test_api_channels import chat, client, receive_until, voice_token  # noqa: F401

ORIGIN = {"origin": "http://testserver"}


def connect(client, token):
    return client.websocket_connect(f"/website/voice?token={token}", headers=ORIGIN)


def test_provider_go_away_reconnects_without_dropping_the_browser_socket(client, fake_live, stub_rag, settings_env):
    settings_env(VOICE_RECONNECT_BACKOFF_SECONDS="0")
    session_id = chat(client, "villa in Sheikh Zayed")["session_id"]
    token = voice_token(client, session_id)["token"]
    fake_live.install_sequence(
        [[live_audio(b"A\x00"), live_go_away()]],
        [[live_audio(b"B\x00"), live_event(turn_complete=True)]],
    )

    with connect(client, token) as ws:
        assert json.loads(ws.receive()["text"])["type"] == "ready"
        ws.send_bytes(b"\x00\x00" * 10)
        first = receive_until(ws, "reconnected")
        assert ("audio", b"A\x00") in first
        assert ("event", {"type": "reconnecting"}) in first

        ws.send_bytes(b"\x00\x00" * 10)  # same browser socket, new provider session
        second = receive_until(ws, "turn_complete")
        assert ("audio", b"B\x00") in second

    assert fake_live.connections == 2
    # The second session was briefed again from the stored conversation.
    assert all("Sheikh Zayed" in instruction for instruction in fake_live.instructions)


def test_audio_captured_during_the_reconnect_gap_is_not_lost(client, fake_live, settings_env):
    settings_env(VOICE_RECONNECT_BACKOFF_SECONDS="0")
    token = voice_token(client)["token"]
    fake_live.install_sequence([[live_go_away()]], [[live_event(turn_complete=True)]])

    with connect(client, token) as ws:
        ws.receive()
        for index in range(1, 4):  # frames 2 and 3 arrive while reconnecting
            ws.send_bytes(bytes([index, 0]) * 8)
        receive_until(ws, "turn_complete")

    delivered = [frame for session in fake_live.sessions for frame in session.audio]
    assert delivered[0] == bytes([1, 0]) * 8
    assert bytes([2, 0]) * 8 in delivered or bytes([3, 0]) * 8 in delivered


def test_cancelled_tool_call_is_not_answered(client, fake_live, stub_rag):
    token = voice_token(client)["token"]
    session = fake_live.install([
        [live_tool_call("villa in Sheikh Zayed", call_id="call-9"), live_tool_cancellation("call-9")],
        [live_event(turn_complete=True)],
    ])

    with connect(client, token) as ws:
        ws.receive()
        ws.send_bytes(b"\x00\x00" * 10)
        receive_until(ws, "agent_result")  # the work still ran...
        ws.send_bytes(b"\x00\x00" * 10)
        receive_until(ws, "turn_complete")

    assert session.tool_responses == []  # ...but the model no longer wanted the answer


def test_interruption_is_delivered_while_the_agent_is_still_working(client, fake_live, monkeypatch):
    """The old inline tool handling would have held the interruption back until
    the (up to 30 s) agent turn finished."""
    import app.channels.website as website

    release = threading.Event()

    def slow_turn(thread_id, message, **_):
        release.wait(timeout=10)
        return SimpleNamespace(response="done", intent="x", to_dict=lambda: {"response": "done", "intent": "x"})

    monkeypatch.setattr(website, "conversation_service", SimpleNamespace(run_turn=slow_turn, context_summary=lambda _t: ""))
    token = voice_token(client)["token"]
    fake_live.install([[live_tool_call("anything"), live_event(interrupted=True)], [live_event(turn_complete=True)]])

    with connect(client, token) as ws:
        ws.receive()
        ws.send_bytes(b"\x00\x00" * 10)
        seen = receive_until(ws, "interrupted")  # arrives although the agent has not finished
        assert not any(kind == "event" and payload["type"] == "agent_result" for kind, payload in seen)
        release.set()
        receive_until(ws, "agent_result")


def test_idle_call_ends_with_a_reason(client, fake_live, settings_env):
    settings_env(VOICE_IDLE_TIMEOUT_SECONDS="0")
    token = voice_token(client)["token"]
    fake_live.install([])
    with connect(client, token) as ws:
        ws.receive()
        frames = receive_until(ws, "session_end")
    assert frames[-1] == ("event", {"type": "session_end", "reason": "idle"})


def test_voice_connections_are_rate_limited_and_capped_per_client(client, fake_live, settings_env):
    settings_env(RATE_LIMIT_ENABLED="true", RATE_LIMIT_VOICE_CONNECT_PER_MINUTE="2", VOICE_MAX_SESSIONS_PER_IP="1")
    fake_live.install([])
    tokens = [voice_token(client)["token"] for _ in range(3)]

    with connect(client, tokens[0]) as first:
        assert json.loads(first.receive()["text"])["type"] == "ready"
        # Second concurrent call from the same address: refused as busy.
        with connect(client, tokens[1]) as second:
            assert json.loads(second.receive()["text"])["code"] == "busy"

    # The handshake budget (2 per minute) is spent, so a third attempt is refused outright.
    with pytest.raises(WebSocketDisconnect):
        with connect(client, tokens[2]) as third:
            third.receive()
