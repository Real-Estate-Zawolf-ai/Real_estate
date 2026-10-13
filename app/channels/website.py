"""Website voice-channel service backed by a browser WebSocket.

Protocol (``/website/voice?token=...``; the token comes from ``POST /voice/token``):

Browser -> server
    binary   mono 16-bit little-endian PCM at 16 kHz
    {"type": "audio_end"}   microphone paused/muted; lets Gemini finish the turn
    {"type": "ping"}

Server -> browser
    binary   mono 16-bit little-endian PCM at 24 kHz (assistant speech)
    {"type": "ready", "input_sample_rate": 16000, "output_sample_rate": 24000}
    {"type": "transcript", "role": "user"|"assistant", "text": "..."}   incremental chunks
    {"type": "tool_start"}            the agent workflow is running
    {"type": "agent_result", ...}     same payload as POST /chat (properties, intent, ...)
    {"type": "interrupted"}           customer barged in: drop queued playback
    {"type": "turn_complete"}
    {"type": "reconnecting"}          the voice provider dropped; the server is re-establishing it
    {"type": "reconnected"}           the call continues (same conversation, nothing to do)
    {"type": "session_end", "reason": "max_duration"|"idle"}
    {"type": "error", "code": "...", "message": "..."}

Gemini Live only converts speech; business answers come from the shared
LangGraph workflow via the ``ask_aqar_agent`` tool, using the same thread as
the customer's typed chat.

Resilience: the browser socket outlives any single Gemini Live connection. When
the provider ends a session (its periodic ``go_away``, a network drop) the
bridge opens a new one, re-briefing the model from the conversation state in
PostgreSQL, so the customer hears at most a short pause. Microphone audio
captured during the gap is buffered (bounded, newest wins).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types
from starlette.concurrency import run_in_threadpool

from ..agents.messages import say
from ..agents.service import conversation_service
from ..api.ratelimit import allow_voice_connect, client_ip
from ..api.sessions import consume_voice_token, origin_allowed, website_thread
from ..helpers.config import get_settings
from ..observability.metrics import record_request
from ..voice.gemini_live import AGENT_TOOL_NAME, build_voice_instruction, connect_live_session


logger = logging.getLogger(__name__)

INPUT_AUDIO_MIME_TYPE = "audio/pcm;rate=16000"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
# Frames buffered while the provider connection is being re-established.
INBOX_FRAMES = 100

_active_sessions = 0
_sessions_per_ip: dict[str, int] = {}


class ProviderDisconnected(RuntimeError):
    """The Gemini Live connection closed underneath an active call."""


class ProviderBusy(ProviderDisconnected):
    """Gemini refused a new Live session because the quota / rate limit is used up.

    Retrying immediately only burns more quota, so the call fails at once with a
    "busy, try again in a minute" message instead of "the connection dropped".
    """


_QUOTA_MARKERS = ("resource_exhausted", "resource has been exhausted", "quota", "rate limit", "too many requests", " 429")


def is_provider_quota_error(error: BaseException) -> bool:
    """True for Gemini 429 / RESOURCE_EXHAUSTED style failures (HTTP, SDK or WebSocket close)."""
    if getattr(error, "code", None) == 429 or getattr(error, "status_code", None) == 429:
        return True
    text = f" {type(error).__name__} {error}".lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


async def handle_website_voice(websocket: WebSocket) -> None:
    """Authenticate the browser, then bridge it to Gemini Live sessions."""
    global _active_sessions
    settings = get_settings()

    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    if not allow_voice_connect(websocket):
        await websocket.close(code=1008, reason="Too many connection attempts")
        return
    claims = await run_in_threadpool(consume_voice_token, websocket.query_params.get("token"))
    if claims is None:
        await websocket.close(code=1008, reason="Invalid or expired voice token")
        return

    ip = client_ip(websocket)
    await websocket.accept()
    if _active_sessions >= settings.VOICE_MAX_CONCURRENT_SESSIONS or _sessions_per_ip.get(ip, 0) >= settings.VOICE_MAX_SESSIONS_PER_IP:
        await websocket.send_json({"type": "error", "code": "busy", "message": "Voice is at capacity, please try again shortly."})
        await websocket.close(code=1013, reason="Voice capacity reached")
        return

    _active_sessions += 1
    _sessions_per_ip[ip] = _sessions_per_ip.get(ip, 0) + 1
    bridge = VoiceBridge(websocket, claims)
    started = time.monotonic()
    outcome = "ok"
    try:
        await bridge.run()
    except WebSocketDisconnect:
        logger.info("Website voice client disconnected | session=%s", bridge.log_id)
    except ProviderBusy:
        outcome = "provider_busy"
        await bridge.fail("busy", "Voice is busy right now, please try again in a minute.")
    except ProviderDisconnected:
        outcome = "provider_disconnected"
        await bridge.fail("provider_disconnected", "The voice service disconnected.")
    except Exception:
        outcome = "error"
        logger.exception("Website voice session failed | session=%s", bridge.log_id)
        await bridge.fail("voice_unavailable", "Voice session unavailable.")
    finally:
        _active_sessions -= 1
        remaining = _sessions_per_ip.get(ip, 1) - 1
        if remaining > 0:
            _sessions_per_ip[ip] = remaining
        else:
            _sessions_per_ip.pop(ip, None)
        duration = time.monotonic() - started
        record_request("website_voice", outcome, duration)
        logger.info(
            "Website voice session closed | session=%s outcome=%s seconds=%.1f tool_calls=%d reconnects=%d",
            bridge.log_id, outcome, duration, bridge.tool_calls, bridge.reconnects,
        )


class VoiceBridge:

    def __init__(self, websocket: WebSocket, claims: dict[str, Any]) -> None:
        self.websocket = websocket
        self.thread_id = website_thread(claims["sid"])
        self.language = claims.get("lang", "en")
        self.log_id = claims["sid"][:8]
        self.tool_calls = 0
        self.reconnects = 0
        self._send_lock = asyncio.Lock()
        self._last_activity = time.monotonic()
        self._inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=INBOX_FRAMES)
        self._tool_tasks: set[asyncio.Task] = set()
        self._cancelled_calls: set[str] = set()
        self._go_away = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload, ensure_ascii=False, default=str))

    async def send_audio(self, audio: bytes) -> None:
        async with self._send_lock:
            await self.websocket.send_bytes(audio)

    async def fail(self, code: str, message: str) -> None:
        with suppress(Exception):
            await self.send_json({"type": "error", "code": code, "message": message})
        with suppress(Exception):
            await self.websocket.close(code=1011, reason=message[:120])

    # ------------------------------------------------------------------
    # Call lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        settings = get_settings()
        # These two live for the whole call, across provider reconnects.
        reader = asyncio.create_task(self._read_browser())
        watchdog = asyncio.create_task(self._watchdog())
        persistent = (reader, watchdog)
        attempts = 0
        try:
            while True:
                outcome = await self._run_provider_session(reader, watchdog, first=attempts == 0)
                if outcome != "provider_ended":
                    break
                attempts += 1
                self.reconnects += 1
                if attempts > settings.VOICE_MAX_PROVIDER_RECONNECTS:
                    raise ProviderDisconnected("Gemini Live kept closing the connection")
                logger.info("Reconnecting to Gemini Live | session=%s attempt=%d", self.log_id, attempts)
                await self.send_json({"type": "reconnecting"})
                await asyncio.sleep(settings.VOICE_RECONNECT_BACKOFF_SECONDS * attempts)
        finally:
            for task in (*persistent, *self._tool_tasks):
                task.cancel()
            for task in (*persistent, *self._tool_tasks):
                with suppress(asyncio.CancelledError, Exception):
                    await task
        with suppress(Exception):
            await self.websocket.close(code=1000)

    async def _run_provider_session(self, reader: asyncio.Task, watchdog: asyncio.Task, *, first: bool) -> str:
        """One Gemini Live connection. Returns why it ended:
        ``browser_closed``, ``limit`` (max duration / idle) or ``provider_ended``."""
        self._go_away = False
        self._cancelled_calls.clear()
        # The conversation state is the source of truth; re-brief the model from it
        # every time so a reconnect continues exactly where the call was.
        context = await run_in_threadpool(conversation_service.context_summary, self.thread_id)
        instruction = build_voice_instruction(self.language, context)
        try:
            async with connect_live_session(instruction) as session:
                if first:
                    await self.send_json({
                        "type": "ready",
                        "input_sample_rate": INPUT_SAMPLE_RATE,
                        "output_sample_rate": OUTPUT_SAMPLE_RATE,
                    })
                else:
                    await self.send_json({"type": "reconnected"})
                pumps = {
                    asyncio.create_task(self._pump_to_gemini(session)),
                    asyncio.create_task(self._gemini_to_browser(session)),
                }
                try:
                    done, _ = await asyncio.wait(pumps | {reader, watchdog}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in pumps:
                        task.cancel()
                    for task in pumps:
                        with suppress(asyncio.CancelledError, Exception):
                            await task
                    for task in list(self._tool_tasks):
                        task.cancel()  # their replies belong to a session that is ending
                if reader in done:
                    reader.result()  # raises WebSocketDisconnect-style errors, else a clean end
                    return "browser_closed"
                if watchdog in done:
                    watchdog.result()
                    return "limit"
                for task in done & pumps:
                    error = task.exception()
                    if error is None or isinstance(error, ProviderDisconnected):
                        return "provider_ended"
                    if isinstance(error, WebSocketDisconnect):
                        raise error
                    logger.warning("Voice provider error (%s); reconnecting | session=%s", type(error).__name__, self.log_id)
                    return "provider_ended"
        except (WebSocketDisconnect, asyncio.CancelledError):
            raise
        except ProviderDisconnected:
            return "provider_ended"
        except Exception as error:
            # Includes failing to open the connection at all: retried a few times
            # (transient network errors), then reported as provider_disconnected.
            # The message is logged (credentials are scrubbed by the log filter) so a bad
            # model name or a revoked key is diagnosable from the Render logs.
            logger.warning(
                "Gemini Live connection failed (%s: %s) | session=%s", type(error).__name__, str(error)[:200], self.log_id
            )
            if is_provider_quota_error(error):
                raise ProviderBusy("Gemini Live quota or rate limit reached") from error
            return "provider_ended"
        return "provider_ended"

    # ------------------------------------------------------------------
    # Browser -> Gemini
    # ------------------------------------------------------------------

    def _enqueue(self, item: tuple[str, Any]) -> None:
        """Buffer for the live provider session; drops the oldest when full."""
        try:
            self._inbox.put_nowait(item)
        except asyncio.QueueFull:
            with suppress(asyncio.QueueEmpty):
                self._inbox.get_nowait()
            self._inbox.put_nowait(item)

    async def _read_browser(self) -> None:
        max_frame = get_settings().VOICE_MAX_FRAME_BYTES
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            audio = message.get("bytes")
            if audio:
                if len(audio) > max_frame or len(audio) % 2:
                    logger.warning("Dropped invalid audio frame | session=%s bytes=%d", self.log_id, len(audio))
                    continue
                # An open microphone streams silence too, so raw frames do not
                # count as activity; transcribed speech does (see below).
                self._enqueue(("audio", audio))
                continue

            text = message.get("text")
            if text:
                await self._handle_control_message(text)

    async def _handle_control_message(self, message: str) -> None:
        """Handle non-audio browser frames without accepting arbitrary commands."""
        if len(message) > 2_000:
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        kind = payload.get("type") or payload.get("event")
        if kind in {"audio_end", "end_of_turn"}:
            self._enqueue(("audio_end", None))
        elif kind == "ping":
            await self.send_json({"type": "pong"})

    async def _pump_to_gemini(self, session: Any) -> None:
        while True:
            kind, payload = await self._inbox.get()
            if kind == "audio":
                await session.send_realtime_input(audio=types.Blob(data=payload, mime_type=INPUT_AUDIO_MIME_TYPE))
            elif kind == "audio_end":
                await session.send_realtime_input(audio_stream_end=True)

    # ------------------------------------------------------------------
    # Gemini -> browser
    # ------------------------------------------------------------------

    async def _gemini_to_browser(self, session: Any) -> None:
        # session.receive() ends after every turn_complete, so keep reading
        # turn after turn until the provider connection itself closes.
        while True:
            received_any = False
            async for response in session.receive():
                received_any = True
                await self._handle_server_message(session, response)
            if not received_any or self._go_away:
                raise ProviderDisconnected("Gemini Live closed the connection")

    async def _handle_server_message(self, session: Any, response: Any) -> None:
        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            self._start_tool_calls(session, tool_call)

        cancellation = getattr(response, "tool_call_cancellation", None)
        if cancellation is not None:
            # The customer talked over a pending tool call: the model no longer
            # wants that answer, so do not send it.
            self._cancelled_calls.update(str(item) for item in (getattr(cancellation, "ids", None) or []))

        content = getattr(response, "server_content", None)
        if content is not None:
            if getattr(content, "interrupted", False):
                await self.send_json({"type": "interrupted"})
            for role, field in (("user", "input_transcription"), ("assistant", "output_transcription")):
                transcription = getattr(content, field, None)
                text = getattr(transcription, "text", None)
                if text:
                    self._last_activity = time.monotonic()
                    await self.send_json({"type": "transcript", "role": role, "text": text})
            model_turn = getattr(content, "model_turn", None)
            for part in getattr(model_turn, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                audio = getattr(inline_data, "data", None)
                if audio:
                    self._last_activity = time.monotonic()
                    await self.send_audio(audio)
            if getattr(content, "turn_complete", False):
                await self.send_json({"type": "turn_complete"})

        if getattr(response, "go_away", None) is not None:
            # The provider will end this connection shortly. Finish the current
            # exchange, then reconnect transparently (see run()).
            self._go_away = True

    # ------------------------------------------------------------------
    # Tool calls: run in the background so audio, interruptions and the
    # provider's own messages keep flowing while the agent works.
    # ------------------------------------------------------------------

    def _start_tool_calls(self, session: Any, tool_call: Any) -> None:
        for call in getattr(tool_call, "function_calls", None) or []:
            task = asyncio.create_task(self._answer_tool_call(session, call))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _answer_tool_call(self, session: Any, call: Any) -> None:
        if call.name != AGENT_TOOL_NAME:
            payload: dict[str, Any] = {"error": "unknown_tool"}
        else:
            self.tool_calls += 1
            self._last_activity = time.monotonic()
            message = str((call.args or {}).get("message", "")).strip()
            payload = await self._run_agent(message)
        if str(call.id) in self._cancelled_calls:
            logger.info("Tool call %s was cancelled by an interruption; reply dropped | session=%s", call.id, self.log_id)
            return
        await session.send_tool_response(
            function_responses=[types.FunctionResponse(id=call.id, name=call.name, response=payload)]
        )

    async def _run_agent(self, message: str) -> dict[str, Any]:
        await self.send_json({"type": "tool_start"})
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(
                    conversation_service.run_turn,
                    self.thread_id,
                    message,
                    channel="website",
                    customer_id=self.thread_id,
                    modality="voice",
                ),
                timeout=get_settings().VOICE_AGENT_TIMEOUT_SECONDS,
            )
        except ValueError:
            record_request("voice_agent_turn", "invalid", time.monotonic() - started)
            return {"error": "empty_request", "reply": say(self.language, "fallback")}
        except asyncio.TimeoutError:
            logger.warning("Voice agent turn timed out | session=%s", self.log_id)
            record_request("voice_agent_turn", "timeout", time.monotonic() - started)
            return {"error": "timeout", "reply": say(self.language, "search_unavailable")}
        except Exception:
            logger.exception("Voice agent turn failed | session=%s", self.log_id)
            record_request("voice_agent_turn", "error", time.monotonic() - started)
            await self.send_json({"type": "error", "code": "agent_unavailable", "message": say(self.language, "search_unavailable")})
            return {"error": "agent_unavailable", "reply": say(self.language, "search_unavailable")}

        record_request("voice_agent_turn", "ok", time.monotonic() - started)
        self._last_activity = time.monotonic()
        await self.send_json({"type": "agent_result", "query": message, **result.to_dict()})
        return {"reply": result.response, "intent": result.intent}

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    async def _watchdog(self) -> None:
        settings = get_settings()
        started = time.monotonic()
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            if now - started > settings.VOICE_MAX_SESSION_SECONDS:
                await self.send_json({"type": "session_end", "reason": "max_duration"})
                return
            if now - self._last_activity > settings.VOICE_IDLE_TIMEOUT_SECONDS:
                await self.send_json({"type": "session_end", "reason": "idle"})
                return


__all__ = ["VoiceBridge", "handle_website_voice"]
