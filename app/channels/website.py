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
    {"type": "session_end", "reason": "max_duration"|"idle"|"reconnect"}
    {"type": "error", "code": "...", "message": "..."}

Gemini Live only converts speech; business answers come from the shared
LangGraph workflow via the ``ask_aqar_agent`` tool, using the same thread as
the customer's typed chat.
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
from ..api.sessions import consume_voice_token, origin_allowed, website_thread
from ..helpers.config import get_settings
from ..observability.metrics import record_request
from ..voice.gemini_live import AGENT_TOOL_NAME, build_voice_instruction, connect_live_session


logger = logging.getLogger(__name__)

INPUT_AUDIO_MIME_TYPE = "audio/pcm;rate=16000"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
AGENT_TIMEOUT_SECONDS = 45

_active_sessions = 0


class ProviderDisconnected(RuntimeError):
    """The Gemini Live connection closed underneath an active call."""


async def handle_website_voice(websocket: WebSocket) -> None:
    """Authenticate the browser, then bridge it to one Gemini Live session."""
    global _active_sessions
    settings = get_settings()

    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    claims = consume_voice_token(websocket.query_params.get("token"))
    if claims is None:
        await websocket.close(code=1008, reason="Invalid or expired voice token")
        return

    await websocket.accept()
    if _active_sessions >= settings.VOICE_MAX_CONCURRENT_SESSIONS:
        await websocket.send_json({"type": "error", "code": "busy", "message": "Voice is at capacity, please try again shortly."})
        await websocket.close(code=1013, reason="Voice capacity reached")
        return

    _active_sessions += 1
    bridge = VoiceBridge(websocket, claims)
    started = time.monotonic()
    outcome = "ok"
    try:
        await bridge.run()
    except WebSocketDisconnect:
        logger.info("Website voice client disconnected | session=%s", bridge.log_id)
    except ProviderDisconnected:
        outcome = "provider_disconnected"
        await bridge.fail("provider_disconnected", "The voice service disconnected.")
    except Exception:
        outcome = "error"
        logger.exception("Website voice session failed | session=%s", bridge.log_id)
        await bridge.fail("voice_unavailable", "Voice session unavailable.")
    finally:
        _active_sessions -= 1
        duration = time.monotonic() - started
        record_request("website_voice", outcome, duration)
        logger.info(
            "Website voice session closed | session=%s outcome=%s seconds=%.1f tool_calls=%d",
            bridge.log_id, outcome, duration, bridge.tool_calls,
        )


class VoiceBridge:

    def __init__(self, websocket: WebSocket, claims: dict[str, Any]) -> None:
        self.websocket = websocket
        self.thread_id = website_thread(claims["sid"])
        self.language = claims.get("lang", "en")
        self.log_id = claims["sid"][:8]
        self.tool_calls = 0
        self._send_lock = asyncio.Lock()
        self._last_activity = time.monotonic()
        self._closed = False

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

    async def run(self) -> None:
        context = await run_in_threadpool(conversation_service.context_summary, self.thread_id)
        instruction = build_voice_instruction(self.language, context)
        async with connect_live_session(instruction) as session:
            await self.send_json({
                "type": "ready",
                "input_sample_rate": INPUT_SAMPLE_RATE,
                "output_sample_rate": OUTPUT_SAMPLE_RATE,
            })
            tasks = {
                asyncio.create_task(self._browser_to_gemini(session)),
                asyncio.create_task(self._gemini_to_browser(session)),
                asyncio.create_task(self._watchdog()),
            }
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError, Exception):
                        await task
            for task in done:
                task.result()
        with suppress(Exception):
            await self.websocket.close(code=1000)

    # ------------------------------------------------------------------
    # Browser -> Gemini
    # ------------------------------------------------------------------

    async def _browser_to_gemini(self, session: Any) -> None:
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
                await session.send_realtime_input(
                    audio=types.Blob(data=audio, mime_type=INPUT_AUDIO_MIME_TYPE)
                )
                continue

            text = message.get("text")
            if text:
                await self._handle_control_message(text, session)

    async def _handle_control_message(self, message: str, session: Any) -> None:
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
            await session.send_realtime_input(audio_stream_end=True)
        elif kind == "ping":
            await self.send_json({"type": "pong"})

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
            if not received_any:
                raise ProviderDisconnected("Gemini Live closed the connection")

    async def _handle_server_message(self, session: Any, response: Any) -> None:
        tool_call = getattr(response, "tool_call", None)
        if tool_call is not None:
            await self._handle_tool_call(session, tool_call)

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
            # Provider is about to end the stream; the client reconnects and
            # the conversation continues from the shared LangGraph state.
            await self.send_json({"type": "session_end", "reason": "reconnect"})

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        responses = []
        for call in getattr(tool_call, "function_calls", None) or []:
            if call.name != AGENT_TOOL_NAME:
                responses.append(types.FunctionResponse(id=call.id, name=call.name, response={"error": "unknown_tool"}))
                continue
            self.tool_calls += 1
            self._last_activity = time.monotonic()
            message = str((call.args or {}).get("message", "")).strip()
            responses.append(
                types.FunctionResponse(id=call.id, name=call.name, response=await self._run_agent(message))
            )
        if responses:
            await session.send_tool_response(function_responses=responses)

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
                timeout=AGENT_TIMEOUT_SECONDS,
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
