"""Website voice-channel service backed by a browser WebSocket."""

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from ..voice.gemini_live import connect_live_session


logger = logging.getLogger(__name__)

INPUT_AUDIO_MIME_TYPE = "audio/pcm;rate=16000"


async def handle_website_voice(websocket: WebSocket) -> None:
    """Bridge browser PCM audio to one Gemini Live session.

    Clients send binary, mono, 16-bit little-endian PCM chunks sampled at
    16 kHz. Gemini audio responses are sent back as binary PCM chunks at
    24 kHz. A client may send ``{"event": "end_of_turn"}`` to finalize
    speech immediately.
    """
    await websocket.accept()

    try:
        async with connect_live_session() as session:
            input_task = asyncio.create_task(
                _forward_browser_audio(websocket, session)
            )
            output_task = asyncio.create_task(
                _forward_gemini_audio(websocket, session)
            )

            done, pending = await asyncio.wait(
                {input_task, output_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()
    except WebSocketDisconnect:
        logger.info("Website voice client disconnected")
    except Exception:
        logger.exception("Website voice session failed")
        with suppress(RuntimeError):
            await websocket.close(code=1011, reason="Voice session unavailable")


async def _forward_browser_audio(websocket: WebSocket, session: object) -> None:
    """Read browser PCM/control frames and forward them to Gemini."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return

        audio = message.get("bytes")
        if audio:
            await session.send_realtime_input(
                audio=types.Blob(data=audio, mime_type=INPUT_AUDIO_MIME_TYPE)
            )
            continue

        text = message.get("text")
        if text:
            await _handle_control_message(text, session)


async def _handle_control_message(message: str, session: object) -> None:
    """Handle non-audio browser frames without accepting arbitrary commands."""
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return

    if payload.get("event") == "end_of_turn":
        await session.send_realtime_input(audio_stream_end=True)


async def _forward_gemini_audio(websocket: WebSocket, session: object) -> None:
    """Send Gemini's raw PCM response chunks back to the browser."""
    async for response in session.receive():
        content = getattr(response, "server_content", None)
        model_turn = getattr(content, "model_turn", None)
        for part in getattr(model_turn, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            audio = getattr(inline_data, "data", None)
            if audio:
                await websocket.send_bytes(audio)
