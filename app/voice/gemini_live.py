"""Shared Gemini Live session configuration for website and WhatsApp voice."""

from google import genai
from google.genai import types

from helpers.config import get_settings


def create_gemini_client() -> genai.Client:
    """Create an authenticated Gemini client for server-side connections."""
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=settings.GEMINI_API_KEY)


def build_live_config(system_instruction: str | None = None) -> types.LiveConnectConfig:
    """Build the common audio-session configuration for all voice channels."""
    settings = get_settings()

    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=settings.GEMINI_VOICE,
                )
            )
        ),
    )


def connect_live_session(system_instruction: str | None = None):
    """Return an async context manager for a configured Gemini Live session.

    Use this from a channel adapter with ``async with`` so the WebSocket is
    always closed when a website or WhatsApp conversation finishes.
    """
    settings = get_settings()
    client = create_gemini_client()
    return client.aio.live.connect(
        model=settings.GEMINI_LIVE_MODEL,
        config=build_live_config(system_instruction),
    )
