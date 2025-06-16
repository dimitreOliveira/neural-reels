import os

from google import genai
from google.adk.events import Event
from google.genai import types


def get_client(api_key=None, http_options=None):
    """Initializes and returns a Gemini client."""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key, http_options=http_options)


def get_generate_content_config(voice_name="Algenib"):
    """Creates the content generation config for audio modality."""
    return types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        ),
    )


def text2event(author: str, text_message: str) -> Event:
    """Creates an ADK Event with a simple text message."""
    return Event(
        author=author,
        content=types.Content(parts=[types.Part(text=text_message)]),
    )
