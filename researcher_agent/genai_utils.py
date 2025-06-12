import os

from google import genai
from google.genai import types


def get_client(api_key=None):
    """Initializes and returns a Gemini client."""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


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
