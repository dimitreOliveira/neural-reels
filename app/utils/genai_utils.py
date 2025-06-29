import os
from typing import Any, Dict, Optional

from google import genai
from google.adk.events import Event
from google.genai import types


def get_client(
    api_key: Optional[str] = None, http_options: Optional[Dict[str, Any]] = None
) -> genai.Client:
    """Initializes and returns a Gemini client.

    It retrieves the API key from the provided argument or the
    `GOOGLE_API_KEY` environment variable.

    Args:
        api_key: The Google API key. If not provided, it's read from the
            environment.
        http_options: Optional dictionary of HTTP options for the client.

    Returns:
        An initialized `genai.Client` instance.

    Raises:
        ValueError: If the GOOGLE_API_KEY is not set and no key is provided.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key, http_options=http_options)


def text2event(author: str, text_message: str) -> Event:
    """Creates an ADK Event with a simple text message.

    Args:
        author: The author of the event (e.g., the agent's name).
        text_message: The text content of the event.

    Returns:
        An ADK `Event` object.
    """
    return Event(
        author=author,
        content=types.Content(parts=[types.Part(text=text_message)]),
    )
