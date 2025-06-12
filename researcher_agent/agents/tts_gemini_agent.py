import logging
import os
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import Field
from typing_extensions import override

from researcher_agent.audio_utils import save_wave_file
from researcher_agent.genai_utils import get_generate_content_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TtsGeminiAgent(BaseAgent):
    """
    An ADK Custom Agent that generates audio from text using Gemini's TTS
    capabilities and saves it to a file.
    """

    generate_content_config: types.GenerateContentConfig = None

    name: str = Field(default="TtsGeminiAgent", description="The name of the agent.")
    description: str = Field(
        default="Generates a voiceover audio from text.",
        description="The description of the agent.",
    )
    model: str = Field(
        default="gemini-2.5-flash-preview-tts",
        description="The Gemini model to use for the TTS request.",
    )
    voice_name: str = Field(
        default="Algenib",
        description="The name of the prebuilt voice to use for speech generation.",
    )
    input_key: str = Field(
        default="script",
        description="The key in the session state holding the text prompt to convert to speech.",
    )
    output_key: str = Field(
        default="voiceover",
        description="The key in the session state to store the path of the saved audio file.",
    )
    output_filename: str = Field(
        default="generated_audio.wav",
        description="The path and filename to save the generated WAV audio file.",
    )
    client: genai.Client = Field(
        description="Google Generative AI client used to query models."
    )

    # This allows Pydantic to manage the model without extra config
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_content_config = get_generate_content_config(self.voice_name)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        The core implementation of the agent's logic.
        """
        logger.info(f"[{self.name}] Starting TTS generation.")
        PROJECT_NAME = os.environ.get("PROJECT_NAME")
        if not PROJECT_NAME:
            error_msg = "PROJECT_NAME environment variable not set. Aborting TTS generation."
            logger.error(f"[{self.name}] {error_msg}")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=error_msg)]),
            )
            return

        # 1. Get the text prompt from the session state
        prompt_to_speak = ctx.session.state.get(self.input_key)
        if not prompt_to_speak:
            error_msg = (
                f"Input key '{self.input_key}' not found in session state. Aborting."
            )
            logger.error(f"[{self.name}] {error_msg}")
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=error_msg)]),
            )

            return

        logger.info(
            f"[{self.name}] Received text to speak: '{prompt_to_speak[:50]}...'"
        )

        try:
            # 2. Call the Gemini API to generate audio content
            logger.info(
                f"[{self.name}] Calling Gemini API with model '{self.model}'..."
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt_to_speak,
                config=self.generate_content_config,
            )

            # 3. Extract audio data and save it to a WAV file
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            save_wave_file(f"projects/{PROJECT_NAME}/{self.output_filename}", audio_data)

            # 4. Store the output filename in the session state using `self.output_key`
            ctx.session.state[self.output_key] = self.output_filename
            logger.info(
                f"[{self.name}] Stored output path 'projects/{PROJECT_NAME}/{self.output_filename}' in session state key '{self.output_key}'."
            )

            # 5. Yield a response event to signal completion
            final_message = (
                f"Audio generated and saved to 'projects/{PROJECT_NAME}/{self.output_filename}'."
            )
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=final_message)]),
            )

        except Exception as e:
            error_msg = f"An error occurred during TTS generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=error_msg)]),
            )
