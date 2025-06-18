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

from researcher_agent.utils.audio_utils import save_wave_file
from researcher_agent.utils.genai_utils import (
    get_client,
    get_generate_content_config,
    text2event,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_ID = "gemini-2.5-flash-preview-tts"
VOICE_NAME = "Algenib"


class VoiceoverGeneratorAgent(BaseAgent):
    """
    An ADK Custom Agent that generates audio from text using Gemini's TTS
    capabilities and saves it to a file.
    """

    client: genai.Client = None
    generate_content_config: types.GenerateContentConfig = None

    # --- Pydantic Fields for Agent Configuration ---
    name: str = Field(
        default="VoiceoverGeneratorAgent", description="The name of the agent."
    )
    description: str = Field(
        default="Generates a voiceover audio from text.",
        description="The description of the agent.",
    )
    input_key: str = Field(
        default="script",
        description="The key in the session state holding the text prompt to convert to speech.",
    )
    output_key: str = Field(
        default="voiceover_path",
        description="The key in the session state to store the path of the saved audio file.",
    )
    output_filename: str = Field(
        default="generated_audio.wav",
        description="The path and filename to save the generated WAV audio file.",
    )
    # --- Gemini-specific configuration ---
    model: str = Field(
        default=MODEL_ID,
        description="The Gemini model to use for the TTS request.",
    )
    voice_name: str = Field(
        default="Algenib",
        description="The name of the prebuilt voice to use for speech generation.",
    )

    # This allows Pydantic to manage the model without extra config
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = get_client()
        self.generate_content_config = get_generate_content_config(self.voice_name)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        The core implementation of the agent's logic.
        """
        logger.info(f"[{self.name}] Starting TTS generation.")

        # Setup
        assets_path = ctx.session.state.get("assets_path")
        output_filepath = os.path.join(assets_path, self.output_filename)

        # 1. Get the text prompt from the session state
        script_container = ctx.session.state.get(self.input_key)
        prompt_to_speak = script_container.script if script_container else ""

        if not prompt_to_speak:
            error_msg = (
                f"Input key '{self.input_key}' not found in session state. Aborting."
            )
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
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
            save_wave_file(output_filepath, audio_data)

            # 4. Store the output filename in the session state using `self.output_key`
            ctx.session.state[self.output_key] = self.output_filename
            logger.info(
                f"[{self.name}] Stored output path '{output_filepath}' in session state key '{self.output_key}'."
            )

            # 5. Yield a response event to signal completion
            final_message = f"Audio generated and saved to '{output_filepath}'."
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An error occurred during TTS generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)


voiceover_generator_agent = VoiceoverGeneratorAgent(
    name="VoiceoverGeneratorAgent",
    description="Generates a voiceover audio from text.",
    model=MODEL_ID,
    voice_name=VOICE_NAME,
    input_key="script",
    output_key="voiceover_path",
    output_filename="voiceover.wav",
)
