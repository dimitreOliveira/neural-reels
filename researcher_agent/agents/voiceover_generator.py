import logging
from pathlib import Path
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
        default="scene_texts",
        description="The key in the session state holding the list of text prompts to convert to speech, one for each scene.",
    )
    output_key: str = Field(
        default="voiceovers_path",
        description="The key in the session state to store the path of the directory where generated audio files are saved.",
    )
    output_subdir: str = Field(
        default="voiceovers",
        description="The subdirectory within the assets path to save the generated WAV audio files.",
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

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = get_client()
        self.generate_content_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    )
                )
            ),
        )

    async def _generate_audio(
        self, scene_idx: int, prompt: str, output_dir: Path
    ) -> AsyncGenerator[Event, None]:
        """
        Generates audio for a single prompt (scene) and saves it.
        """
        logger.info(
            f"[{self.name}] Generating audio for scene {scene_idx + 1} with prompt: '{prompt[:70]}...'"
        )
        yield text2event(self.name, f"Generating audio for scene {scene_idx + 1}...")
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.generate_content_config,
        )

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        output_filepath = str(output_dir / "voiceover.wav")
        save_wave_file(output_filepath, audio_data)

        logger.info(
            f"[{self.name}] Audio for scene {scene_idx + 1} generated and saved to '{output_filepath}'."
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        The core implementation of the agent's logic.
        """
        logger.info(f"[{self.name}] Starting voiceover generation for multiple scenes.")

        # Setup
        assets_path = Path(ctx.session.state.get("assets_path"))
        prompts = ctx.session.state.get(self.input_key).get(self.input_key)

        if not prompts:
            error_msg = (
                f"Input key '{self.input_key}' not found in session state. Aborting."
            )
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Received {len(prompts)} prompt(s) for voiceover generation."
        )
        logger.info(f"[{self.name}] Calling Gemini API with model '{self.model}'...")

        try:
            ctx.session.state[self.output_key] = []
            for scene_idx, prompt in enumerate(prompts):
                output_dir = assets_path / f"scene_{scene_idx + 1}" / self.output_subdir
                output_dir.mkdir(parents=True, exist_ok=True)
                ctx.session.state[self.output_key].append(str(output_dir))

                async for event in self._generate_audio(scene_idx, prompt, output_dir):
                    yield event

            final_message = f"Voiceovers generated and saved to '{output_dir}'."
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An error occurred during TTS generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)


voiceover_generator_agent = VoiceoverGeneratorAgent(
    name="VoiceoverGeneratorAgent",
    description="Generates voiceover audio files from a list of text scripts, one for each scene.",
    model=MODEL_ID,
    voice_name=VOICE_NAME,
    input_key="scenes",
    output_key="voiceovers_path",
    output_subdir="voiceovers",
)
