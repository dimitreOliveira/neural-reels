import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import Field
from typing_extensions import override

from researcher_agent.utils.genai_utils import get_client, text2event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "veo-2.0-generate-001"
ASPECT_RATIO = "9:16"
DURATION_SECONDS = 8
POLLING_INTERVAL_SECONDS = 10
API_VERSION = "v1beta"


class VeoAgent(BaseAgent):
    """
    An ADK Custom Agent that generates videos using Veo
    based on prompts and saves them to files.
    """

    client: genai.Client = None

    # --- Pydantic Fields for Agent Configuration ---
    name: str = Field(
        default="VideoGeneratorAgent", description="The name of the agent."
    )
    description: str = Field(
        default="Generates videos from text prompts using Veo.",
        description="The description of the agent.",
    )
    input_key: str = Field(
        default="video_prompts",
        description="The key in the session state holding the dict with the text prompt(s) for video generation.",
    )
    intro_input_key: str = Field(
        default="intro_video_prompt",
        description="The key in the session state holding the text prompt(s) for intro video generation.",
    )
    outro_input_key: str = Field(
        default="outro_video_prompts",
        description="The key in the session state holding the text prompt(s) for outro video generation.",
    )
    output_key: str = Field(
        default="videos_path",
        description="The key in the session state to store the path of the directory where video files are saved.",
    )
    output_subdir: str = Field(
        default="videos",
        description="The subdirectory within the project folder to save the generated video files.",
    )
    # --- Veo-specific configuration ---
    model: str = Field(
        default=MODEL_ID,
        description="The Veo model to use for video generation.",
    )
    number_of_videos: int = Field(
        default=1, description="Number of videos to generate for each prompt (1-4)."
    )
    person_generation: str = Field(
        default="allow_all",
        description="Policy for generating people. Supported: 'dont_allow', 'allow_adult', 'allow_all'.",
    )
    aspect_ratio: str = Field(
        default=ASPECT_RATIO,
        description="Aspect ratio of the generated video (e.g., '16:9', '9:16').",
    )
    duration_seconds: int = Field(
        default=DURATION_SECONDS,
        description="Duration of the generated video in seconds (default is 8s, range 5-8s for Veo 2.0).",
    )
    polling_interval_seconds: int = Field(
        default=POLLING_INTERVAL_SECONDS,
        description="Interval in seconds to poll for video generation status.",
    )

    # This allows Pydantic to manage the model without extra config
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = get_client(http_options={"api_version": API_VERSION})

    async def _generate_video(
        self,
        video_name: str,
        prompt_text: str,
        video_gen_config: types.GenerateVideosConfig,
        output_dir: Path,
    ) -> AsyncGenerator[Event, bool]:
        """
        Generates a video for a single scene/prompt, yields status events,
        and returns True if successful, False otherwise.
        """
        logger.info(
            f"[{self.name}] Generating video for '{video_name}' with prompt: '{prompt_text[:70]}...'"
        )
        yield text2event(self.name, f"Generating video for '{video_name}'...")

        operation = self.client.models.generate_videos(
            model=self.model, prompt=prompt_text, config=video_gen_config
        )
        logger.info(f"[{self.name}] Video generation started for '{video_name}'.")

        while not operation.done:
            status_msg = f"Video for '{video_name}' generation in progess. Will check again in {self.polling_interval_seconds}s."
            logger.info(f"[{self.name}] {status_msg}")
            yield text2event(self.name, status_msg)
            await asyncio.sleep(self.polling_interval_seconds)
            operation = self.client.operations.get(operation)

        result = operation.result
        if not result or not result.generated_videos:
            error_msg = f"Video generation failed or no videos returned for '{video_name}' (prompt: '{prompt_text[:70]}...')."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Generated {len(result.generated_videos)} video(s) for '{video_name}'."
        )
        for video_idx, generated_video_entry in enumerate(result.generated_videos):
            logger.info(
                f"[{self.name}] Video has been generated: {generated_video_entry.video.uri}"
            )
            video_filename = f"{video_name}_video_{video_idx}.mp4"
            output_filepath = output_dir / video_filename
            try:
                self.client.files.download(file=generated_video_entry.video)
                generated_video_entry.video.save(str(output_filepath))
                logger.info(
                    f"[{self.name}] Video for '{video_name}', variant {video_idx + 1} saved to '{output_filepath}'"
                )
            except Exception as e_save:
                error_msg = f"Error saving video {output_filepath}: {e_save}"
                logger.error(f"[{self.name}] {error_msg}", exc_info=True)
                yield text2event(self.name, error_msg)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting video generation.")
        PROJECT_NAME = os.environ.get("PROJECT_NAME")
        if not PROJECT_NAME:
            error_msg = (
                "PROJECT_NAME environment variable not set. Aborting video generation."
            )
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        video_prompts = ctx.session.state.get(self.input_key)
        if not video_prompts:
            error_msg = f"Input key '{self.input_key}' (object/dict container for prompts) not found in session state. Aborting."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        # Get intro video prompt
        intro_video_prompt = video_prompts.get(self.intro_input_key)
        if not intro_video_prompt:
            error_msg = f"Intro input key '{self.intro_input_key}' (object/dict container for prompts) not found in session state. Aborting."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        # Get outro video prompt
        outro_video_prompt = video_prompts.get(self.outro_input_key)
        if not outro_video_prompt:
            error_msg = f"Outro input key '{self.outro_input_key}' (object/dict container for prompts) not found in session state. Aborting."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        output_dir = Path(f"projects/{PROJECT_NAME}") / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            video_gen_config = types.GenerateVideosConfig(
                number_of_videos=self.number_of_videos,
                person_generation=self.person_generation,
                aspect_ratio=self.aspect_ratio,
                duration_seconds=self.duration_seconds,
            )

            # Generate intro video
            async for event in self._generate_video(
                "intro", intro_video_prompt, video_gen_config, output_dir
            ):
                yield event

            # Generate outro video
            async for event in self._generate_video(
                "outro", outro_video_prompt, video_gen_config, output_dir
            ):
                yield event

            ctx.session.state[self.output_key] = str(output_dir)
            logger.info(
                f"[{self.name}] Stored output path '{output_dir}' in session state key '{self.output_key}'."
            )
            final_message = (
                f"Video generation process completed. Videos saved in '{output_dir}'."
            )
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An unexpected error occurred during video generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)


video_generator_agent = VeoAgent(
    name="VideoGeneratorAgent",
    description="Generates videos from a text prompts.",
    input_key="video_prompts",
    intro_input_key="intro_video_prompt",
    outro_input_key="outro_video_prompt",
    output_key="videos_path",
    output_subdir="videos",
    model=MODEL_ID,
    aspect_ratio=ASPECT_RATIO,
)
