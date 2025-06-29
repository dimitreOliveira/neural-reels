import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import Field
from typing_extensions import override

from app.utils.genai_utils import get_client, text2event

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
    video_gen_config: types.GenerateVideosConfig = None

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
    output_key: str = Field(
        default="videos_path",
        description="The key in the session state to store the path of the directory where generated video files are saved.",
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
    enhance_prompt: bool = Field(
        default=True,
        description="If should use Veo's prompt enchacing feature.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs: Any):
        """Initializes the VeoAgent.

        Args:
            **kwargs: Keyword arguments to pass to the BaseAgent constructor.
        """
        super().__init__(**kwargs)
        self.client = get_client(http_options={"api_version": API_VERSION})
        self.video_gen_config = types.GenerateVideosConfig(
            number_of_videos=self.number_of_videos,
            person_generation=self.person_generation,
            aspect_ratio=self.aspect_ratio,
            duration_seconds=self.duration_seconds,
            enhance_prompt=self.enhance_prompt,
        )

    async def _generate_video(
        self,
        # video_name: str,
        scene_idx: int,
        prompt: str,
        output_dir: Path,
    ) -> AsyncGenerator[Event, None]:
        """Generates and saves a video for a single scene.

        This method calls the Veo API to generate a video based on the
        provided prompt. It polls the operation status until the video is
        ready, then downloads and saves it to the specified output directory.

        Args:
            scene_idx: The index of the scene.
            prompt: The text prompt for video generation.
            output_dir: The directory where the generated video will be saved.

        Yields:
            Events indicating the progress of video generation.
        """
        logger.info(
            f"[{self.name}] Generating video for scene {scene_idx + 1} with prompt: '{prompt[:70]}...'"
        )
        yield text2event(self.name, f"Generating video for scene {scene_idx + 1}...")

        operation = self.client.models.generate_videos(
            model=self.model, prompt=prompt, config=self.video_gen_config
        )
        logger.info(
            f"[{self.name}] Video generation started for scene '{scene_idx + 1}'."
        )

        while not operation.done:
            status_msg = f"Video for scene '{scene_idx + 1}' generation in progress. Will check again in {self.polling_interval_seconds}s."
            logger.info(f"[{self.name}] {status_msg}")
            yield text2event(self.name, status_msg)
            await asyncio.sleep(self.polling_interval_seconds)
            operation = self.client.operations.get(operation)

        result = operation.result
        if not result or not result.generated_videos:
            error_msg = f"Video generation failed for scene {scene_idx + 1}. The API returned no videos."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Generated {len(result.generated_videos)} video(s) for scene {scene_idx + 1}'."
        )
        for video_idx, generated_video_entry in enumerate(result.generated_videos):
            logger.info(
                f"[{self.name}] Video has been generated: {generated_video_entry.video.uri}"
            )
            video_filename = f"video_{video_idx + 1}.mp4"
            output_filepath = output_dir / video_filename
            try:
                self.client.files.download(file=generated_video_entry.video)
                generated_video_entry.video.save(str(output_filepath))
                logger.info(
                    f"[{self.name}] Video for scene {scene_idx + 1}', variant {video_idx + 1} saved to '{output_filepath}'"
                )
            except Exception as e_save:
                error_msg = f"Error saving video {output_filepath}: {e_save}"
                logger.error(f"[{self.name}] {error_msg}", exc_info=True)
                yield text2event(self.name, error_msg)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Generates videos for all prompts in the session state.

        This is the main entry point for the agent. It retrieves video prompts
        from the session state, creates the necessary output directories, and
        iterates through the prompts, calling `_generate_video` for each one.

        Args:
            ctx: The invocation context, containing session state with prompts
                 and asset paths.

        Yields:
            Events indicating the overall progress of video generation.
        """
        logger.info(f"[{self.name}] Starting video generation.")

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
            f"[{self.name}] Received {len(prompts)} prompt(s) for video generation."
        )
        logger.info(f"[{self.name}] Calling Veo API with model '{self.model}'...")

        try:
            ctx.session.state[self.output_key] = []
            for scene_idx, prompt in enumerate(prompts):
                output_dir = assets_path / f"scene_{scene_idx + 1}" / self.output_subdir
                output_dir.mkdir(parents=True, exist_ok=True)
                ctx.session.state[self.output_key].append(str(output_dir))

                async for event in self._generate_video(scene_idx, prompt, output_dir):
                    yield event

            final_message = f"Video generated and saved to '{output_dir}'."
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An unexpected error occurred during video generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)


video_generator_agent = VeoAgent(
    name="VideoGeneratorAgent",
    description="Generates videos from a text prompts.",
    input_key="video_prompts",
    output_key="videos_path",
    output_subdir="videos",
    model=MODEL_ID,
    aspect_ratio=ASPECT_RATIO,
)
