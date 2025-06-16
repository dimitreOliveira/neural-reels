import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from pydantic import Field
from typing_extensions import override

from researcher_agent.utils.genai_utils import text2event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAssemblerAgent(BaseAgent):
    """
    An ADK Custom Agent that assembles video clips, images, and audio
    into a final video file using MoviePy.
    """

    name: str = Field(
        default="VideoAssemblerAgent", description="The name of the agent."
    )
    description: str = Field(
        default="Assembles video clips, images, and audio into a final video.",
        description="The description of the agent.",
    )

    # --- Input keys from session state ---
    voiceover_file_input_key: str = Field(
        default="voiceover_path",
        description="Key in session state for the voiceover audio filename (from VoiceoverGeneratorAgent).",
    )
    images_dir_input_key: str = Field(
        default="images_path",
        description="Key in session state for the directory containing image files (from ImageGeneratorAgent).",
    )
    videos_dir_input_key: str = Field(
        default="videos_path",
        description="Key in session state for the directory containing intro/outro videos (from VideoGeneratorAgent).",
    )
    # --- Configuration for specific asset names and output ---
    intro_video_filename: str = Field(
        default="intro_video_0.mp4",
        description="Filename of the intro video within the videos directory.",
    )
    outro_video_filename: str = Field(
        default="outro_video_0.mp4",
        description="Filename of the outro video within the videos directory.",
    )
    output_key: str = Field(
        default="assembled_video_path",
        description="Key in session state to store the path of the assembled video file.",
    )
    output_subdir: str = Field(
        default="assembled_video",
        description="Subdirectory within the project folder to save the assembled video.",
    )
    output_filename: str = Field(
        default="short_video.mp4",
        description="Filename for the assembled video.",
    )
    fps: int = Field(default=24, description="Frames per second for the output video.")

    # This allows Pydantic to manage the model without extra config
    model_config = {"arbitrary_types_allowed": True}

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting video assembly.")
        PROJECT_NAME = os.environ.get("PROJECT_NAME")
        if not PROJECT_NAME:
            error_msg = (
                "PROJECT_NAME environment variable not set. Aborting video assembly."
            )
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        try:
            # 1. Get asset paths from session state
            videos_base_dir_str = ctx.session.state.get(self.videos_dir_input_key)
            images_dir_str = ctx.session.state.get(self.images_dir_input_key)
            voiceover_filename_in_state = ctx.session.state.get(
                self.voiceover_file_input_key
            )

            if not all(
                [videos_base_dir_str, images_dir_str, voiceover_filename_in_state]
            ):
                missing_keys = [
                    key
                    for key, val in {
                        self.videos_dir_input_key: videos_base_dir_str,
                        self.images_dir_input_key: images_dir_str,
                        self.voiceover_file_input_key: voiceover_filename_in_state,
                    }.items()
                    if not val
                ]
                error_msg = f"Missing required input keys in session state: {', '.join(missing_keys)}. Aborting."
                logger.error(f"[{self.name}] {error_msg}")
                yield text2event(self.name, error_msg)
                return

            project_base_path = Path(f"projects/{PROJECT_NAME}")
            videos_dir = Path(videos_base_dir_str)
            images_dir = Path(images_dir_str)
            voiceover_audio_path = project_base_path / voiceover_filename_in_state

            intro_video_path = videos_dir / self.intro_video_filename
            outro_video_path = videos_dir / self.outro_video_filename

            # Validate file existence
            for p, desc in [
                (intro_video_path, "Intro video"),
                (outro_video_path, "Outro video"),
                (voiceover_audio_path, "Voiceover audio"),
            ]:
                if not p.is_file():
                    error_msg = f"{desc} not found at {p}. Aborting."
                    logger.error(f"[{self.name}] {error_msg}")
                    yield text2event(self.name, error_msg)
                    return
            if not images_dir.is_dir():
                error_msg = f"Images directory not found at {images_dir}. Aborting."
                logger.error(f"[{self.name}] {error_msg}")
                yield text2event(self.name, error_msg)
                return

            yield text2event(self.name, "Loading assets for video assembly...")
            intro_clip = VideoFileClip(str(intro_video_path))
            outro_clip = VideoFileClip(str(outro_video_path))
            audio_clip = AudioFileClip(str(voiceover_audio_path))

            image_files = sorted(list(images_dir.glob("*.jpg")))
            image_clips = []
            if image_files:
                yield text2event(
                    self.name, f"Found {len(image_files)} images for the sequence."
                )
                intro_duration = intro_clip.duration or 0
                outro_duration = outro_clip.duration or 0
                audio_duration = audio_clip.duration or 0

                remaining_duration = audio_duration - (intro_duration + outro_duration)
                image_duration_per_clip = 0.1  # Default small duration

                if remaining_duration > 0 and image_files:
                    image_duration_per_clip = remaining_duration / len(image_files)
                elif image_files:  # remaining_duration <= 0 but images exist
                    logger.warning(
                        f"[{self.name}] Not enough audio duration for images or negative remaining duration ({remaining_duration:.2f}s). Using default duration {image_duration_per_clip}s per image."
                    )

                logger.info(
                    f"[{self.name}] Duration per image clip: {image_duration_per_clip:.2f} seconds"
                )
                image_clips = [
                    ImageClip(str(img_path), duration=image_duration_per_clip)
                    for img_path in image_files
                ]
            else:
                logger.info(
                    f"[{self.name}] No image files found in {images_dir} or remaining duration is not positive."
                )

            yield text2event(self.name, "Concatenating video clips...")
            clips_to_concatenate = [intro_clip] + image_clips + [outro_clip]
            final_video_clip = concatenate_videoclips(
                clips_to_concatenate, method="compose"
            )

            yield text2event(self.name, "Setting audio for the final video...")
            final_video_clip = final_video_clip.set_audio(audio_clip)
            if (
                final_video_clip.duration > audio_clip.duration + 0.1
            ):  # Add small tolerance
                logger.warning(
                    f"[{self.name}] Final video duration ({final_video_clip.duration:.2f}s) is longer than audio duration ({audio_clip.duration:.2f}s)."
                )
            elif final_video_clip.duration < audio_clip.duration - 0.1:
                logger.warning(
                    f"[{self.name}] Final video duration ({final_video_clip.duration:.2f}s) is shorter than audio duration ({audio_clip.duration:.2f}s). Video may end before audio does."
                )
                # Optionally trim audio: audio_clip = audio_clip.subclip(0, final_video_clip.duration)
                # final_video_clip = final_video_clip.set_audio(audio_clip)

            output_dir = project_base_path / self.output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            final_video_filepath = output_dir / self.output_filename

            yield text2event(
                self.name, f"Exporting final video to {final_video_filepath}..."
            )
            final_video_clip.write_videofile(
                str(final_video_filepath),
                codec="libx264",
                fps=self.fps,
                audio_codec="aac",
            )

            ctx.session.state[self.output_key] = str(final_video_filepath)
            logger.info(
                f"[{self.name}] Stored final video path '{final_video_filepath}' in session state key '{self.output_key}'."
            )
            final_message = (
                f"Video successfully assembled and saved to '{final_video_filepath}'."
            )
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An error occurred during video assembly: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)
        finally:
            # Close clips to free resources
            for clip_obj in (
                [intro_clip, outro_clip, audio_clip, final_video_clip] + image_clips
                if "image_clips" in locals()
                else []
            ):
                if clip_obj and hasattr(clip_obj, "close") and callable(clip_obj.close):
                    try:
                        clip_obj.close()
                    except Exception as e_close:
                        logger.error(f"[{self.name}] Error closing clip: {e_close}")


video_assembler_agent = VideoAssemblerAgent(
    name="VideoAssemblerAgent",
    description="Assembles video clips, images, and audio into a final video.",
    voiceover_file_input_key="voiceover_path",
    images_dir_input_key="images_path",
    videos_dir_input_key="videos_path",
    output_subdir="assembled_video",
    output_filename="short_video.mp4",
)
