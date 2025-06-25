import logging
import re
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
    voiceover_subdir: str = Field(
        default="voiceovers",
        description="Key in session state for the voiceover audio filename (from VoiceoverGeneratorAgent).",
    )
    images_subdir: str = Field(
        default="images",
        description="Key in session state for the directory containing image files (from ImageGeneratorAgent).",
    )
    videos_subdir: str = Field(
        default="videos",
        description="Key in session state for the directory containing intro/outro videos (from VideoGeneratorAgent).",
    )
    # --- Configuration for specific asset names and output ---
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
    codec: str = Field(
        default="libx264", description="Video codec for the output video."
    )

    model_config = {"arbitrary_types_allowed": True}

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"[{self.name}] Starting video assembly.")

        # Setup
        assets_path = Path(ctx.session.state.get("assets_path"))

        # Available scenes
        scenes = [
            str(x) for x in assets_path.glob("*") if (x.is_dir() and "scene_" in x.name)
        ]
        scenes = sorted(scenes, key=lambda s: int(re.search(r"\d+", s).group()))

        logger.info(f"[{self.name}] Found '{len(scenes)}' scenes available")
        yield text2event(
            self.name, f"[{self.name}] Found '{len(scenes)}' scenes available"
        )

        scene_clips = []
        for scene_idx, scene in enumerate(scenes):
            scene_clip = None
            scene_voiceovers = list(
                Path(f"{scene}/{self.voiceover_subdir}").glob("*.wav")
            )
            scene_images = list(Path(f"{scene}/{self.images_subdir}").glob("*.jpg"))
            scene_videos = list(Path(f"{scene}/{self.videos_subdir}").glob("*.mp4"))

            logger.info(
                f"[{self.name}] \t {len(scene_voiceovers)} voiceover(s) available"
            )
            logger.info(f"[{self.name}] \t {len(scene_images)} image(s) available")
            logger.info(f"[{self.name}] \t {len(scene_videos)} video(s) available")

            # compose a subclip for each of scene
            for voiceover_idx, scene_voiceover in enumerate(scene_voiceovers):
                logger.info(
                    f"[{self.name}] Starting assembly of clip {voiceover_idx + 1} for scene '{scene_idx + 1}'"
                )
                # Load audio clip from the voiceovers
                audio_clip = AudioFileClip(scene_voiceover)
                # Load video clips from the videos
                video_clips = [
                    VideoFileClip(str(video_path)) for video_path in scene_videos
                ]
                # Prioritize using videos if available
                remaining_duration = audio_clip.duration
                logger.info(
                    f"[{self.name}] \t Voiceover duration {audio_clip.duration}"
                )
                if scene_videos:
                    for video_clip in video_clips:
                        remaining_duration -= video_clip.duration
                else:
                    logger.info(
                        f"[{self.name}] \t No videos available for scene '{scene_idx + 1}'"
                    )

                if scene_images and remaining_duration > 0:
                    image_duration_per_clip = remaining_duration / len(scene_images)
                    logger.info(
                        f"[{self.name}] \t image_duration_per_clip {image_duration_per_clip}"
                    )
                    # Load image clips from the images (based on the remaining clip duration)
                    image_clips = [
                        ImageClip(str(img_path), duration=image_duration_per_clip)
                        for img_path in scene_images
                    ]
                else:
                    image_clips = []
                    logger.info(
                        f"[{self.name}] \t No images available for scene '{scene_idx + 1}'"
                    )

                if video_clips or image_clips:
                    # Concatenaten the clips
                    scene_clip = concatenate_videoclips(
                        video_clips + image_clips, method="compose"
                    )

                    # Set voiceover as the clip audio
                    scene_clip = scene_clip.with_audio(audio_clip)

                    # Save scene clip
                    scene_clip_outputpath = (
                        assets_path
                        / f"scene_{scene_idx + 1}/{self.output_subdir}/voiceover_{voiceover_idx + 1}_{self.output_filename}"
                    )
                    scene_clip_outputpath.parent.mkdir(parents=True, exist_ok=True)
                    scene_clip.write_videofile(
                        scene_clip_outputpath,
                        codec=self.codec,
                        fps=self.fps,
                    )
                else:
                    logger.info(
                        f"[{self.name}] \t Neither videos nor images available for scene '{scene_idx + 1}'"
                    )

            if scene_clip:
                # Keeping the last generated clip for the sake of simplicity
                scene_clips.append(scene_clip)
            else:
                logger.info(
                    f"[{self.name}] \t Skipping scene clip for scene '{scene_idx + 1}'"
                )

        # outside of this "scene loop" combine all the subclips
        final_video_clip = concatenate_videoclips(scene_clips, method="compose")

        # Combine all scene clips into the final video
        final_video_outputpath = (
            assets_path / f"{self.output_subdir}/{self.output_filename}"
        )
        final_video_outputpath.parent.mkdir(parents=True, exist_ok=True)
        final_video_clip.write_videofile(
            final_video_outputpath,
            codec=self.codec,
            fps=self.fps,
        )

        ctx.session.state[self.output_key] = final_video_outputpath
        completion_msg = (
            f"Video assembly finished. Video stored at: '{final_video_outputpath}'"
        )
        logger.info(f"[{self.name}] {completion_msg}")
        yield text2event(self.name, completion_msg)


video_assembler_agent = VideoAssemblerAgent(
    name="VideoAssemblerAgent",
    description="Assembles video clips, images, and audio into a final video.",
    voiceover_subdir="voiceovers",
    images_subdir="images",
    videos_subdir="videos",
    output_key="assembled_video_path",
    output_subdir="assembled_video",
    output_filename="short_video.mp4",
)
