import logging
import random
import re
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx import Loop, TimeMirror
from pydantic import Field
from typing_extensions import override

from app.utils.genai_utils import text2event

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

    def _apply_random_effect_to_img(self, image_clip: ImageClip) -> CompositeVideoClip:
        """Applies a random visual effect to an ImageClip.

        This method selects a random zoom or pan effect from a predefined list
        and applies it to the given image clip to create simple motion. It also
        randomly decides whether to reverse the clip's playback.

        Args:
            image_clip: The MoviePy ImageClip to apply the effect to.

        Returns:
            A MoviePy CompositeVideoClip with the applied effect.
        """
        effects_with_names = [
            ("No effect", lambda clip: clip.resized(lambda t: 1)),
            ("Zoom-in (very slow)", lambda clip: clip.resized(lambda t: 1 + 0.01 * t)),
            ("Zoom-in (slow)", lambda clip: clip.resized(lambda t: 1 + 0.03 * t)),
            ("Zoom-in (fast)", lambda clip: clip.resized(lambda t: 1 + 0.06 * t)),
            (
                "Zoom-in (sin-based)",
                lambda clip: clip.resized(lambda t: 1.3 + 0.3 * np.sin(t / 3)),
            ),
        ]

        # Choose a random effect
        effect_name, chosen_effect_func = random.choice(effects_with_names)
        logger.info(f"[{self.name}] Applying effect: '{effect_name}' to image clip.")

        # Apply the chosen effect
        try:
            image_clip = chosen_effect_func(image_clip)
        except Exception as e:
            logger.warning(
                f"[{self.name}] Failed to apply effect '{effect_name}': {e}. Skipping effect.",
                exc_info=True,
            )
            # Fallback to no effect if the chosen one fails
            pass

        image_clip = image_clip.with_position(("center", "center"))
        image_clip.fps = self.fps
        image_clip = CompositeVideoClip([image_clip], size=image_clip.size)

        reverse_clip = random.choice([True, False])
        if reverse_clip:
            image_clip = TimeMirror().apply(image_clip)
            logger.info(f"[{self.name}] Applying effect: 'reverse' to image clip.")

        return image_clip

    def _assemble_scene_clip(
        self, scene_idx: int, scene: str, assets_path: Path
    ) -> Optional[CompositeVideoClip]:
        """Assembles a single video clip for a given scene.

        This method gathers all assets (voiceover, images, videos) for a
        specific scene, combines them into a single video clip using MoviePy,
        and sets the voiceover as the audio. It prioritizes video assets and
        uses image assets to fill the remaining duration of the voiceover.

        Args:
            scene_idx: The index of the scene being assembled.
            scene: The path to the scene's asset directory.
            assets_path: The root path for the project's assets.

        Returns:
            A MoviePy CompositeVideoClip for the scene, or None if no assets
            were available to create a clip.
        """
        scene_clip = None
        scene_voiceovers = list(Path(f"{scene}/{self.voiceover_subdir}").glob("*.wav"))
        scene_images = list(Path(f"{scene}/{self.images_subdir}").glob("*.jpg"))
        scene_videos = list(Path(f"{scene}/{self.videos_subdir}").glob("*.mp4"))

        logger.info(f"[{self.name}] \t {len(scene_voiceovers)} voiceover(s) available")
        logger.info(f"[{self.name}] \t {len(scene_images)} image(s) available")
        logger.info(f"[{self.name}] \t {len(scene_videos)} video(s) available")

        # compose a subclip for each of scene
        for voiceover_idx, scene_voiceover in enumerate(scene_voiceovers):
            logger.info(
                f"[{self.name}] Starting assembly of clip {voiceover_idx + 1} for scene '{scene_idx + 1}'"
            )
            # Load audio clips from the voiceovers
            audio_clip = AudioFileClip(str(scene_voiceover))
            # Load video clips from the videos
            video_clips = [
                VideoFileClip(str(video_path)) for video_path in scene_videos
            ]
            # Load image clips from the images
            image_clips = [ImageClip(str(img_path)) for img_path in scene_images]

            # Prioritize using videos if available
            remaining_duration = audio_clip.duration
            logger.info(f"[{self.name}] \t Voiceover duration {audio_clip.duration}")
            if video_clips:
                remaining_duration -= sum(
                    video_clip.duration for video_clip in video_clips
                )
            else:
                logger.info(
                    f"[{self.name}] \t No videos available for scene '{scene_idx + 1}'"
                )

            if image_clips and remaining_duration > 0:
                image_duration_per_clip = remaining_duration / len(image_clips)
                logger.info(
                    f"[{self.name}] \t image_duration_per_clip {image_duration_per_clip}"
                )
                # Update image clips durations based on the remaining clip duration
                image_clips = [
                    image_clip.with_duration(image_duration_per_clip)
                    for image_clip in image_clips
                ]
                # Add random effect to image clips
                image_clips = [
                    self._apply_random_effect_to_img(img_clip)
                    for img_clip in image_clips
                ]
            else:
                # TODO: If there are not images and we still need audio to fill,
                # we need to extend the duration of the available videos
                image_clips = []
                logger.info(
                    f"[{self.name}] \t No images available for scene '{scene_idx + 1}'."
                )

            if video_clips or image_clips:
                # Concatenaten the clips
                scene_clip = concatenate_videoclips(
                    video_clips + image_clips, method="compose"
                )

                if scene_clip.duration > audio_clip.duration:
                    scene_clip = scene_clip.subclipped(0, audio_clip.duration)
                elif audio_clip.duration > scene_clip.duration:
                    scene_clip = Loop(duration=audio_clip.duration).apply(scene_clip)

                # Set voiceover as the clip audio
                scene_clip = scene_clip.with_audio(audio_clip)

                # Save scene clip
                scene_clip_outputpath = (
                    assets_path
                    / f"scene_{scene_idx + 1}/{self.output_subdir}/voiceover_{voiceover_idx + 1}_{self.output_filename}"
                )
                scene_clip_outputpath.parent.mkdir(parents=True, exist_ok=True)
                scene_clip.write_videofile(
                    str(scene_clip_outputpath),
                    codec=self.codec,
                    fps=self.fps,
                )
            else:
                logger.info(
                    f"[{self.name}] \t Neither videos nor images available for scene '{scene_idx + 1}'"
                )
        return scene_clip

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Assembles the final video from individual scene clips.

        This is the main entry point for the agent. It finds all scene
        directories, assembles a clip for each one using `_assemble_scene_clip`,
        and then concatenates all scene clips into a final video. The path to
        the final video is stored in the session state.

        Args:
            ctx: The invocation context, containing the path to the assets.

        Yields:
            Events indicating the progress of the video assembly.
        """
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
            scene_clip = self._assemble_scene_clip(scene_idx, scene, assets_path)

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
            str(final_video_outputpath),
            codec=self.codec,
            fps=self.fps,
        )

        ctx.session.state[self.output_key] = str(final_video_outputpath)
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
