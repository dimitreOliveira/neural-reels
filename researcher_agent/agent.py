import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing_extensions import override

from researcher_agent.agents.image_generator import (
    ImagenAgent,
    image_generator_agent,
)
from researcher_agent.agents.image_prompt_generator import (
    image_prompt_generator_agent,
)
from researcher_agent.agents.script_writer import script_writer_agent
from researcher_agent.agents.theme_definer import theme_definer_agent
from researcher_agent.agents.video_assembler import (
    VideoAssemblerAgent,
    video_assembler_agent,
)
from researcher_agent.agents.video_generator import (
    VeoAgent,
    video_generator_agent,
)
from researcher_agent.agents.video_prompt_generator import (
    video_prompt_generator_agent,
)
from researcher_agent.agents.voiceover_generator import (
    VoiceoverGeneratorAgent,
    voiceover_generator_agent,
)
from researcher_agent.utils.genai_utils import text2event

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCreatorWorkflowAgent(BaseAgent):
    """
    Orchestrates the video creation workflow, including user approval steps.
    """

    theme_definer: Agent
    script_writer: Agent
    image_prompt_generator: Agent
    video_prompt_generator: Agent
    voiceover_generator: VoiceoverGeneratorAgent
    image_generator: ImagenAgent
    video_generator: VeoAgent
    video_assembler: VideoAssemblerAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        theme_definer: Agent,
        script_writer: Agent,
        image_prompt_generator: Agent,
        video_prompt_generator: Agent,
        voiceover_generator: VoiceoverGeneratorAgent,
        image_generator: ImagenAgent,
        video_generator: VeoAgent,
        video_assembler: VideoAssemblerAgent,
    ):
        sub_agents_list = [
            theme_definer,
            script_writer,
            image_prompt_generator,
            video_prompt_generator,
            voiceover_generator,
            image_generator,
            video_generator,
            video_assembler,
        ]
        super().__init__(
            name=name,
            theme_definer=theme_definer,
            script_writer=script_writer,
            image_prompt_generator=image_prompt_generator,
            video_prompt_generator=video_prompt_generator,
            voiceover_generator=voiceover_generator,
            image_generator=image_generator,
            video_generator=video_generator,
            video_assembler=video_assembler,
            sub_agents=sub_agents_list,
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info("=" * 100)
        logger.info(
            f"\n\n[{self.name}] Starting short content video creation workflow.\n\n"
        )

        # Setup
        PROJECT_NAME = os.environ.get("PROJECT_NAME")
        assets_path = f"projects/{PROJECT_NAME}"
        # TODO: Pass the assets to the context and retrieve from the agens
        # ctx.session.state["assets_path"] = assets_path

        # # 1. Define Theme
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.theme_definer.name}...\n\n")
        async for event in self.theme_definer.run_async(ctx):
            yield event

        if (
            self.theme_definer.output_key not in ctx.session.state
            or not ctx.session.state[self.theme_definer.output_key]
        ):
            error_msg = f"[{self.name}] {self.theme_definer.name} did not produce '{self.theme_definer.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Theme defined: {ctx.session.state[self.theme_definer.output_key]}."
        )

        # 2. Script creation
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.script_writer.name}...\n\n")
        async for event in self.script_writer.run_async(ctx):
            yield event

        if (
            self.script_writer.output_key not in ctx.session.state
            or not ctx.session.state[self.script_writer.output_key]
        ):
            error_msg = f"[{self.name}] {self.script_writer.name} did not produce '{self.script_writer.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Script created: {ctx.session.state[self.script_writer.output_key]}."
        )

        # 3. Image prompts generation
        logger.info("=" * 100)
        logger.info(
            f"\n\n[{self.name}] Running {self.image_prompt_generator.name}...\n\n"
        )
        async for event in self.image_prompt_generator.run_async(ctx):
            yield event

        if (
            self.image_prompt_generator.output_key not in ctx.session.state
            or not ctx.session.state[self.image_prompt_generator.output_key]
        ):
            error_msg = f"[{self.name}] {self.image_prompt_generator.name} did not produce '{self.image_prompt_generator.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Image prompts generated: {ctx.session.state[self.image_prompt_generator.output_key]}."
        )

        # 4. Video prompts generation
        logger.info("=" * 100)
        logger.info(
            f"\n\n[{self.name}] Running {self.video_prompt_generator.name}...\n\n"
        )
        async for event in self.video_prompt_generator.run_async(ctx):
            yield event

        if (
            self.video_prompt_generator.output_key not in ctx.session.state
            or not ctx.session.state[self.video_prompt_generator.output_key]
        ):
            error_msg = f"[{self.name}] {self.video_prompt_generator.name} did not produce '{self.video_prompt_generator.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Video prompts generated: {ctx.session.state[self.video_prompt_generator.output_key]}."
        )

        # 5. Voiceover generation
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.voiceover_generator.name}...\n\n")
        async for event in self.voiceover_generator.run_async(ctx):
            yield event

        if (
            self.voiceover_generator.output_key not in ctx.session.state
            or not ctx.session.state[self.voiceover_generator.output_key]
        ):
            error_msg = f"[{self.name}] {self.voiceover_generator.name} did not produce '{self.voiceover_generator.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Voiceover generated: {ctx.session.state[self.voiceover_generator.output_key]}."
        )

        # 6. Image generation
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.image_generator.name}...\n\n")
        async for event in self.image_generator.run_async(ctx):
            yield event

        if (
            self.image_generator.output_key not in ctx.session.state
            or not ctx.session.state[self.image_generator.output_key]
        ):
            error_msg = f"[{self.name}] {self.image_generator.name} did not produce '{self.image_generator.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Images generated: {ctx.session.state[self.image_generator.output_key]}."
        )

        # 7. Video generation
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.video_generator.name}...\n\n")
        async for event in self.video_generator.run_async(ctx):
            yield event

        if (
            self.video_generator.output_key not in ctx.session.state
            or not ctx.session.state[self.video_generator.output_key]
        ):
            error_msg = f"[{self.name}] {self.video_generator.name} did not produce '{self.video_generator.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Videos generated: {ctx.session.state[self.video_generator.output_key]}."
        )

        # 7. Video assembling
        logger.info("=" * 100)
        logger.info(f"\n\n[{self.name}] Running {self.video_assembler.name}...\n\n")
        async for event in self.video_assembler.run_async(ctx):
            yield event

        if (
            self.video_assembler.output_key not in ctx.session.state
            or not ctx.session.state[self.video_assembler.output_key]
        ):
            error_msg = f"[{self.name}] {self.video_assembler.name} did not produce '{self.video_assembler.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] Short video content assembled: {ctx.session.state[self.video_assembler.output_key]}."
        )

        # Workflow Completion Summary
        completion_msg = f"Short video content creation workflow finished, assests saved to '{assets_path}'"
        yield text2event(self.name, completion_msg)
        logger.info(f"[{self.name}] {completion_msg}")

        logger.info("=" * 50)
        logger.info(
            f"\n\n[{self.name}] Finishing short content video creation workflow.\n\n"
        )
        return


video_creator_workflow_agent = VideoCreatorWorkflowAgent(
    name="VideoCreatorWorkflowAgent",
    theme_definer=theme_definer_agent,
    script_writer=script_writer_agent,
    image_prompt_generator=image_prompt_generator_agent,
    video_prompt_generator=video_prompt_generator_agent,
    voiceover_generator=voiceover_generator_agent,
    image_generator=image_generator_agent,
    video_generator=video_generator_agent,
    video_assembler=video_assembler_agent,
)

root_agent = video_creator_workflow_agent
