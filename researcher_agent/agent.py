import logging
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
from researcher_agent.agents.scene_breakdown import (
    scene_breakdown_agent,
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
    scene_breakdown: Agent
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
        scene_breakdown: Agent,
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
            scene_breakdown,
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
            scene_breakdown=scene_breakdown,
            image_prompt_generator=image_prompt_generator,
            video_prompt_generator=video_prompt_generator,
            voiceover_generator=voiceover_generator,
            image_generator=image_generator,
            video_generator=video_generator,
            video_assembler=video_assembler,
            sub_agents=sub_agents_list,
        )

    async def _run_sub_agent(
        self, agent: BaseAgent, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Helper function to run a sub-agent, yield its events, and validate its output."""
        logger.info("\n\n\n")
        logger.info("=" * 150)
        logger.info(f"[{self.name}] Running {agent.name}...")

        async for event in agent.run_async(ctx):
            yield event

        agent_output = ctx.session.state.get(agent.output_key)
        if not agent_output:
            error_msg = f"[{self.name}] {agent.name} did not produce '{agent.output_key}' in session state. Aborting workflow."
            logger.error(error_msg)
            yield text2event(self.name, error_msg)
            return

        logger.info(
            f"[{self.name}] {agent.name} completed. Output for '{agent.output_key}': {agent_output}"
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"\n\n[{self.name}] Starting short content video creation workflow.\n\n"
        )

        # 1. Define Theme
        async for event in self._run_sub_agent(self.theme_definer, ctx):
            yield event

        theme = ctx.session.state[self.theme_definer.output_key].get(
            self.theme_definer.output_key, "default"
        )

        assets_path = f"projects/{theme}".lower().replace(" ", "_")
        # set the theme as the output folder
        ctx.session.state["assets_path"] = assets_path
        logger.info(
            f"[{self.name}] Generated assets will be stored at: '{assets_path}'."
        )

        # 2. Script creation
        async for event in self._run_sub_agent(self.script_writer, ctx):
            yield event

        # 3. Scene breakdown
        async for event in self._run_sub_agent(self.scene_breakdown, ctx):
            yield event

        # 4. Image prompts generation
        async for event in self._run_sub_agent(self.image_prompt_generator, ctx):
            yield event

        # # 5. Video prompts generation
        # async for event in self._run_sub_agent(self.video_prompt_generator, ctx):
        #     yield event

        # # 6. Voiceover generation
        # async for event in self._run_sub_agent(self.voiceover_generator, ctx):
        #     yield event

        # 7. Image generation
        async for event in self._run_sub_agent(self.image_generator, ctx):
            yield event

        # # 8. Video generation
        # async for event in self._run_sub_agent(self.video_generator, ctx):
        #     yield event

        # # 9. Video assembling
        # async for event in self._run_sub_agent(self.video_assembler, ctx):
        #     yield event

        # Workflow Completion Summary
        completion_msg = f"Short video content creation workflow finished, assests saved to '{assets_path}'"
        logger.info(f"[{self.name}] {completion_msg}")
        yield text2event(self.name, completion_msg)

        logger.info(
            f"\n\n[{self.name}] Finishing short content video creation workflow.\n\n"
        )
        return


video_creator_workflow_agent = VideoCreatorWorkflowAgent(
    name="VideoCreatorWorkflowAgent",
    theme_definer=theme_definer_agent,
    script_writer=script_writer_agent,
    scene_breakdown=scene_breakdown_agent,
    image_prompt_generator=image_prompt_generator_agent,
    video_prompt_generator=video_prompt_generator_agent,
    voiceover_generator=voiceover_generator_agent,
    image_generator=image_generator_agent,
    video_generator=video_generator_agent,
    video_assembler=video_assembler_agent,
)

root_agent = video_creator_workflow_agent
