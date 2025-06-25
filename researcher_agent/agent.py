import logging
from enum import Enum
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
from researcher_agent.agents.user_feedback import (
    user_feedback_agent,
)
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


class WorkflowStage(Enum):
    THEME_DEFINITION = 1
    SCRIPT_REFINEMENT = 2
    VIDEO_CREATION = 3


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCreatorWorkflowAgent(BaseAgent):
    """
    Orchestrates the video creation workflow, including user approval steps.
    """

    script_writer: Agent
    user_feedback: Agent
    theme_definer: Agent
    scene_breakdown: Agent
    image_prompt_generator: Agent
    video_prompt_generator: Agent
    voiceover_generator: VoiceoverGeneratorAgent
    image_generator: ImagenAgent
    video_generator: VeoAgent
    video_assembler: VideoAssemblerAgent
    workflow_stage: WorkflowStage = WorkflowStage.THEME_DEFINITION
    theme_approved: bool = False
    script_approved: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        script_writer: Agent,
        user_feedback: Agent,
        theme_definer: Agent,
        scene_breakdown: Agent,
        image_prompt_generator: Agent,
        video_prompt_generator: Agent,
        voiceover_generator: VoiceoverGeneratorAgent,
        image_generator: ImagenAgent,
        video_generator: VeoAgent,
        video_assembler: VideoAssemblerAgent,
    ):
        sub_agents_list = [
            script_writer,
            user_feedback,
            theme_definer,
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
            script_writer=script_writer,
            user_feedback=user_feedback,
            theme_definer=theme_definer,
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

    async def _define_theme_and_ask_for_feedback(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 1. Define Theme
        async for event in self._run_sub_agent(self.theme_definer, ctx):
            yield event

        # 2. Ask for user feedback
        theme = ctx.session.state[self.theme_definer.output_key].get(
            self.theme_definer.output_key
        )
        yield text2event(
            self.name,
            f"It seems that you want to create a short video content about '{theme}' is this correct?\nAnswer with 'yes' or describe what theme you want.",
        )

        self.theme_approved = True

    async def _draft_script_and_ask_for_feedback(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 1. Script creation feedback loop
        async for event in self._run_sub_agent(self.script_writer, ctx):
            yield event

        ctx.session.state["current_script"] = ctx.session.state.get(
            self.script_writer.output_key
        ).get(self.script_writer.output_key)

        # 2. Ask for user feedback
        yield text2event(
            self.name,
            "Do you approve this script?\nAnswer with 'yes' or 'no' or provide feedback for improvement.",
        )
        self.script_approved = True

    async def _setup_assets_folder(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        theme_output = ctx.session.state[self.theme_definer.output_key]
        theme = (
            theme_output[self.theme_definer.output_key] if theme_output else "default"
        )

        assets_path = f"projects/{theme}".lower().replace(" ", "_")
        # set the theme as the output folder
        ctx.session.state["assets_path"] = assets_path

        assets_path_msg = f"Generated assets will be stored at: '{assets_path}'."
        logger.info(f"[{self.name}] {assets_path_msg}")
        yield text2event(self.name, assets_path_msg)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(
            f"\n\n[{self.name}] Starting short content video creation workflow.\n\n"
        )

        if self.workflow_stage == WorkflowStage.THEME_DEFINITION:
            if not self.theme_approved:
                # 1. Theme definition feedback loop
                async for event in self._define_theme_and_ask_for_feedback(ctx):
                    yield event
                return
            else:
                # 1.1. Process user's feedback
                async for event in self._run_sub_agent(self.user_feedback, ctx):
                    yield event

                user_input = ctx.session.state.get(self.user_feedback.output_key).get(
                    self.user_feedback.output_key
                )

                # Theme not approved
                if user_input.lower() != "approved":
                    # 1.2. If not approved keep iterating
                    self.theme_approved = False
                    async for event in self._define_theme_and_ask_for_feedback(ctx):
                        yield event
                    return
                # Theme approved
                else:
                    self.workflow_stage = WorkflowStage.SCRIPT_REFINEMENT
                    yield text2event(
                        self.name, "Theme approved moving to script refinement stage"
                    )

                    async for event in self._setup_assets_folder(ctx):
                        yield event

                    # 2. Script creation feedback loop
                    async for event in self._draft_script_and_ask_for_feedback(ctx):
                        yield event
                    return
        elif self.workflow_stage == WorkflowStage.SCRIPT_REFINEMENT:
            # 2.1. Process user's feedback
            async for event in self._run_sub_agent(self.user_feedback, ctx):
                yield event

            user_input = ctx.session.state.get(self.user_feedback.output_key).get(
                self.user_feedback.output_key
            )

            # Script not approved
            if user_input.lower() != "approved":
                # 2.2. If not approved keep iterating
                self.script_approved = False
                async for event in self._draft_script_and_ask_for_feedback(ctx):
                    yield event
                return
            # Script approved
            else:
                self.workflow_stage = WorkflowStage.VIDEO_CREATION
                yield text2event(
                    self.name, "Script approved, starting the video generation process."
                )
        elif self.workflow_stage == WorkflowStage.VIDEO_CREATION:
            # 5. Scene breakdown
            async for event in self._run_sub_agent(self.scene_breakdown, ctx):
                yield event

            # 6. Image prompts generation
            async for event in self._run_sub_agent(self.image_prompt_generator, ctx):
                yield event

            # 7. Video prompts generation
            async for event in self._run_sub_agent(self.video_prompt_generator, ctx):
                yield event

            # 8. Voiceover generation
            async for event in self._run_sub_agent(self.voiceover_generator, ctx):
                yield event

            # 9. Image generation
            async for event in self._run_sub_agent(self.image_generator, ctx):
                yield event

            # 10. Video generation
            async for event in self._run_sub_agent(self.video_generator, ctx):
                yield event

            # 11. Video assembling
            async for event in self._run_sub_agent(self.video_assembler, ctx):
                yield event

            yield text2event(
                self.name,
                f"Short video content creation workflow finished. Video stored at: '{ctx.session.state['assets_path']}'.",
            )

            logger.info(
                f"\n\n[{self.name}] Finishing short content video creation workflow.\n\n"
            )

        return


video_creator_workflow_agent = VideoCreatorWorkflowAgent(
    name="VideoCreatorWorkflowAgent",
    script_writer=script_writer_agent,
    user_feedback=user_feedback_agent,
    theme_definer=theme_definer_agent,
    scene_breakdown=scene_breakdown_agent,
    image_prompt_generator=image_prompt_generator_agent,
    video_prompt_generator=video_prompt_generator_agent,
    voiceover_generator=voiceover_generator_agent,
    image_generator=image_generator_agent,
    video_generator=video_generator_agent,
    video_assembler=video_assembler_agent,
)

root_agent = video_creator_workflow_agent
