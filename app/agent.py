import logging
from enum import Enum
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing_extensions import override

from app.agents.expert_researcher import (
    expert_researcher_agent,
)
from app.agents.image_generator import (
    ImagenAgent,
    image_generator_agent,
)
from app.agents.image_prompt_generator import (
    image_prompt_generator_agent,
)
from app.agents.research_compiler import (
    research_compiler_agent,
)
from app.agents.scene_breakdown import (
    scene_breakdown_agent,
)
from app.agents.script_writer import script_writer_agent
from app.agents.seo_optimizer import (
    seo_optimizer_agent,
)
from app.agents.theme_definer import theme_definer_agent
from app.agents.user_feedback import (
    user_feedback_agent,
)
from app.agents.video_assembler import (
    VideoAssemblerAgent,
    video_assembler_agent,
)
from app.agents.video_generator import (
    VeoAgent,
    video_generator_agent,
)
from app.agents.video_prompt_generator import (
    video_prompt_generator_agent,
)
from app.agents.voiceover_generator import (
    VoiceoverGeneratorAgent,
    voiceover_generator_agent,
)
from app.agents.web_researcher import (
    web_researcher_agent,
)
from app.utils.genai_utils import text2event


class WorkflowStage(Enum):
    THEME_DEFINITION = 1
    SCRIPT_REFINEMENT = 2


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCreatorWorkflowAgent(BaseAgent):
    """
    Orchestrates the video creation workflow, including user approval steps.
    """

    theme_definer: Agent
    user_feedback: Agent
    expert_researcher: Agent
    web_researcher: Agent
    research_compiler: Agent
    script_writer: Agent
    scene_breakdown: Agent
    image_prompt_generator: Agent
    video_prompt_generator: Agent
    voiceover_generator: VoiceoverGeneratorAgent
    image_generator: ImagenAgent
    video_generator: VeoAgent
    video_assembler: VideoAssemblerAgent
    seo_optimizer: Agent
    workflow_stage: WorkflowStage = WorkflowStage.THEME_DEFINITION
    theme_approved: bool = False
    script_approved: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        theme_definer: Agent,
        user_feedback: Agent,
        expert_researcher: Agent,
        web_researcher: Agent,
        research_compiler: Agent,
        script_writer: Agent,
        scene_breakdown: Agent,
        image_prompt_generator: Agent,
        video_prompt_generator: Agent,
        voiceover_generator: VoiceoverGeneratorAgent,
        image_generator: ImagenAgent,
        video_generator: VeoAgent,
        video_assembler: VideoAssemblerAgent,
        seo_optimizer: Agent,
    ):
        """Initializes the VideoCreatorWorkflowAgent with all its sub-agents.

        Args:
            name: The name of the agent.
            theme_definer: Agent to define the video's theme.
            user_feedback: Agent to process user feedback.
            expert_researcher: Agent for internal knowledge-based research.
            web_researcher: Agent for web-based research.
            research_compiler: Agent to compile research findings.
            script_writer: Agent to write the video script.
            scene_breakdown: Agent to break the script into scenes.
            image_prompt_generator: Agent to generate image prompts.
            video_prompt_generator: Agent to generate video prompts.
            voiceover_generator: Agent to generate voiceovers.
            image_generator: Agent to generate images.
            video_generator: Agent to generate videos.
            video_assembler: Agent to assemble the final video.
            seo_optimizer: Agent to optimize title and description for SEO.
        """
        sub_agents_list = [
            theme_definer,
            user_feedback,
            expert_researcher,
            web_researcher,
            research_compiler,
            script_writer,
            scene_breakdown,
            image_prompt_generator,
            video_prompt_generator,
            voiceover_generator,
            image_generator,
            video_generator,
            video_assembler,
            seo_optimizer,
        ]

        super().__init__(
            name=name,
            theme_definer=theme_definer,
            user_feedback=user_feedback,
            expert_researcher=expert_researcher,
            web_researcher=web_researcher,
            research_compiler=research_compiler,
            script_writer=script_writer,
            scene_breakdown=scene_breakdown,
            image_prompt_generator=image_prompt_generator,
            video_prompt_generator=video_prompt_generator,
            voiceover_generator=voiceover_generator,
            image_generator=image_generator,
            video_generator=video_generator,
            video_assembler=video_assembler,
            seo_optimizer=seo_optimizer,
            sub_agents=sub_agents_list,
        )

    async def _run_sub_agent(
        self, agent: BaseAgent, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Runs a sub-agent, yields its events, and validates its output.

        This helper function executes a given sub-agent within the current
        invocation context. It streams the events generated by the sub-agent.
        After the sub-agent finishes, it checks if the expected output was
        written to the session state. If not, it logs an error and yields an
        error event.

        Args:
            agent: The sub-agent to run.
            ctx: The current invocation context.

        Yields:
            Events from the sub-agent.
        """
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
        """Defines the video theme and asks for user feedback.

        This method runs the theme definer agent to determine the video's theme
        based on user input. It then asks the user for confirmation before
        proceeding.

        Args:
            ctx: The current invocation context.

        Yields:
            Events from the theme definer agent and the feedback request.
        """
        # 1. Define Theme
        async for event in self._run_sub_agent(self.theme_definer, ctx):
            yield event

        # 2. Ask for user feedback
        theme = ctx.session.state[self.theme_definer.output_key]["theme"]
        yield text2event(
            self.name,
            f"It seems that you want to create a short video content about '{theme}' is this correct?\n\nAnswer with 'yes' or describe what theme you want.",
        )

        self.theme_approved = True

    async def _draft_script_and_ask_for_feedback(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Drafts a video script and asks for user feedback.

        This method runs the script writer agent to generate a draft of the
        video script. It then asks the user for approval or feedback for
        revisions.

        Args:
            ctx: The current invocation context.

        Yields:
            Events from the script writer agent and the feedback request.
        """
        # 1. Script creation feedback loop
        async for event in self._run_sub_agent(self.script_writer, ctx):
            yield event

        ctx.session.state["current_script"] = ctx.session.state.get(
            self.script_writer.output_key
        )

        # 2. Ask for user feedback
        yield text2event(
            self.name,
            "Do you approve this script?\nAnswer with 'yes' or 'no' or provide feedback for improvement.",
        )
        self.script_approved = True

    async def _setup_assets_folder(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Sets up the directory for storing generated assets.

        Based on the video theme, this method creates a unique directory path
        for all generated assets (images, videos, audio, etc.) and stores
        this path in the session state.

        Args:
            ctx: The current invocation context.

        Yields:
            An event confirming the asset path.
        """
        theme_intent = ctx.session.state[self.theme_definer.output_key]

        theme = theme_intent["theme"]
        assets_path = f"projects/{theme}".lower().replace(" ", "_")

        # Update session state
        ctx.session.state["theme"] = theme
        ctx.session.state["intent"] = theme_intent["user_intent"]
        # set the theme as the output folder
        ctx.session.state["assets_path"] = assets_path

        assets_path_msg = f"Generated assets will be stored at: '{assets_path}'."
        logger.info(f"[{self.name}] {assets_path_msg}")
        yield text2event(self.name, assets_path_msg)

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Executes the main video creation workflow.

        This method orchestrates the entire process of creating a short video,
        from theme definition and script writing to asset generation and final
        video assembly. It manages the workflow state, handles user feedback
        loops, and calls the appropriate sub-agents at each stage.

        The workflow is divided into stages:
        1.  **Theme Definition**: The agent proposes a theme and asks for user
            approval. It iterates until the theme is approved.
        2.  **Script Refinement**: After research, the agent drafts a script
            and asks for user approval. It iterates until the script is
            approved.
        3.  **Asset Generation & Assembly**: Once the script is approved, the
            agent generates all necessary assets (voiceover, images, videos)
            and assembles them into the final video.

        Args:
            ctx: The invocation context, containing the session state.

        Yields:
            Events indicating the progress and results of the workflow.
        """
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

                    # 2. Theme research
                    yield text2event(self.name, "Starting research")

                    # 2.1. Expert research
                    yield text2event(self.name, "Prompting expert researcher...")
                    async for event in self._run_sub_agent(self.expert_researcher, ctx):
                        yield event

                    # 2.2. Web researcher
                    yield text2event(self.name, "Running web research...")
                    async for event in self._run_sub_agent(self.web_researcher, ctx):
                        yield event

                    # 2.3. Research compiler
                    yield text2event(self.name, "Compiling researched content...")
                    async for event in self._run_sub_agent(self.research_compiler, ctx):
                        yield event

                    # 3. Script creation feedback loop
                    yield text2event(
                        self.name, "Research finished, starting script creation"
                    )
                    async for event in self._draft_script_and_ask_for_feedback(ctx):
                        yield event
                    return
        elif self.workflow_stage == WorkflowStage.SCRIPT_REFINEMENT:
            # This needs to be reset
            async for event in self._setup_assets_folder(ctx):
                yield event

            # 3.1. Process user's feedback
            async for event in self._run_sub_agent(self.user_feedback, ctx):
                yield event

            user_input = ctx.session.state.get(self.user_feedback.output_key).get(
                self.user_feedback.output_key
            )

            # Script not approved
            if user_input.lower() != "approved":
                # 3.2. If not approved keep iterating
                self.script_approved = False
                async for event in self._draft_script_and_ask_for_feedback(ctx):
                    yield event
                return
            # Script approved
            else:
                yield text2event(
                    self.name, "Script approved, starting the video generation process."
                )

                # 4. Scene breakdown
                yield text2event(self.name, "Breaking script into scenes...")
                async for event in self._run_sub_agent(self.scene_breakdown, ctx):
                    yield event

                # 5. Image prompts generation
                yield text2event(self.name, "Generating prompts for the images...")
                async for event in self._run_sub_agent(
                    self.image_prompt_generator, ctx
                ):
                    yield event

                # 6. Video prompts generation
                yield text2event(self.name, "Generating prompts for the videos...")
                async for event in self._run_sub_agent(
                    self.video_prompt_generator, ctx
                ):
                    yield event

                # 7. Voiceover generation
                yield text2event(self.name, "Generating voiceovers...")
                async for event in self._run_sub_agent(self.voiceover_generator, ctx):
                    yield event

                # 8. Image generation
                yield text2event(self.name, "Generating images...")
                async for event in self._run_sub_agent(self.image_generator, ctx):
                    yield event

                # 9. Video generation
                yield text2event(self.name, "Generating videos...")
                async for event in self._run_sub_agent(self.video_generator, ctx):
                    yield event

                # 10. Video assembling
                yield text2event(self.name, "Assembling final video...")
                async for event in self._run_sub_agent(self.video_assembler, ctx):
                    yield event

                # 11. SEO Optimization
                yield text2event(
                    self.name, "Optimizing video title and description for SEO..."
                )
                async for event in self._run_sub_agent(self.seo_optimizer, ctx):
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
    theme_definer=theme_definer_agent,
    user_feedback=user_feedback_agent,
    expert_researcher=expert_researcher_agent,
    web_researcher=web_researcher_agent,
    research_compiler=research_compiler_agent,
    script_writer=script_writer_agent,
    scene_breakdown=scene_breakdown_agent,
    image_prompt_generator=image_prompt_generator_agent,
    video_prompt_generator=video_prompt_generator_agent,
    voiceover_generator=voiceover_generator_agent,
    image_generator=image_generator_agent,
    video_generator=video_generator_agent,
    video_assembler=video_assembler_agent,
    seo_optimizer=seo_optimizer_agent,
)

root_agent = video_creator_workflow_agent
