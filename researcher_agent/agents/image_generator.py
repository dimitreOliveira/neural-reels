import logging
from pathlib import Path
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from pydantic import Field
from typing_extensions import override

from researcher_agent.utils.genai_utils import get_client, text2event
from researcher_agent.utils.image_utils import save_image_from_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_ID = "models/imagen-3.0-generate-002"
ASPECT_RATIO = "9:16"


class ImagenAgent(BaseAgent):
    """
    An ADK Custom Agent that generates an image using Imagen
    based on a prompt and saves it to a file.
    """

    client: genai.Client = None
    image_gen_config: dict = None

    # --- Pydantic Fields for Agent Configuration ---
    name: str = Field(
        default="ImageGeneratorAgent", description="The name of the agent."
    )
    description: str = Field(
        default="Generates images from text prompts using Imagen.",
        description="The description of the agent.",
    )
    input_key: str = Field(
        default="image_prompts",
        description="The key in the session state holding the text prompt for image generation.",
    )
    output_key: str = Field(
        default="images_path",
        description="The key in the session state to store the path of the directory where generated images files are saved.",
    )
    output_subdir: str = Field(
        default="images",
        description="The folder to save the generated image file.",
    )
    # --- Imagen-specific configuration ---
    model: str = Field(
        default=MODEL_ID,
        description="The Imagen model to use for image generation.",
    )
    images_per_prompt: int = Field(
        default=1, description="Number of images to generate for each prompt."
    )
    output_mime_type: str = Field(
        default="image/jpeg", description="MIME type for the output image."
    )
    person_generation: str = Field(
        default="ALLOW_ADULT", description="Policy for generating people."
    )
    aspect_ratio: str = Field(
        default=ASPECT_RATIO,
        description="Aspect ratio of the generated image (e.g., '1:1', '16:9').",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = get_client()
        self.image_gen_config = dict(
            number_of_images=self.images_per_prompt,
            output_mime_type=self.output_mime_type,
            person_generation=self.person_generation,
            aspect_ratio=self.aspect_ratio,
        )

    async def _generate_image(
        self, scene_idx: int, prompt: str, output_dir: Path
    ) -> AsyncGenerator[Event, None]:
        """
        Generates an image for a single scene/prompt, yields status events,
        and saves the image.
        """
        logger.info(
            f"[{self.name}] Generating image for scene {scene_idx + 1} with prompt: '{prompt[:70]}...'"
        )
        yield text2event(self.name, f"Generating image for scene {scene_idx + 1}...")
        result = self.client.models.generate_images(
            model=self.model, prompt=prompt, config=self.image_gen_config
        )

        if not result.generated_images:
            error_msg = f"Image generation failed for scene {scene_idx + 1}. The API returned no images."
            logger.error(f"[{self.name}] {error_msg}")
            yield text2event(self.name, error_msg)
            return

        for image_idx, generated_image in enumerate(result.generated_images):
            image_bytes = generated_image.image.image_bytes
            img_filename = f"image_{image_idx + 1}.jpg"
            output_filepath = output_dir / img_filename
            save_image_from_bytes(image_bytes, output_filepath)

            logger.info(
                f"[{self.name}] Image for scene {scene_idx + 1}, variant {image_idx + 1} saved to '{output_filepath}'"
            )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        The core implementation of the agent's logic.
        """
        logger.info(f"[{self.name}] Starting image generation.")

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
            f"[{self.name}] Received {len(prompts)} prompt(s) for image generation."
        )
        logger.info(f"[{self.name}] Calling Imagen API with model '{self.model}'...")

        try:
            ctx.session.state[self.output_key] = []
            for scene_idx, prompt in enumerate(prompts):
                output_dir = assets_path / f"scene_{scene_idx + 1}" / self.output_subdir
                output_dir.mkdir(parents=True, exist_ok=True)
                ctx.session.state[self.output_key].append(str(output_dir))

                async for event in self._generate_image(scene_idx, prompt, output_dir):
                    yield event

            final_message = f"Images generated and saved to '{output_dir}'."
            yield text2event(self.name, final_message)

        except Exception as e:
            error_msg = f"An error occurred during image generation: {e}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            yield text2event(self.name, error_msg)


image_generator_agent = ImagenAgent(
    name="ImageGeneratorAgent",
    description="Generates images from a text prompts.",
    input_key="image_prompts",
    output_subdir="images",
    output_key="images_path",
    model=MODEL_ID,
    aspect_ratio=ASPECT_RATIO,
)
