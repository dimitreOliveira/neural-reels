from google.adk.agents import Agent
from pydantic import BaseModel, Field

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-flash"

IMAGE_PROMPT_GENERATOR_PROMPT = """
Based on the following scenes, generate an image generation prompt for each scene.
Each prompt should be highly descriptive, focusing on visual elements, style, and mood suitable for the scene's text.
Ensure the prompts are optimized for an AI image generation model like Imagen.
Create only a single image prompt for each scene.


# Output format
[
"Image prompt for Scene 1 goes here",
"Image prompt for Scene 2 goes here",
"Image prompt for Scene 3 goes here",
...
"Image prompt for Scene N goes here"
]


# List with the scenes from the script:
{scenes}
"""


class ImagePromptGeneratorOutput(BaseModel):
    image_prompts: list[str] = Field(
        description="The image generation prompts used to generate the images for the short video content."
    )


image_prompt_generator_agent = Agent(
    name="ImagePromptGeneratorAgent",
    description="Generates image prompts to illustrate a script.",
    instruction=IMAGE_PROMPT_GENERATOR_PROMPT,
    model=MODEL_ID,
    include_contents="none",
    output_key="image_prompts",
    output_schema=ImagePromptGeneratorOutput,
    after_agent_callback=save_agent_output,
)
