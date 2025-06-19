from google.adk.agents import Agent
from pydantic import BaseModel, Field

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-flash"

VIDEO_PROMPT_GENERATOR_PROMPT = """
Based on the following scenes, generate a video generation prompt for each scene.
Each prompt should be highly descriptive, focusing on visual elements, style, and mood suitable for the scene's text.
Ensure the prompts are optimized for an AI video generation model like Veo.
Create only a single video prompt for each scene.


# Output format
[
"Video prompt for Scene 1 goes here",
"Video prompt for Scene 2 goes here",
"Video prompt for Scene 3 goes here",
...
"Video prompt for Scene N goes here"
]


# List with the scenes from the script:
{scenes}
"""


class VideoPromptsOutput(BaseModel):
    video_prompts: list[str] = Field(
        description="The video generation prompts used to generate the videos for the short video content."
    )


video_prompt_generator_agent = Agent(
    name="VideoPromptGeneratorAgent",
    description="Generates video prompts to illustrate a script.",
    instruction=VIDEO_PROMPT_GENERATOR_PROMPT,
    model=MODEL_ID,
    include_contents="none",
    output_key="video_prompts",
    output_schema=VideoPromptsOutput,
    after_agent_callback=save_agent_output,
)
