from google.adk.agents import Agent
from pydantic import BaseModel, Field

MODEL_ID = "gemini-2.5-flash"

THEME_DEFINER_PROMPT = """
# Your role

You are responsible for defining the theme of the short video content with the user.

# Your task

Talk with the user to define the topic/theme that will be used to create the short video content.
Once the user provides the topic summarize it into 1 ~ 3 words.
You may handle any other request or interactions with the user but make sure to guide them through those steps and define the topic.

# Examples
- User input: Lets create a video about Machine Learning
- Theme: Machine Learning

- User input: Gemini models
- Theme: Gemini models

- User input: I would like to create a video on the context of Python programming language to help students learning the language.
- Theme: Python programming
"""


class ThemeOutput(BaseModel):
    theme: str = Field(
        description="The theme of the short video content provided by the user."
    )


theme_definer_agent = Agent(
    name="ThemeDefinerAgent",
    description="Defines the theme of the short video content with the user.",
    instruction=THEME_DEFINER_PROMPT,
    model=MODEL_ID,
    output_key="theme",
    output_schema=ThemeOutput,
)
