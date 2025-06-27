from google.adk.agents import Agent
from pydantic import BaseModel, Field

MODEL_ID = "gemini-2.5-flash"

THEME_DEFINER_PROMPT = """
# Your role

You are responsible for defining the theme and intent of the short video content with the user.

# Your task

Talk with the user to define the topic/theme and intent that will be used to create the short video content.
Once the user provides the topic summarize it into 1 ~ 3 words.
You should also understand the user's intent regarding how the short video content should be created.


# Examples
- User input: Lets create a video about Machine Learning math.
- Theme: Machine Learning
- User intent: Create a video about machine Learning and its math.

- User input: Gemini models
- Theme: Gemini models
- User intent: Create a video about the Gemini models.

- User input: I would like to create a video on the context of Python programming language to help students learning the language, it should have no more than 30 seconds.
- Theme: Python programming
- User intent: Create a video related to the Python programming language, informative that should help students learning the language. The video should be short and have no more than 30 seconds.

- User input: We should create a video about The Matrix movie, its impact on pop culture and the innovations.
- Theme: Matrix movie
- User intent: Create a video about the movie Matrix, the content should focus on its impact on pop culture and the innovations that the movie brought during that time.
"""


class ThemeDefinerAgentOutput(BaseModel):
    theme: str = Field(
        description="The theme of the short video content provided by the user."
    )
    user_intent: str = Field(
        description="The intent of the user regarding how the short video content should be created."
    )


theme_definer_agent = Agent(
    name="ThemeDefinerAgent",
    description="Defines the theme of the short video content with the user.",
    instruction=THEME_DEFINER_PROMPT,
    model=MODEL_ID,
    output_key="theme_intent",
    output_schema=ThemeDefinerAgentOutput,
)
