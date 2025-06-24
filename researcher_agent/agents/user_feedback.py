from google.adk.agents import Agent
from pydantic import BaseModel, Field

MODEL_ID = "gemini-2.5-flash"

USER_FEEDBACK_PROMPT = """
Your task is to understand and parse the feedback from the user.
You should parse the user's input and understand if the current state was approved or not.
If the user provided some feedback parse and digest it into actionable steps.

# Output format
If the user indicated in anyway that the current state was approved just output 'approved'.
If not the above and the user provided feedback parse and digest it into actionable steps.
If neither options above and the user indicated that it was not approved or for any other case, just output 'not approved'
"""


class UserFeedbackAgentOutput(BaseModel):
    feedback: str = Field(
        description="The feedback of the user about the current state."
    )


user_feedback_agent = Agent(
    name="UserFeedbackAgent",
    description="Understands and parses the feedback from the user about the current state.",
    instruction=USER_FEEDBACK_PROMPT,
    model=MODEL_ID,
    output_key="feedback",
    output_schema=UserFeedbackAgentOutput,
)
