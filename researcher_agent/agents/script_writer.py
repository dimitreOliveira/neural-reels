from google.adk.agents import Agent
from pydantic import BaseModel, Field

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-flash"

SCRIPT_WRITER_PROMPT = """
Based on the user's input and the provided theme '{theme}', and , write a short content video script.
The script should be engaging, concise, and suitable for a 1~2 minutes video.
This script must contain only the text narration.

The script must contain only the text and no text artifacts.

If the user approves it or if you don't know how to handle the request handle it back the to the orchestrator [OrchestratorAgent],
other wise iterate with the user to update the result.

The script must have a max of 100 chars.
"""


class ScriptOutput(BaseModel):
    script: str = Field(description="The script of the short video content.")


script_writer_agent = Agent(
    name="ScriptWriterAgent",
    description="Generates a script for a short video content.",
    instruction=SCRIPT_WRITER_PROMPT,
    model=MODEL_ID,
    output_key="script",
    output_schema=ScriptOutput,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    after_agent_callback=save_agent_output,
)
