from google.adk.agents import Agent

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-pro"

EXPERT_RESEARCHER_PROMPT = """
You are an expert researcher in many different topics, excelling at compiling and presenting relevant information.

# Your task

Create a short report for the theme given by the user: "{theme}".
Here is the user's intent regarding the report: "{intent}".

# Output format

Your output should be a Markdown-formatted text.
"""

expert_researcher_agent = Agent(
    name="ExpertResearcherAgent",
    description="Generates a report for a given theme.",
    instruction=EXPERT_RESEARCHER_PROMPT,
    model=MODEL_ID,
    output_key="expert_researcher_report",
    after_agent_callback=save_agent_output,
    include_contents="none",
)
