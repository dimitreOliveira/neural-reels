from google.adk.agents import Agent

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-flash"

RESEARCH_COMPILER_PROMPT = """
Given the report and resources gathered below, synthesize a concise research brief that later will be used to create a YouTube Short video.
This report should highligh key facts, interesting points, curiosities, and any potential angles for a short, engaging video.
To create the report take into accout the user's intent, to make sure that you also focus on keeping information relevant to that.
Here is the user's intent regarding the report: "{intent}".

# Report compiled by the expert agent:
{expert_researcher_report}

# Resources gathered by the web search agent:
{web_researcher_report}
"""

research_compiler_agent = Agent(
    name="ResearchCompilerAgent",
    description="Compiles the results of a research into a report for a given theme.",
    instruction=RESEARCH_COMPILER_PROMPT,
    model=MODEL_ID,
    output_key="compiled_research_report",
    after_agent_callback=save_agent_output,
    include_contents="none",
)
