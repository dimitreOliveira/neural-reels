from google.adk.agents import Agent

from researcher_agent.callbacks.callbacks import save_agent_output

MODEL_ID = "gemini-2.5-flash"

SCRIPT_WRITER_PROMPT = """
Based on the user's input and intent, write a short content video script about the theme "{theme}".
The script should be engaging, concise, and suitable for a 1~2 minutes video.
This script must contain only the text narration.
The script must contain only the text and no text artifacts.

Here is the user's intent regarding the script: "{intent}".

You should base the script on the information below gathered and compiled by the researchers:
# Researched information

{compiled_research_report?}

You may need to iterate over this script, below you can find the current version, if it exsits:
# Current script

{current_script?}

Here you can see the user's feedback, if it exsits, that must be incorporated for the next iteration:
# User's feedbacl

{user_input?}
"""


script_writer_agent = Agent(
    name="ScriptWriterAgent",
    description="Generates a script for a short video content.",
    instruction=SCRIPT_WRITER_PROMPT,
    model=MODEL_ID,
    output_key="script",
    after_agent_callback=save_agent_output,
    include_contents="none",
)
