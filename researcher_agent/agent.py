import logging

from dotenv import load_dotenv
from google.adk.agents import Agent

from researcher_agent.agents.tts_gemini_agent import TtsGeminiAgent
from researcher_agent.callbacks.callbacks import save_agent_output
from researcher_agent.genai_utils import get_client

MODEL_ID = "gemini-2.5-flash-preview-05-20"
TTS_MODEL_ID = "gemini-2.5-flash-preview-tts"
VOICE_NAME = "Algenib"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = get_client()


SCRIPT_WRITER_PROMPT = """
Based on the provided input, write a short content video script.
The script should be engaging, concise, and suitable for a 1~2 minute video.
This script must contain only the text narration.
"""

ORCHESTRATOR_PROMPT = """
You are an orchestrator responsible for managing a team of agents with the goal of creating a short content video based on the user's request.

# Your task

1. Ask the user to provide an input that will be used as the topic/theme to create the short content.
2. Prompt the script writer to write an script based on this input.
3. Ask the user if the current script is satisfactory.
4. If the current script was not accepted ask the script writer to rewirte it based on the user's feedback.
5. If the current script was accepted request the TTS agent (text-to-speech agent) to generate an audio of it.

You may handle any other request or interactions with the user.
"""

script_writer_agent = Agent(
    name="ScriptWriterAgent",
    description="Generates a script for a short video content.",
    instruction=SCRIPT_WRITER_PROMPT,
    model=MODEL_ID,
    output_key="script",
    after_agent_callback=save_agent_output,
)

tts_gemini_agent = TtsGeminiAgent(
    name="TtsGeminiAgent",
    description="Generates a voiceover audio from text.",
    model=TTS_MODEL_ID,
    voice_name=VOICE_NAME,
    input_key="script",
    output_key="voiceover",
    output_filename="voiceover.wav",
    client=client,
)

orchestrator_agent = Agent(
    name="OrchestratorAgent",
    instruction=ORCHESTRATOR_PROMPT,
    model=MODEL_ID,
    output_key="orchestrator",
    after_agent_callback=save_agent_output,
    sub_agents=[
        script_writer_agent,
        tts_gemini_agent,
    ],
)

root_agent = orchestrator_agent
