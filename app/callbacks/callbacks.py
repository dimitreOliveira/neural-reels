import json
import logging
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_agent_output(callback_context: CallbackContext) -> Optional[types.Content]:
    """Saves the output of an agent to a file.

    This callback function is triggered after an agent runs. It takes the
    current session state, determines the output directory, and saves the
    state's contents to files. Dictionaries and Pydantic models are saved as
    JSON, while other values are saved as Markdown files.

    Args:
        callback_context: The context object provided by the ADK, containing
            the current session state.

    Returns:
        None. This callback does not modify the agent's response.
    """
    current_state = callback_context.state.to_dict()
    output_dir = Path(current_state.get("assets_path", "projects/default"))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Persist the current state
    for key, value in current_state.items():
        # Save json output
        if isinstance(value, dict) or isinstance(value, BaseModel):
            if isinstance(value, BaseModel):
                value = value.model_dump()
            response_filename = output_dir / f"{key}.json"
            with open(response_filename, "w") as file:
                json.dump(value, file, indent=4)
        # Save text output
        elif isinstance(value, dict):
            response_filename = output_dir / f"{key}.md"
            response_filename.write_text(value)
        else:
            logger.info(f"Could no save state value {key} of type '{type(value)}'")

    return None
