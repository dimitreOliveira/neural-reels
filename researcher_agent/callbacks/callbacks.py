import json
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types


def save_agent_output(callback_context: CallbackContext) -> Optional[types.Content]:
    current_state = callback_context.state.to_dict()
    output_dir = Path(current_state.get("assets_path", "test"))
    output_dir.mkdir(exist_ok=True, parents=True)

    output_key = list(current_state.keys())[-1]
    agent_response = current_state[output_key]

    # Save json output
    if isinstance(agent_response, dict):
        response_filename = output_dir / f"{output_key}.json"
        with open(response_filename, "w") as file:
            json.dump(agent_response, file, indent=4)
    # Save text output
    else:
        response_filename = output_dir / f"{output_key}.md"
        response_filename.write_text(agent_response)

    return None
