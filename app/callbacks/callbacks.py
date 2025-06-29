import json
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from pydantic import BaseModel


def save_agent_output(callback_context: CallbackContext) -> Optional[types.Content]:
    current_state = callback_context.state.to_dict()
    output_dir = Path(current_state.get("assets_path", "projects/default"))
    output_dir.mkdir(exist_ok=True, parents=True)

    # Persist the current state
    for key, value in current_state.items():
        # Save json output
        if isinstance(value, dict):
            response_filename = output_dir / f"{key}.json"
            with open(response_filename, "w") as file:
                json.dump(value, file, indent=4)
        # Save json output
        elif isinstance(value, BaseModel):
            value = value.model_dump()
            response_filename = output_dir / f"{key}.json"
            with open(response_filename, "w") as file:
                json.dump(value, file, indent=4)
        # Save text output
        else:
            response_filename = output_dir / f"{key}.md"
            response_filename.write_text(value)

    return None
