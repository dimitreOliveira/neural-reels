import json
import os
from pathlib import Path
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types


def save_agent_output(callback_context: CallbackContext) -> Optional[types.Content]:
    current_state = callback_context.state.to_dict()
    PROJECT_NAME = os.environ.get("PROJECT_NAME")
    if not PROJECT_NAME:
        # Log an error or raise an exception, as this callback might not be able to yield Events.
        # For example, log and return, or raise a ValueError.
        print(f"Error: PROJECT_NAME environment variable not set. Cannot save agent output.") # Or use logger
        # Depending on ADK capabilities, you might not be able to yield an error event here.
        # Consider how to signal this failure. For now, returning None as per original signature.
        return None

    output_key = list(current_state.keys())[-1]
    agent_response = current_state[output_key]
    response_dir = Path(f"projects/{PROJECT_NAME}".lower().replace(" ", "_"))
    response_dir.mkdir(exist_ok=True, parents=True)

    # print("\n","="*50,"\n")
    # print(f"\t[After Agent Callback] Current State: {json.dumps(current_state, indent=4)}")
    # print("\n","="*50,"\n")
    # print(f"\t[After Agent Callback] current_state keys: {list(current_state.keys())}")
    # print("\n","="*50,"\n")
    # print(f"\t[After Agent Callback] callback_context.user_content: {callback_context.user_content.parts[0].text}")
    # print("\n","="*50,"\n")

    if isinstance(agent_response, dict):
        response_filename = response_dir / f"{output_key}.json"
        with open(response_filename, "w") as file:
            json.dump(agent_response, file, indent=4)
    else:
        response_filename = response_dir / f"{output_key}.md"
        response_filename.write_text(agent_response)

    return None


# def _render_reference(
#     callback_context: CallbackContext,
#     llm_response: LlmResponse,
# ) -> LlmResponse:
#     """Appends grounding references to the response."""
#     del callback_context
#     if (
#         not llm_response.content
#         or not llm_response.content.parts
#         or not llm_response.grounding_metadata
#     ):
#         return llm_response
#     references = []
#     for chunk in llm_response.grounding_metadata.grounding_chunks or []:
#         title, uri, text = "", "", ""
#         if chunk.retrieved_context:
#             title = chunk.retrieved_context.title
#             uri = chunk.retrieved_context.uri
#             text = chunk.retrieved_context.text
#         elif chunk.web:
#             title = chunk.web.title
#             uri = chunk.web.uri
#         parts = [s for s in (title, text) if s]
#         if uri and parts:
#             parts[0] = f"[{parts[0]}]({uri})"
#         if parts:
#             references.append("* " + ": ".join(parts) + "\n")
#     if references:
#         reference_text = "".join(["\n\nReference:\n\n"] + references)
#         llm_response.content.parts.append(types.Part(text=reference_text))
#     if all(part.text is not None for part in llm_response.content.parts):
#         all_text = "\n".join(part.text for part in llm_response.content.parts)
#         llm_response.content.parts[0].text = all_text
#         del llm_response.content.parts[1:]
#     return llm_response
