"""
Neural Reels Streamlit Frontend
===============================

This Streamlit application provides a user-friendly chat interface for interacting
with the Neural Reels AI agent. It connects to the backend ADK API server to
process user requests and display the generated results.

Requirements:
------------
- The ADK backend server must be running (e.g., `make dev-backend`).
- All project dependencies must be installed (`uv sync --frozen`).

Usage:
------
1. Start the backend server: `make dev-backend`
2. In a new terminal, run this app: `make dev-frontend`
3. Open the provided URL in your browser to start creating!

"""

import logging
import time
import uuid

import requests
import streamlit as st

# Set up basic logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Neural Reels",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Constants ---
API_BASE_URL = "http://localhost:8000"
APP_NAME = "app"  # Based on `adk run app` in Makefile.mk

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{uuid.uuid4()}"

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def create_session():
    """Create a new session with the ADK agent."""
    session_id = f"session-{int(time.time())}"
    try:
        response = requests.post(
            f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions/{session_id}",
            headers={"Content-Type": "application/json"},
            json={},
            timeout=10,
        )
        response.raise_for_status()

        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.success(f"New session created: {session_id}")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create session: {e}. Is the backend running?")
        return False


def send_message(message: str):
    """Send a message to the agent and handle the response."""
    if not st.session_state.session_id:
        st.error("No active session. Please create a session first.")
        return

    st.session_state.messages.append({"role": "user", "content": message})

    try:
        with st.spinner("The AI is generating your reel..."):
            response = requests.post(
                f"{API_BASE_URL}/run",
                headers={"Content-Type": "application/json"},
                json={
                    "app_name": APP_NAME,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                    "new_message": {"role": "user", "parts": [{"text": message}]},
                },
                timeout=3600,  # Increased timeout for potentially long video generation
            )
            response.raise_for_status()

        events = response.json()
        assistant_messages_added = 0
        for event in events:
            content = event.get("content")

            # Skip user messages to avoid echoing them back to the user.
            if content and content.get("role") == "user":
                continue

            # Process and display text from any part of the event content.
            if content and content.get("parts"):
                for part in content.get("parts", []):
                    if "text" in part:
                        text_content = part["text"].strip()
                        if text_content:
                            st.session_state.messages.append(
                                {"role": "assistant", "content": text_content}
                            )
                            assistant_messages_added += 1

        if assistant_messages_added == 0 and events:
            assistant_message = "The agent responded, but no text message was found. See raw response below."
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message}
            )
            st.json(events)

    except requests.exceptions.RequestException as e:
        st.error(
            f"Network or HTTP error: {e}. Is the backend running on {API_BASE_URL}?"
        )
        logger.error(f"RequestException sending message: {e}", exc_info=True)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error in send_message: {e}", exc_info=True)


# --- UI Rendering ---
st.title("🎥 Neural Reels")
st.caption("AI-Powered Short-Form Video Asset Generator")

with st.sidebar:
    st.header("Controls")
    if st.button("➕ New Session"):
        create_session()
        st.rerun()
    st.info(f"Session ID: {st.session_state.session_id or 'None'}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.session_id:
    if user_input := st.chat_input("Describe the video you want to create..."):
        send_message(user_input)
        st.rerun()
else:
    st.info("👈 Create a new session to begin.")
