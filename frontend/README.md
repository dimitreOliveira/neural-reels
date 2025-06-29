# Neural Reels Streamlit Frontend

This directory contains the Streamlit-based user interface for the Neural Reels application.

## Prerequisites

- Python and `uv` installed.
- All dependencies from the root `pyproject.toml` installed (`uv sync --frozen`).
- The ADK backend server must be running.

## How to Run

1.  **Start the Backend Server:**

    From the root directory of the `neural-reels` project, run the backend server:

    ```bash
    make backend
    ```

    This will start the ADK API server, typically on `http://localhost:8000`.

2.  **Run the Frontend Application:**

    In a separate terminal, from the root directory of the `neural-reels` project, run the Streamlit app:

    ```bash
    make frontend
    ```

3.  **Interact with the UI:**

    Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`). You can now interact with the Neural Reels agent through the chat interface.