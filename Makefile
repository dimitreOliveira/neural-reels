app_web:
	uv run adk web

app_cli:
	uv run adk run app

dev-app:
	make dev-backend & make dev-frontend

dev-backend:
	uv run adk api_server --allow_origins="*"

dev-frontend:
	uv run streamlit run frontend/neural_reels_app.py

install:
	uv sync --frozen

lint:
	uv run codespell app
	uv run codespell frontend
	uv run ruff check . --select I --fix
	uv run ruff check .
	uv run ruff format .
	uv run mypy .