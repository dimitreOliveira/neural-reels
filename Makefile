app:
	adk web

app_cli:
	adk run app

backend:
	uv run adk api_server --allow_origins="*"

install:
	uv sync --frozen

lint:
	uv run codespell
	uv run ruff check --select I --fix
	uv run ruff check
	uv run ruff format
	uv run mypy