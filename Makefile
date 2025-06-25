app:
	adk web

app_cli:
	adk run researcher_agent

lint:
	ruff check --select I --fix
	ruff check
	ruff format

# build:
# 	uv pip install -r requirements.txt