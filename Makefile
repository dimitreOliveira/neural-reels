app:
	adk web

# build:
# 	uv pip install -r requirements.txt

lint:
	ruff check --select I --fix
	ruff format