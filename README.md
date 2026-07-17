# DeepEquity

Multi-agent equity research desk. Give it a ticker, a bull agent and a bear agent argue opposite
sides of the thesis using the same evidence, and a synthesis agent writes a cited research note
with a confidence breakdown.

Full architecture and reasoning: [equity-research-desk-plan_2.md](equity-research-desk-plan_2.md).
Plain-language change log: [PROGRESS.md](PROGRESS.md).

## Status

Phase 1 (skeleton and infrastructure) in progress. `docker compose up` gives a running,
authenticated, rate-limited API that doesn't do anything useful yet, that's the point of this
phase, the agents and retrieval come in Phase 2 and 3.

## Local setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and virtualenv management.

```bash
uv sync                 # creates .venv and installs everything, including dev tools
cp .env.example .env    # fill in secrets before running anything for real
uv run uvicorn deepequity.api.main:app --reload
```

## Running with Docker Compose

```bash
docker compose up --build
```

Spins up: `api` (FastAPI), `worker`, `postgres` (with the `pgvector` extension), `redis`, and
`mcp-server` (currently a placeholder, real tools land in Phase 2).

## Tests

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
