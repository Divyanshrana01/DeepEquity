# One image, three roles. The api, worker, and mcp-server containers all
# run from this same image, docker-compose just overrides the command for
# each. They share the same dependencies and codebase, so building three
# separate images would just mean rebuilding the same layers three times.

FROM python:3.12-slim AS base

# Install uv by copying the prebuilt binary from its official image, faster
# and more reliable than pip-installing it.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so `uv sync` only re-runs when pyproject.toml or the
# lockfile actually change, not on every code edit.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Download the embedding model at build time instead of on first use. Without this the
# worker would stall for however long a 130MB download takes the first time it processes
# a document, and a container with no internet access would never work at all.
ENV FASTEMBED_CACHE_PATH=/app/.model_cache
RUN python -c \
    "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY README.md ./
COPY src ./src
COPY mcp_server ./mcp_server
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uvicorn", "deepequity.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
