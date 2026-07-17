# Progress Log

Plain-language log of what got built and why. Updated every time we make a change.
Full architecture reasoning lives in [equity-research-desk-plan_2.md](equity-research-desk-plan_2.md),
this file is just "what happened, in order."

---

## 2026-07-17: Phase 1 skeleton (infra, no real features yet)

**What we built**

- Set up the project with `uv` (a fast Python package manager) instead of plain pip. Pinned
  Python to 3.12. `uv sync` creates a `.venv` folder and installs everything from `pyproject.toml`.
- Laid out the code in `src/deepequity/`:
  - `core/` — shared stuff every part of the app needs: settings (reads from `.env`), logging
    setup, and one shared Redis connection.
  - `api/` — the FastAPI web server. Has:
    - `/health` — just says "the process is alive," used by Docker to know if it should restart
      a crashed container.
    - `/ready` — actually checks Redis and Postgres are reachable, so traffic doesn't get sent
      here until the app can really do its job.
    - JWT auth — a way to check that whoever's calling the API has a valid token. No login
      endpoint yet, that comes later when there are real user accounts.
    - Rate limiting — stops one caller from hammering the API. Uses Redis to count requests in
      a rolling time window (not just "reset every minute," which can be gamed by bursting right
      at the reset).
    - Request logging — every request gets a unique ID, and every log line about that request
      carries the ID, so if something breaks in production we can find every log line that
      belongs to one specific failed request.
  - `worker/` — a separate background process. Right now it just says "I'm alive" every 30
    seconds. Later it'll do the actual work of pulling filings and processing them.
- `mcp_server/` — a placeholder for the tool server the AI agents will call later (fetch a
  filing, fetch a transcript, etc). Just a health check for now.
- Wrote tests for the auth, health checks, and rate limiter. Tests use a fake in-memory Redis
  so they run fast and don't need a real Redis server.
- `docker-compose.yml` — one command (`docker compose up`) starts everything: the API, the
  worker, the mcp-server, a Postgres database (with the pgvector extension for storing
  embeddings later), and Redis.
- GitHub Actions CI — automatically runs the linter, type checker, and tests on every push.

**Verified it actually works**

- Ran the linter (ruff) and type checker (mypy): both clean.
- Ran all 8 tests: all passing.
- Booted the app locally and hit `/health`: got a real response with proper logging.
- Ran the full Docker stack (`docker compose up --build`): all 5 containers came up healthy,
  hit `/health` and `/ready` on the real running containers, both good.

**One bug found and fixed along the way**

The Docker build failed the first time. The project's `pyproject.toml` points to `README.md`
for its description, but the Dockerfile never copied that file into the image, so the build
step that installs the project couldn't find it. Fixed by adding `COPY README.md ./` before the
install step in `docker/app.Dockerfile`.

**What's NOT here yet (on purpose)**

No filings ingestion, no retrieval, no agents, no actual research happening. This phase is just
the plumbing: a working, authenticated, rate-limited, logged API that doesn't do anything useful
yet. That's the whole point of Phase 1, everything after this builds on top of infra we know
already works.
