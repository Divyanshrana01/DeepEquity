from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


#all the settings the app needs, pulled from environment variables. using pydantic-settings
#means we get validation for free: if redis_url is missing or malformed, the app fails fast
#at startup instead of crashing later when something actually tries to use redis.
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    app_name: str = "deepequity"
    environment: str = "development"
    log_level: str = "INFO"

    # Postgres
    postgres_dsn: str = "postgresql://deepequity:deepequity@localhost:5432/deepequity"

    # Redis (rate limiting, caching, checkpoints)
    redis_url: str = "redis://localhost:6379/0"
    # How long a single read from redis may take. This MUST stay comfortably above
    # ingestion_block_ms below, because the worker's blocking stream read holds the
    # socket open for that long on purpose. redis-py quietly defaults this to 5s, which
    # is the same as our block duration, so without setting it here the worker's very
    # first idle read times out and kills the process.
    redis_socket_timeout: float = 30.0

    # JWT auth
    jwt_secret_key: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate limiting: N requests per window_seconds, per API key/IP
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # MCP server host/port (the container listens here for tool calls)
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    # Where the api and worker reach the MCP server. Inside docker that's the service
    # name, locally it's localhost on the mapped port.
    mcp_server_url: str = "http://localhost:8001/mcp"

    # The MCP SDK blocks requests whose Host header it doesn't recognise, which is what
    # stops a malicious webpage from pointing your browser at a local MCP server (DNS
    # rebinding). We keep that protection on and just name the hosts we actually serve:
    # the docker service name for container-to-container calls, plus localhost for
    # running things by hand. Comma separated so it can be set from one env var.
    mcp_allowed_hosts: str = "mcp-server:8000,localhost:8001,127.0.0.1:8001"

    # Ingestion retry policy. Transient failures get retried with an exponential backoff
    # (1s, 4s, 16s with a base of 4), anything left over after this many tries goes to
    # the dead letter queue.
    ingestion_max_attempts: int = 3
    ingestion_backoff_base_seconds: float = 4.0
    # How long the worker parks on an empty queue before looping again. Keep this well
    # under redis_socket_timeout, see the note there.
    ingestion_block_ms: int = 5000
    # How long a message must sit untouched before another worker is allowed to take it
    # over. This is the crash recovery window: too short and we'd steal work from a
    # worker that's just being slow, too long and a dead worker's documents sit idle.
    ingestion_reclaim_idle_ms: int = 60_000

    # Embeddings. fastembed runs the model on onnxruntime, so no pytorch in the image.
    # If you change the model you must change embedding_dim to match and rebuild the
    # chunks table, the vector column size is fixed at creation.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # How many chunks to embed in one call. Bigger batches are faster but use more memory.
    embedding_batch_size: int = 64

    # Parent-child chunking. Children are what we search over (small, so a match is
    # precise), parents are what we hand the LLM (big, so it gets enough context to
    # actually use the match).
    child_chunk_chars: int = 400
    child_chunk_overlap_chars: int = 80
    parent_chunk_chars: int = 2000

    # Retrieval. We pull a wide net of candidates from each search method, fuse them,
    # then let the reranker pick the final few. Candidates need to be comfortably bigger
    # than the final count or the reranker has nothing to improve on.
    retrieval_candidates_per_method: int = 30
    retrieval_final_top_k: int = 5
    # Reciprocal Rank Fusion constant. 60 is the value from the original paper, it
    # softens the gap between rank 1 and rank 2 so a single method can't dominate purely
    # by being confident.
    rrf_k: int = 60
    # Cross-encoder that rescores the fused candidates. Slower than the embedding model
    # because it reads the query and passage together, which is exactly why it's more
    # accurate and why we only run it on a shortlist.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # LLM. Groq's free tier, which is an OpenAI-compatible API over open models.
    groq_api_key: str = ""
    # Two tiers, which is the model routing the plan calls for: the small one handles
    # scoping and extraction where the job is mostly following instructions, the big one
    # handles synthesis where the reasoning actually matters and a mistake is expensive.
    # Both of these support strict json schema output, which the agents rely on.
    llm_fast_model: str = "openai/gpt-oss-20b"
    llm_strong_model: str = "openai/gpt-oss-120b"
    # Which agent roles get the strong model. A setting rather than a hardcoded list
    # because "does the cheap model hold up on this job" is a measurable experiment, and
    # it should be a config change plus an eval rerun, not a code edit.
    llm_strong_roles: str = "synthesis"
    # Low but not zero. Zero makes the debate agents repetitive, high makes them invent.
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 4096
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3

    # How much evidence each debate agent is given.
    #
    # Two reasons this is capped rather than "send everything we found". The practical
    # one: Groq's free tier allows 8000 tokens per minute, and bull and bear both fire in
    # the same minute, so an unbounded evidence pool gets the request rejected outright.
    # The better one: past a dozen or so passages the arguments get worse, not better,
    # because the passages that actually matter are diluted by ones that merely matched.
    # We keep the highest scoring ones.
    max_evidence_chunks: int = 8
    # Whether bull and bear argue at the same time.
    #
    # Concurrent is faster and was the original design, but each request runs to roughly
    # 5000 tokens and the free tier allows 8000 per minute, so firing both together is
    # over the limit before either finishes and both get rejected. Sequential fits, and a
    # rate limited request that succeeds beats two parallel ones that fail. Worth turning
    # back on with a paid tier.
    debate_concurrent: bool = False
    # Each passage is trimmed to this. Parent chunks run to 2000 characters and the tail
    # end is usually the least relevant part, since the match was nearer the start.
    max_evidence_chars_per_chunk: int = 1100

    # Agent graph limits. These are what stop a debate running away, both in time and in
    # money, and they're the answer to "how do you stop an agent looping forever".
    max_debate_rounds: int = 2
    # Hard ceiling per research run. When it's hit the graph stops and returns what it
    # has rather than continuing to spend.
    max_tokens_per_run: int = 120_000

    # Semantic cache. Answers are keyed by the meaning of the prompt rather than its exact
    # text, so a repeat run on the same ticker reuses work instead of paying for it again.
    semantic_cache_enabled: bool = True
    # How alike two prompts must be to count as the same question. Set high on purpose:
    # the bull and bear prompts for one company are built from identical evidence and sit
    # around 0.95 similar, so anything looser would serve one agent the other's answer.
    # Missing a real hit costs a few cents, returning the wrong note costs trust.
    semantic_cache_threshold: float = 0.97
    semantic_cache_ttl_seconds: int = 7 * 24 * 60 * 60
    # Cap on entries kept per namespace, and on how many we compare against per lookup.
    # Comparing 384-dim vectors is about a millisecond for a few hundred, which is free
    # next to an LLM call, but it should still not grow without limit.
    semantic_cache_max_entries: int = 200
    semantic_cache_scan_limit: int = 200
    # Which roles may be served from cache. Synthesis is deliberately left out.
    #
    # It is the expensive call and the obvious one to cache, which is exactly why it's
    # tempting and exactly why it's wrong here. The note is the only part anyone reads,
    # and a cached note is one nobody reasoned about this time round: it would still be
    # returned confidently, with a fresh timestamp, after the evidence underneath it had
    # moved. Reusing the debate is a saving, reusing the conclusion is a stale answer
    # wearing a new date. The debate is where most of the calls are anyway.
    semantic_cache_roles: str = "planner,bull,bear"

    # Long-term memory. Finished notes are embedded into pgvector and recalled when the
    # same ticker comes round again, so a later run starts from what we already concluded
    # instead of from nothing.
    memory_enabled: bool = True
    # How many past notes to put in front of the planner. Three is enough to show a trend
    # without the prompt turning into a history lesson.
    memory_recall_limit: int = 3

    # Filings are megabytes of html. This is the ceiling on what the worker will pull
    # down for one document, a guard against a pathological file eating all our memory.
    max_document_bytes: int = 20_000_000

    # SEC EDGAR requires a User-Agent header with real contact info, they block
    # requests without one. Format they ask for: "Sample Company AdminContact@example.com".
    sec_user_agent: str = "DeepEquity research divyanshr141@gmail.com"

    # NewsAPI.org key for fetch_news. Free tier works for dev, leave blank and the
    # news tool will report it's not configured instead of crashing.
    news_api_key: str = ""

    #splits the comma separated host list into something the MCP server can use, and
    #drops any stray whitespace so "a, b" works the same as "a,b"
    def allowed_hosts(self) -> list[str]:
        return [host.strip() for host in self.mcp_allowed_hosts.split(",") if host.strip()]


#cached so we build the Settings object once per process, not once per request
@lru_cache
def get_settings() -> Settings:
    return Settings()
