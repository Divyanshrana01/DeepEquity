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
