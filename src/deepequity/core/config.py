"""All the settings the app needs, pulled from environment variables.

Using pydantic-settings here means we get validation for free: if REDIS_URL
is missing or malformed, the app fails fast at startup instead of crashing
later when something actually tries to use Redis.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # JWT auth
    jwt_secret_key: str = "change-me-in-.env"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate limiting: N requests per window_seconds, per API key/IP
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60


@lru_cache
def get_settings() -> Settings:
    # Cached so we build the Settings object once per process, not once per request.
    return Settings()
