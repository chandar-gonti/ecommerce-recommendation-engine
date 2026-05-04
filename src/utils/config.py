"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # AWS
    aws_region: str = "us-east-1"
    model_bucket: str = "recommender-models"
    opensearch_endpoint: str = "https://search-products.us-east-1.es.amazonaws.com"
    user_state_table: str = "user_state"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ML
    blend_alpha: float = 0.7
    cache_ttl_seconds: int = 300


@lru_cache
def _load() -> Settings:
    return Settings()


settings = _load()
