"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Conservative application-level embedding budgets.
    embedding_rpm_limit: int = 80
    embedding_tpm_limit: int = 24000
    embedding_rpd_limit: int = 800
    embedding_batch_size: int = 80
    embedding_chars_per_token: float = 3.5

    cors_origins: str = "*"

    chunk_size: int = 700
    chunk_overlap: int = 100

    initial_retrieval_k: int = 20
    rerank_top_k: int = 5
    min_retrieval_score: float = 0.25

    max_file_size_mb: int = 20
    max_query_length: int = 500

    grounding_max_retries: int = 2
    query_rewrite_enabled: bool = True

    log_level: str = "INFO"

    storage_dir: str = "data/storage"
    index_dir: str = "data/index"
    conversation_db_path: str = "data/storage/conversations.sqlite3"
    conversation_history_limit: int = 12

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | List[str]) -> str:
        if isinstance(v, list):
            return ",".join(v)
        return v or "*"

    @property
    def cors_origin_list(self) -> List[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def storage_path(self) -> Path:
        return Path(self.storage_dir)

    def index_path(self) -> Path:
        return Path(self.index_dir)


settings = Settings()
