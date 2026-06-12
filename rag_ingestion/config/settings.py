import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENV', 'dev')}",
        env_file_encoding="utf-8"
    )

    # BookStack
    bookstack_url: str
    bookstack_token_id: str
    bookstack_token_secret: str

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_manuals"

    # Ollama (embeddings remotos)
    ollama_base_url: str
    embedding_model: str
    embedding_batch_size: int = 32

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    tiktoken_encoding: str = "cl100k_base"

    # Postgres
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "rag_ingestion"
    postgres_user: str = "rag"
    postgres_password: str = "rag"

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
