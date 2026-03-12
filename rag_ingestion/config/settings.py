from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # BookStack
    bookstack_url: str = ""
    bookstack_token_id: str = ""
    bookstack_token_secret: str = ""

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_manuals"

    # Embeddings (Vectores de 768 dimensiones)
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cuda"

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
