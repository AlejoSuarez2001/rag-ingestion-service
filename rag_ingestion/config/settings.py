from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # BookStack
    bookstack_url: str = "https://docs.example.com"
    bookstack_token_id: str = ""
    bookstack_token_secret: str = ""
    bookstack_page_size: int = 500  # max items per API call

    # Qdrant (shared with hybrid-rag-backend)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "tech_manuals"

    # Embeddings
    # ⚠️ MUST match the model used in hybrid-rag-backend for retrieval to work.
    # Default produces 768-dim vectors, same as nomic-embed-text.
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 100
    tiktoken_encoding: str = "cl100k_base"

    # State tracking (SQLite)
    tracker_db_path: str = "ingestion_state.db"

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
