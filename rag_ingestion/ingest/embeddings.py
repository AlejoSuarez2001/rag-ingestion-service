import logging
import httpx
from rag_ingestion.ingest.chunking import Chunk

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Generates dense vector embeddings via Ollama remote API."""

    def __init__(self, ollama_base_url: str, model_name: str, batch_size: int = 32) -> None:
        self._url = f"{ollama_base_url}/api/embed"
        self._model = model_name
        self._batch_size = batch_size
        self._vector_size: int | None = None
        logger.info("Using Ollama embeddings: %s via %s", model_name, ollama_base_url)

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            try:
                with httpx.Client(timeout=60) as client:
                    response = client.post(
                        self._url,
                        json={"model": self._model, "input": "test"},
                    )
                    response.raise_for_status()
                    embedding = response.json()["embeddings"][0]
                    self._vector_size = len(embedding)
            except Exception as e:
                logger.error("Failed to determine vector size: %s", e)
                raise
        return self._vector_size

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """
        Generate embeddings for all chunks via remote Ollama.
        Returns a list of (chunk, embedding_vector) tuples.
        """
        if not chunks:
            return []

        texts = [self._chunk_to_text(c) for c in chunks]
        results = []

        logger.info("Embedding %d chunks in batches of %d via Ollama", len(chunks), self._batch_size)

        with httpx.Client(timeout=300) as client:
            for i in range(0, len(chunks), self._batch_size):
                batch_chunks = chunks[i:i + self._batch_size]
                batch_texts = texts[i:i + self._batch_size]

                try:
                    response = client.post(
                        self._url,
                        json={"model": self._model, "input": batch_texts},
                    )
                    response.raise_for_status()
                    embeddings = response.json()["embeddings"]
                    results.extend(zip(batch_chunks, embeddings))
                except Exception as e:
                    logger.error("Embedding batch failed: %s", e)
                    raise

        return results

    @staticmethod
    def _chunk_to_text(chunk: Chunk) -> str:
        """Prepend title to content for better semantic representation."""
        if chunk.title and chunk.title.lower() not in chunk.content.lower()[:100]:
            return f"{chunk.title}\n\n{chunk.content}"
        return chunk.content
