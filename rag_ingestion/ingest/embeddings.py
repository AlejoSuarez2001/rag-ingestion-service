import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_ingestion.ingest.chunking import Chunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates dense vector embeddings using Sentence Transformers."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        logger.info("Loading embedding model: %s (device=%s)", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._vector_size: int | None = None

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = self._model.get_sentence_embedding_dimension()
        return self._vector_size

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """
        Generate embeddings for all chunks.
        Returns a list of (chunk, embedding_vector) tuples.
        The input text combines title + content for richer contextual embeddings.
        """
        if not chunks:
            return []

        texts = [self._chunk_to_text(c) for c in chunks]

        logger.info("Embedding %d chunks in batches of %d", len(chunks), self._batch_size)
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(chunks) > 50,
            normalize_embeddings=True,  # cosine similarity optimized
            convert_to_numpy=True,
        )

        return [(chunk, vec.tolist()) for chunk, vec in zip(chunks, vectors)]

    @staticmethod
    def _chunk_to_text(chunk: Chunk) -> str:
        """Prepend title to content for better semantic representation."""
        if chunk.title and chunk.title.lower() not in chunk.content.lower()[:100]:
            return f"{chunk.title}\n\n{chunk.content}"
        return chunk.content
