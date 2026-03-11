import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Literal

from rag_ingestion.config.settings import Settings
from rag_ingestion.ingest.bookstack import BookStackClient
from rag_ingestion.ingest.chunking import ChunkingService
from rag_ingestion.ingest.cleaner import DoclingCleaner
from rag_ingestion.ingest.db import QdrantStore, PageTracker, content_hash
from rag_ingestion.ingest.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

JobStatus = Literal["idle", "running", "completed", "failed"]


class IngestionJob:
    """Estado en memoria del último job de ingestion."""

    def __init__(self) -> None:
        self.status: JobStatus = "idle"
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.stats: dict = {}
        self.error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stats": self.stats,
            "error": self.error,
        }


class IngestionService:
    """
    Orquesta el pipeline completo: BookStack → clean → chunk → embed → Qdrant.
    Mantiene un job singleton para evitar ejecuciones concurrentes.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self.job = IngestionJob()

        # Servicios reutilizables inicializados una vez
        self._embedder = EmbeddingService(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )
        self._qdrant = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection=settings.qdrant_collection,
        )
        self._qdrant.ensure_collection(vector_size=self._embedder.vector_size)
        self._cleaner = DoclingCleaner()
        self._chunker = ChunkingService(
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
            tiktoken_encoding=settings.tiktoken_encoding,
        )

    def is_running(self) -> bool:
        return self.job.status == "running"

    def run_full_ingestion(self, force: bool = False, page_id: int | None = None) -> None:
        """
        Ejecuta el pipeline completo de ingestion de forma sincrónica.
        Llamar desde un thread separado para no bloquear el event loop.
        """
        with self._lock:
            self.job.status = "running"
            self.job.started_at = datetime.now(timezone.utc).isoformat()
            self.job.completed_at = None
            self.job.error = None
            self.job.stats = {}

        stats = {"total": 0, "ingested": 0, "skipped": 0, "errors": 0}

        try:
            with BookStackClient(
                base_url=self._settings.bookstack_url,
                token_id=self._settings.bookstack_token_id,
                token_secret=self._settings.bookstack_token_secret,
                page_size=self._settings.bookstack_page_size,
            ) as bookstack, PageTracker(
                host=self._settings.postgres_host,
                port=self._settings.postgres_port,
                dbname=self._settings.postgres_db,
                user=self._settings.postgres_user,
                password=self._settings.postgres_password,
            ) as tracker:

                pages = bookstack.get_all_pages()

                if page_id is not None:
                    pages = [p for p in pages if p.id == page_id]

                stats["total"] = len(pages)
                logger.info("Ingestion iniciada: %d páginas a procesar", len(pages))

                for page in pages:
                    try:
                        page_hash = content_hash(page.content_markdown)

                        if not force and not tracker.needs_reingestion(page.id, page.updated_at, page_hash):
                            logger.debug("Página %d sin cambios, saltando", page.id)
                            stats["skipped"] += 1
                            continue

                        version = tracker.get_version(page.id)
                        cleaned = self._cleaner.clean(page.content_markdown)
                        chunks = self._chunker.chunk_page(page, cleaned, version=version)

                        if not chunks:
                            logger.warning("Página %d generó 0 chunks, saltando", page.id)
                            stats["skipped"] += 1
                            continue

                        chunk_vectors = self._embedder.embed_chunks(chunks)
                        self._qdrant.delete_page_chunks(page.id)
                        self._qdrant.upsert_chunks(chunk_vectors)
                        tracker.save(
                            page_id=page.id,
                            updated_at=page.updated_at,
                            content_hash=page_hash,
                            chunk_count=len(chunks),
                        )

                        logger.info("✓ Página %d '%s' → %d chunks (v%d)", page.id, page.title, len(chunks), version)
                        stats["ingested"] += 1

                    except Exception:
                        logger.error("✗ Error procesando página %d '%s'", page.id, page.title, exc_info=True)
                        stats["errors"] += 1

            with self._lock:
                self.job.status = "completed"
                self.job.completed_at = datetime.now(timezone.utc).isoformat()
                self.job.stats = stats

            logger.info("Ingestion completada: %s", stats)

        except Exception as exc:
            logger.error("Ingestion fallida: %s", exc, exc_info=True)
            with self._lock:
                self.job.status = "failed"
                self.job.completed_at = datetime.now(timezone.utc).isoformat()
                self.job.error = str(exc)
                self.job.stats = stats
