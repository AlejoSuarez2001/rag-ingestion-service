import hashlib
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TokenizerType,
    VectorParams,
)

from rag_ingestion.ingest.chunking import Chunk

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Qdrant vector store
# ──────────────────────────────────────────────────────────────────────────────


class QdrantStore:
    """Manages chunk upserts and deletions in Qdrant."""

    def __init__(self, host: str, port: int, collection: str) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection and payload indexes if they don't exist yet."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            logger.info("Creating Qdrant collection '%s' (dim=%d)", self._collection, vector_size)
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            self._create_indexes()
        else:
            logger.info("Collection '%s' already exists", self._collection)

    def upsert_chunks(self, chunk_vectors: list[tuple[Chunk, list[float]]]) -> None:
        """Batch-upsert chunks into Qdrant."""
        if not chunk_vectors:
            return

        points = [
            PointStruct(
                id=self._chunk_uuid(chunk.chunk_id),
                vector=vector,
                payload=self._chunk_to_payload(chunk),
            )
            for chunk, vector in chunk_vectors
        ]

        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.debug("Upserted %d points for page_id=%s", len(points), chunk_vectors[0][0].page_id)

    def delete_page_chunks(self, page_id: int) -> None:
        """Delete all chunks belonging to a page (called before re-ingestion)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="page_id", match=MatchValue(value=page_id))]
            ),
            wait=True,
        )
        logger.info("Deleted existing chunks for page_id=%d", page_id)

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_indexes(self) -> None:
        """Create keyword (full-text) and payload field indexes."""
        # Full-text index on "text" for keyword search in hybrid-rag-backend
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
        # Numeric index on page_id for fast delete-by-page
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="page_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
        logger.info("Created payload indexes on 'text' and 'page_id'")

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> dict:
        return {
            # DER fields
            "chunk_id": chunk.chunk_id,
            "position": chunk.position,
            "page_id": chunk.page_id,
            "title": chunk.title,
            "book": chunk.book,
            "chapter": chunk.chapter,
            "page": chunk.page,
            "content": chunk.content,
            "url": chunk.url,
            "tokens": chunk.tokens,
            "version": chunk.version,
            # Compatibility aliases for hybrid-rag-backend
            "text": chunk.content,      # used by retrieval_service keyword search
            "source": chunk.url,        # used by retrieval_service for source attribution
        }

    @staticmethod
    def _chunk_uuid(chunk_id: str) -> str:
        """Deterministic UUID v5 from chunk_id — stable across re-ingestions."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rag_chunk:{chunk_id}"))


# ──────────────────────────────────────────────────────────────────────────────
# SQLite page tracker
# ──────────────────────────────────────────────────────────────────────────────


class PageTracker:
    """
    Lightweight SQLite tracker that records which pages have been ingested
    and their content hash, enabling incremental re-ingestion.

    This is the foundation for the future scheduled change-detection job.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS pages (
            page_id      INTEGER PRIMARY KEY,
            updated_at   TEXT    NOT NULL,
            content_hash TEXT    NOT NULL,
            version      INTEGER NOT NULL DEFAULT 1,
            chunk_count  INTEGER NOT NULL DEFAULT 0,
            ingested_at  TEXT    NOT NULL
        )
    """

    def __init__(self, db_path: str = "ingestion_state.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()
        logger.info("PageTracker connected to %s", db_path)

    def get(self, page_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        return dict(row) if row else None

    def needs_reingestion(self, page_id: int, updated_at: str, content_hash: str) -> bool:
        """Return True if the page is new or its content has changed."""
        record = self.get(page_id)
        if record is None:
            return True
        return record["content_hash"] != content_hash or record["updated_at"] != updated_at

    def save(self, page_id: int, updated_at: str, content_hash: str, chunk_count: int) -> None:
        existing = self.get(page_id)
        version = (existing["version"] + 1) if existing else 1
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO pages (page_id, updated_at, content_hash, version, chunk_count, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
                updated_at   = excluded.updated_at,
                content_hash = excluded.content_hash,
                version      = excluded.version,
                chunk_count  = excluded.chunk_count,
                ingested_at  = excluded.ingested_at
            """,
            (page_id, updated_at, content_hash, version, chunk_count, now),
        )
        self._conn.commit()

    def get_version(self, page_id: int) -> int:
        record = self.get(page_id)
        return (record["version"] + 1) if record else 1

    def all_tracked_pages(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM pages ORDER BY ingested_at DESC").fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Content hashing utility
# ──────────────────────────────────────────────────────────────────────────────


def content_hash(text: str) -> str:
    """MD5 hash of the page content — used for change detection."""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()
