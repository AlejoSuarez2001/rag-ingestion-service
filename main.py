"""
Main ingestion pipeline orchestrator.

Usage:
    python main.py ingest          # ingest all pages (skips unchanged)
    python main.py ingest --force  # re-ingest everything regardless of changes
    python main.py status          # show ingestion state from SQLite
    python main.py stats           # show Qdrant collection stats
"""
import logging
import sys
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from rag_ingestion.config.settings import get_settings
from rag_ingestion.ingest.bookstack import BookStackClient
from rag_ingestion.ingest.cleaner import DoclingCleaner
from rag_ingestion.ingest.chunking import ChunkingService
from rag_ingestion.ingest.embeddings import EmbeddingService
from rag_ingestion.ingest.db import QdrantStore, PageTracker, content_hash

app = typer.Typer(help="RAG Ingestion Service — BookStack → Qdrant")
console = Console()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest all pages regardless of changes"),
    page_id: int = typer.Option(None, "--page-id", "-p", help="Ingest a single page by ID"),
) -> None:
    """Download, clean, chunk, embed and store pages into Qdrant."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger("main")

    # ── Bootstrap services ────────────────────────────────────────────
    embedder = EmbeddingService(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )
    qdrant = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
    )
    qdrant.ensure_collection(vector_size=embedder.vector_size)

    cleaner = DoclingCleaner()
    chunker = ChunkingService(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
        tiktoken_encoding=settings.tiktoken_encoding,
    )

    stats = {"total": 0, "ingested": 0, "skipped": 0, "errors": 0}

    with BookStackClient(
        base_url=settings.bookstack_url,
        token_id=settings.bookstack_token_id,
        token_secret=settings.bookstack_token_secret,
        page_size=settings.bookstack_page_size,
    ) as bookstack, PageTracker(settings.tracker_db_path) as tracker:

        pages = bookstack.get_all_pages()
        if page_id is not None:
            pages = [p for p in pages if p.id == page_id]
            if not pages:
                console.print(f"[red]Page {page_id} not found in BookStack[/red]")
                raise typer.Exit(code=1)

        stats["total"] = len(pages)
        console.print(f"[bold]Found {len(pages)} page(s) to process[/bold]")

        for page in track(pages, description="Ingesting pages..."):
            try:
                page_hash = content_hash(page.content_markdown)

                if not force and not tracker.needs_reingestion(page.id, page.updated_at, page_hash):
                    logger.debug("Page %d unchanged, skipping", page.id)
                    stats["skipped"] += 1
                    continue

                version = tracker.get_version(page.id)

                # 1. Clean markdown with Docling
                cleaned = cleaner.clean(page.content_markdown)

                # 2. Chunk
                chunks = chunker.chunk_page(page, cleaned, version=version)
                if not chunks:
                    logger.warning("Page %d produced 0 chunks, skipping", page.id)
                    stats["skipped"] += 1
                    continue

                # 3. Embed
                chunk_vectors = embedder.embed_chunks(chunks)

                # 4. Delete old vectors + upsert new ones
                qdrant.delete_page_chunks(page.id)
                qdrant.upsert_chunks(chunk_vectors)

                # 5. Update tracker
                tracker.save(
                    page_id=page.id,
                    updated_at=page.updated_at,
                    content_hash=page_hash,
                    chunk_count=len(chunks),
                )

                logger.info(
                    "✓ Page %d '%s' → %d chunks (v%d)",
                    page.id, page.title, len(chunks), version,
                )
                stats["ingested"] += 1

            except Exception:
                logger.error("✗ Failed to ingest page %d '%s'", page.id, page.title, exc_info=True)
                stats["errors"] += 1

    # ── Summary ──────────────────────────────────────────────────────
    console.print("\n[bold green]Ingestion complete[/bold green]")
    _print_stats_table(stats)


@app.command()
def status() -> None:
    """Show the ingestion state of all tracked pages (from SQLite)."""
    settings = get_settings()
    setup_logging(settings.log_level)

    with PageTracker(settings.tracker_db_path) as tracker:
        rows = tracker.all_tracked_pages()

    if not rows:
        console.print("[yellow]No pages tracked yet. Run 'ingest' first.[/yellow]")
        return

    table = Table(title="Ingestion State", show_lines=True)
    table.add_column("page_id", style="cyan")
    table.add_column("title / updated_at")
    table.add_column("version", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("ingested_at")

    for r in rows:
        table.add_row(
            str(r["page_id"]),
            r["updated_at"],
            str(r["version"]),
            str(r["chunk_count"]),
            r["ingested_at"],
        )

    console.print(table)


@app.command()
def stats() -> None:
    """Show Qdrant collection statistics."""
    settings = get_settings()
    setup_logging(settings.log_level)

    qdrant = QdrantStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection=settings.qdrant_collection,
    )
    info = qdrant.collection_info()

    table = Table(title=f"Qdrant Collection: {settings.qdrant_collection}")
    table.add_column("Metric")
    table.add_column("Value", justify="right", style="green")

    for k, v in info.items():
        table.add_row(k, str(v))

    console.print(table)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _print_stats_table(stats: dict) -> None:
    table = Table(title="Run Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")

    colors = {"total": "white", "ingested": "green", "skipped": "yellow", "errors": "red"}
    for key, val in stats.items():
        table.add_row(f"[{colors[key]}]{key}[/{colors[key]}]", str(val))

    console.print(table)


if __name__ == "__main__":
    app()
