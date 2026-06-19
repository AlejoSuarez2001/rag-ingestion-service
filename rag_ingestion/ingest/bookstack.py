import logging
import time
from typing import Any

import httpx

from rag_ingestion.ingest.models import BookStackPage

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0
_LIST_PAGE_SIZE = 500

class BookStackClient:
    """HTTP client for the BookStack REST API."""

    def __init__(self, base_url: str, token_id: str, token_secret: str) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Token {token_id}:{token_secret}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(headers=self._headers, timeout=30)

    # Public API

    def get_all_pages(self) -> list[BookStackPage]:
        """Fetch every page across all books with full hierarchy metadata.

        Uses the ``/api/pages`` listing endpoint (paginated) to enumerate the
        complete catalogue. The previous ``/api/search?query=*`` approach was a
        relevance search and silently capped results, dropping whole books.

        Since ``/api/pages`` does not embed the book/chapter ``name`` (only
        ``book_slug``), the book and chapter names/slugs are resolved from
        ``/api/books`` and ``/api/chapters``, each fetched once into a lookup map.
        """
        logger.info("Fetching all pages from BookStack at %s", self._base)
        books_map = self._fetch_books_map()
        chapters_map = self._fetch_chapters_map()
        raw_pages = self._fetch_all("/api/pages")
        logger.info(
            "BookStack /api/pages devolvió %d páginas (%d libros, %d capítulos en los mapas)",
            len(raw_pages), len(books_map), len(chapters_map),
        )

        pages: list[BookStackPage] = []
        skipped_drafts = 0
        for raw in raw_pages:
            if raw.get("draft"):
                skipped_drafts += 1
                continue
            try:
                pages.append(self._enrich_page(raw, books_map, chapters_map))
            except Exception:
                logger.warning("Failed to enrich page id=%s, skipping", raw.get("id"), exc_info=True)

        logger.info(
            "Páginas enriquecidas: %d (descartadas %d drafts, %d con error de enriquecimiento)",
            len(pages), skipped_drafts, len(raw_pages) - skipped_drafts - len(pages),
        )
        return pages

    def get_page_markdown(self, page_id: int) -> str:
        """Return the markdown content of a single page."""
        resp = self._get_with_retry(f"{self._base}/api/pages/{page_id}/export/markdown")
        markdown = resp.text
        if not markdown.strip():
            raise RuntimeError(f"BookStack returned empty markdown for page {page_id}")
        return markdown

    # Private helpers

    def _get_with_retry(self, url: str, **kwargs) -> httpx.Response:
        """
        Execute a GET request with exponential backoff retry on transient errors.
        Retries on connection errors and HTTP 5xx/429 responses.
        """
        delay = _RETRY_BASE_DELAY
        last_exc: Exception | None = None

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = self._client.get(url, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "BookStack devolvió HTTP %d (intento %d/%d), reintentando en %.1fs…",
                        resp.status_code, attempt, _RETRY_ATTEMPTS, delay,
                    )
                    if attempt < _RETRY_ATTEMPTS:
                        time.sleep(delay)
                        delay *= 2
                        continue
                resp.raise_for_status()
                return resp
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(
                    "Error de red al llamar a BookStack (intento %d/%d): %s. Reintentando en %.1fs…",
                    attempt, _RETRY_ATTEMPTS, exc, delay,
                )
                if attempt < _RETRY_ATTEMPTS:
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(
            f"BookStack API no respondió tras {_RETRY_ATTEMPTS} intentos: {last_exc}"
        )

    def _enrich_page(
        self,
        raw: dict[str, Any],
        books_map: dict[int, dict[str, Any]],
        chapters_map: dict[int, dict[str, Any]],
    ) -> BookStackPage:
        """Combine raw page listing data with its markdown content and hierarchy info."""
        page_id: int = raw["id"]
        book_id: int = raw.get("book_id", 0)
        chapter_id: int | None = raw.get("chapter_id") or None

        book = books_map.get(book_id, {})
        book_name = book.get("name", f"book_{book_id}")
        book_slug = book.get("slug") or raw.get("book_slug") or str(book_id)

        chapter = chapters_map.get(chapter_id) if chapter_id else None
        chapter_name = chapter.get("name") if chapter else None

        url = f"{self._base}/books/{book_slug}/page/{raw.get('slug', page_id)}"

        # content_markdown is intentionally left empty here; it is fetched lazily
        # by the ingestion loop only for pages that actually changed, so that an
        # incremental run does not export all ~1800 pages every time.
        return BookStackPage(
            id=page_id,
            title=raw.get("name", ""),
            slug=raw.get("slug", ""),
            url=url,
            updated_at=raw.get("updated_at", ""),
            book_id=book_id,
            book_name=book_name,
            book_slug=book_slug,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
        )

    def _fetch_books_map(self) -> dict[int, dict[str, Any]]:
        """Fetch all books once, keyed by id, for name/slug resolution."""
        return {book["id"]: book for book in self._fetch_all("/api/books")}

    def _fetch_chapters_map(self) -> dict[int, dict[str, Any]]:
        """Fetch all chapters once, keyed by id, for name resolution."""
        return {chapter["id"]: chapter for chapter in self._fetch_all("/api/chapters")}

    def _fetch_all(self, endpoint: str) -> list[dict[str, Any]]:
        """Fetch every item from a BookStack listing endpoint via offset pagination."""
        items: list[dict[str, Any]] = []
        total = 0
        offset = 0

        while True:
            resp = self._get_with_retry(
                f"{self._base}{endpoint}",
                params={"count": _LIST_PAGE_SIZE, "offset": offset},
            )
            body = resp.json()
            batch = body.get("data", [])
            total = body.get("total", total)

            if not batch:
                break

            items.extend(batch)
            offset += len(batch)

            if offset >= total:
                break

        logger.debug("BookStack %s: recuperados %d/%d ítems", endpoint, len(items), total)
        return items

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
