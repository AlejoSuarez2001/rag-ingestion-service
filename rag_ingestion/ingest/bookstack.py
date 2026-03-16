import logging
import time
from typing import Any

import httpx

from rag_ingestion.ingest.models import BookStackPage

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 1.0
_SEARCH_PAGE_SIZE = 100

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
        """Fetch every page across all books with full hierarchy metadata."""
        logger.info("Fetching all pages from BookStack at %s", self._base)
        raw_pages = self._fetch_search_pages()
        logger.info("Found %d page documents in total", len(raw_pages))

        pages: list[BookStackPage] = []
        for raw in raw_pages:
            try:
                pages.append(self._enrich_page(raw))
            except Exception:
                logger.warning("Failed to enrich page id=%s, skipping", raw.get("id"), exc_info=True)

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

    def _enrich_page(self, raw: dict[str, Any]) -> BookStackPage:
        """Combine raw page listing data with its markdown content and hierarchy info."""
        page_id: int = raw["id"]
        book_id: int = raw.get("book_id", 0)
        chapter_id: int | None = raw.get("chapter_id") or None

        book = raw.get("book") or {}
        book_name = book.get("name", f"book_{book_id}")
        book_slug = book.get("slug", str(book_id))

        chapter = raw.get("chapter") or {}
        chapter_name = chapter.get("name") if chapter else None

        content_markdown = self.get_page_markdown(page_id)

        url = f"{self._base}/books/{book_slug}/page/{raw.get('slug', page_id)}"

        return BookStackPage(
            id=page_id,
            title=raw.get("name", ""),
            slug=raw.get("slug", ""),
            url=url,
            updated_at=raw.get("updated_at", ""),
            content_markdown=content_markdown,
            book_id=book_id,
            book_name=book_name,
            book_slug=book_slug,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
        )

    def _fetch_search_pages(self) -> list[dict[str, Any]]:
        """Fetch the complete page list from the BookStack search endpoint."""
        documents: list[dict[str, Any]] = []
        total = 0
        page = 1

        while True:
            resp = self._get_with_retry(
                f"{self._base}/api/search",
                params={"query": "*", "count": _SEARCH_PAGE_SIZE, "page": page},
            )
            body = resp.json()
            batch = body.get("data", [])
            total = body.get("total", total)

            if not batch:
                break

            documents.extend(batch)
            page += 1

            if len(documents) >= total:
                break

        pages = [item for item in documents if item.get("type") == "page"]
        logger.info(
            "BookStack /api/search reportó total=%d; se recuperaron %d documentos; %d con type=page",
            total,
            len(documents),
            len(pages),
        )
        return pages

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
