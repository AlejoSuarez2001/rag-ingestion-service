import logging
import httpx
from typing import Iterator
from rag_ingestion.ingest.models import BookStackPage

logger = logging.getLogger(__name__)


class BookStackClient:
    """HTTP client for the BookStack REST API."""

    def __init__(self, base_url: str, token_id: str, token_secret: str, page_size: int = 500) -> None:
        self._base = base_url.rstrip("/")
        self._page_size = page_size
        self._headers = {
            "Authorization": f"Token {token_id}:{token_secret}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(headers=self._headers, timeout=30)

        # In-memory caches to avoid redundant API calls
        self._books_cache: dict[int, dict] = {}
        self._chapters_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_pages(self) -> list[BookStackPage]:
        """Fetch every page across all books with full hierarchy metadata."""
        logger.info("Fetching all pages from BookStack at %s", self._base)
        self._warm_caches()

        raw_pages = list(self._paginate("/api/pages"))
        logger.info("Found %d pages in total", len(raw_pages))

        pages: list[BookStackPage] = []
        for raw in raw_pages:
            try:
                pages.append(self._enrich_page(raw))
            except Exception:
                logger.warning("Failed to enrich page id=%s, skipping", raw.get("id"), exc_info=True)

        return pages

    def get_page_markdown(self, page_id: int) -> str:
        """Return the markdown content of a single page."""
        resp = self._client.get(f"{self._base}/api/pages/{page_id}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("markdown") or self._html_to_md_fallback(data.get("html", ""))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warm_caches(self) -> None:
        """Pre-load all books and chapters to avoid N+1 API calls."""
        for book in self._paginate("/api/books"):
            self._books_cache[book["id"]] = book

        for chapter in self._paginate("/api/chapters"):
            self._chapters_cache[chapter["id"]] = chapter

        logger.debug(
            "Cache warmed: %d books, %d chapters",
            len(self._books_cache),
            len(self._chapters_cache),
        )

    def _enrich_page(self, raw: dict) -> BookStackPage:
        """Combine raw page listing data with its markdown content and hierarchy info."""
        page_id: int = raw["id"]
        book_id: int = raw.get("book_id", 0)
        chapter_id: int | None = raw.get("chapter_id") or None

        book = self._books_cache.get(book_id, {})
        book_name = book.get("name", f"book_{book_id}")
        book_slug = book.get("slug", str(book_id))

        chapter = self._chapters_cache.get(chapter_id, {}) if chapter_id else {}
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

    def _paginate(self, endpoint: str) -> Iterator[dict]:
        """Generic paginator for BookStack list endpoints."""
        offset = 0
        while True:
            resp = self._client.get(
                f"{self._base}{endpoint}",
                params={"count": self._page_size, "offset": offset},
            )
            resp.raise_for_status()
            body = resp.json()
            items: list[dict] = body.get("data", [])
            if not items:
                break
            yield from items
            offset += len(items)
            if offset >= body.get("total", 0):
                break

    @staticmethod
    def _html_to_md_fallback(html: str) -> str:
        """Very basic HTML-to-text fallback when markdown field is empty."""
        import re
        text = re.sub(r"<[^>]+>", "", html)
        return text.strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
