from dataclasses import dataclass, field


@dataclass
class BookStackPage:
    """Represents a BookStack page with its full hierarchy metadata."""

    id: int
    title: str
    slug: str
    url: str
    updated_at: str

    book_id: int
    book_name: str
    book_slug: str

    chapter_id: int | None
    chapter_name: str | None

    # Fetched lazily per-page during ingestion (see BookStackClient.get_page_markdown),
    # not during enumeration — keeps get_all_pages() cheap.
    content_markdown: str = ""

    @property
    def display_path(self) -> str:
        parts = [self.book_name]
        if self.chapter_name:
            parts.append(self.chapter_name)
        parts.append(self.title)
        return " > ".join(parts)
