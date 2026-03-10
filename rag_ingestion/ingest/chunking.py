import logging
import tiktoken
from dataclasses import dataclass
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from rag_ingestion.ingest.models import BookStackPage

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single chunk matching the vector DB DER."""

    chunk_id: str       # "{page_id}_{position}"
    position: int
    page_id: int
    title: str          # heading of this specific section
    book: str
    chapter: str        # empty string if page is directly under book
    page: str           # BookStack page title
    content: str
    url: str
    tokens: int
    version: int = 1


class ChunkingService:
    """
    Splits a BookStack page into chunks following a heading-priority strategy:
      1. MarkdownNodeParser splits on headings (semantic sections).
      2. Sections exceeding chunk_size are further split by SentenceSplitter.
      3. Sections below chunk_size become a single chunk.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        tiktoken_encoding: str = "cl100k_base",
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._enc = tiktoken.get_encoding(tiktoken_encoding)

        self._md_parser = MarkdownNodeParser()
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            tokenizer=self._count_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_page(self, page: BookStackPage, cleaned_markdown: str, version: int = 1) -> list[Chunk]:
        """Return all chunks for a single BookStack page."""
        if not cleaned_markdown.strip():
            logger.warning("Page %d (%s) has empty content after cleaning, skipping", page.id, page.title)
            return []

        doc = Document(
            text=cleaned_markdown,
            metadata={"page_id": str(page.id), "page_title": page.title},
        )

        # Step 1: split by headings
        heading_nodes = self._md_parser.get_nodes_from_documents([doc])
        if not heading_nodes:
            heading_nodes = [doc]  # no headings found — treat full page as one node

        chunks: list[Chunk] = []
        position = 0

        for node in heading_nodes:
            node_text = node.text.strip() if hasattr(node, "text") else str(node)
            if not node_text:
                continue

            heading = self._extract_heading(node)
            token_count = self._count_tokens(node_text)

            if token_count <= self._chunk_size:
                # Small enough — one chunk
                chunks.append(
                    self._make_chunk(
                        page=page,
                        content=node_text,
                        heading=heading,
                        position=position,
                        version=version,
                    )
                )
                position += 1
            else:
                # Too large — split further while preserving heading context
                sub_doc = Document(text=node_text)
                sub_nodes = self._splitter.get_nodes_from_documents([sub_doc])
                for sub_node in sub_nodes:
                    sub_text = sub_node.text.strip()
                    if not sub_text:
                        continue
                    chunks.append(
                        self._make_chunk(
                            page=page,
                            content=sub_text,
                            heading=heading,
                            position=position,
                            version=version,
                        )
                    )
                    position += 1

        logger.debug("Page %d → %d chunks", page.id, len(chunks))
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        page: BookStackPage,
        content: str,
        heading: str,
        position: int,
        version: int,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{page.id}_{position}",
            position=position,
            page_id=page.id,
            title=heading or page.title,
            book=page.book_name,
            chapter=page.chapter_name or "",
            page=page.title,
            content=content,
            url=page.url,
            tokens=self._count_tokens(content),
            version=version,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    @staticmethod
    def _extract_heading(node) -> str:
        """Extract the section heading from a LlamaIndex node's metadata."""
        metadata = getattr(node, "metadata", {}) or {}

        # MarkdownNodeParser stores headers under keys like "Header 1", "Header 2", etc.
        # We want the deepest (most specific) non-null header.
        header_keys = sorted(
            [k for k in metadata if k.lower().startswith("header")],
            reverse=True,
        )
        for key in header_keys:
            value = metadata.get(key)
            if value:
                return str(value)

        # Fallback: try to extract first markdown heading from raw text
        import re
        text = getattr(node, "text", "") or ""
        match = re.match(r"^#{1,6}\s+(.+)", text.strip())
        if match:
            return match.group(1).strip()

        return ""
