import logging
import re
import tiktoken
from dataclasses import dataclass
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from rag_ingestion.ingest.models import BookStackPage

logger = logging.getLogger(__name__)

_PROCEDURAL_TITLE_RE = re.compile(r"^(paso\s+\d+|step\s+\d+)\b", re.IGNORECASE)

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
    source: str         # URL de la página en BookStack
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
            tokenizer=self._tokenize,
        )

    # Public API

    def chunk_page(self, page: BookStackPage, cleaned_markdown: str, version: int = 1) -> list[Chunk]:
        """Return all chunks for a single BookStack page."""
        if not cleaned_markdown.strip():
            logger.warning("Page %d (%s) has empty content after cleaning, skipping", page.id, page.title)
            return []

        doc = Document(
            text=cleaned_markdown,
            metadata={"page_id": str(page.id), "page_title": page.title},
        )

        heading_nodes = self._md_parser.get_nodes_from_documents([doc])
        if not heading_nodes:
            heading_nodes = [doc]

        chunks: list[Chunk] = []
        position = 0

        for node in heading_nodes:
            node_text = node.text.strip() if hasattr(node, "text") else str(node)
            if not node_text:
                continue

            heading = self._extract_heading(node)
            if not self._is_meaningful_chunk(node_text, heading, page.title):
                logger.debug("Skipping heading-only chunk candidate on page %d: %r", page.id, heading or page.title)
                continue
            token_count = self._count_tokens(node_text)

            if token_count <= self._chunk_size:
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
                sub_doc = Document(text=node_text)
                sub_nodes = self._splitter.get_nodes_from_documents([sub_doc])
                for sub_node in sub_nodes:
                    sub_text = sub_node.text.strip()
                    if not sub_text:
                        continue
                    if not self._is_meaningful_chunk(sub_text, heading, page.title):
                        logger.debug("Skipping heading-only sub-chunk on page %d: %r", page.id, heading or page.title)
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

        chunks = self._merge_small_procedural_chunks(chunks)
        logger.debug("Page %d → %d chunks", page.id, len(chunks))
        return chunks

    # Private helpers

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
            source=page.url,
            tokens=self._count_tokens(content),
            version=version,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _tokenize(self, text: str) -> list[int]:
        return self._enc.encode(str(text))

    def _merge_small_procedural_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) < 2:
            return chunks

        merged: list[Chunk] = []
        buffer: list[Chunk] = []
        buffer_tokens = 0

        for chunk in chunks:
            if self._should_merge_procedural_chunk(chunk):
                if buffer_tokens + chunk.tokens <= self._chunk_size:
                    buffer.append(chunk)
                    buffer_tokens += chunk.tokens
                    continue

                merged.extend(self._flush_merged_chunks(buffer))
                buffer = [chunk]
                buffer_tokens = chunk.tokens
                continue

            merged.extend(self._flush_merged_chunks(buffer))
            buffer = []
            buffer_tokens = 0
            merged.append(chunk)

        merged.extend(self._flush_merged_chunks(buffer))
        return self._reindex_chunks(merged)

    def _flush_merged_chunks(self, buffer: list[Chunk]) -> list[Chunk]:
        if len(buffer) <= 1:
            return list(buffer)

        first = buffer[0]
        combined_content = "\n\n".join(chunk.content.strip() for chunk in buffer if chunk.content.strip())
        merged_chunk = Chunk(
            chunk_id=first.chunk_id,
            position=first.position,
            page_id=first.page_id,
            title=first.title,
            book=first.book,
            chapter=first.chapter,
            page=first.page,
            content=combined_content,
            source=first.source,
            tokens=self._count_tokens(combined_content),
            version=first.version,
        )
        return [merged_chunk]

    def _reindex_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        for position, chunk in enumerate(chunks):
            chunk.position = position
            chunk.chunk_id = f"{chunk.page_id}_{position}"
        return chunks

    @staticmethod
    def _should_merge_procedural_chunk(chunk: Chunk) -> bool:
        if chunk.tokens > 80:
            return False
        return bool(_PROCEDURAL_TITLE_RE.match(chunk.title.strip()))

    @staticmethod
    def _is_meaningful_chunk(content: str, heading: str, page_title: str) -> bool:
        non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not non_empty_lines:
            return False

        normalized_lines = [ChunkingService._normalize_quality_text(line) for line in non_empty_lines]
        normalized_heading = ChunkingService._normalize_quality_text(heading)
        normalized_page_title = ChunkingService._normalize_quality_text(page_title)

        if len(normalized_lines) == 1 and normalized_lines[0] in {normalized_heading, normalized_page_title}:
            return False

        unique_lines = {line for line in normalized_lines if line}
        if unique_lines and unique_lines.issubset({normalized_heading, normalized_page_title}):
            return False

        body_lines = normalized_lines[1:] if len(normalized_lines) > 1 else []
        if body_lines and all(line in {normalized_heading, normalized_page_title, ""} for line in body_lines):
            return False

        return True

    @staticmethod
    def _normalize_quality_text(text: str) -> str:
        text = re.sub(r"^#{1,6}\s+", "", text.strip())
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
        return text.casefold().strip()

    @staticmethod
    def _extract_heading(node) -> str:
        """Extract the section heading from a LlamaIndex node's metadata."""
        metadata = getattr(node, "metadata", {}) or {}

        header_keys = sorted(
            [k for k in metadata if k.lower().startswith("header")],
            reverse=True,
        )
        for key in header_keys:
            value = metadata.get(key)
            if value:
                return str(value)

        text = getattr(node, "text", "") or ""
        match = re.match(r"^#{1,6}\s+(.+)", text.strip())
        if match:
            return match.group(1).strip()

        return ""
