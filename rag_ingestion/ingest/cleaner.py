import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)


class DoclingCleaner:
    """
    Uses Docling to parse and re-export markdown, normalising formatting
    artefacts that BookStack introduces (callouts, inline HTML, etc.).
    Falls back to a regex-based cleaner if Docling is unavailable.
    """

    def __init__(self) -> None:
        self._converter = self._load_converter()

    def clean(self, markdown_text: str) -> str:
        if not markdown_text.strip():
            return ""
        if self._converter is not None:
            try:
                return self._docling_clean(markdown_text)
            except Exception:
                logger.warning("Docling cleaning failed, using fallback cleaner", exc_info=True)
        return self._regex_clean(markdown_text)

    # ------------------------------------------------------------------
    # Docling path
    # ------------------------------------------------------------------

    def _docling_clean(self, text: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmp_path = f.name
        try:
            result = self._converter.convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _load_converter():
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat

            return DocumentConverter(allowed_formats=[InputFormat.MD])
        except ImportError:
            logger.warning("Docling not installed — using regex cleaner as fallback")
            return None

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _regex_clean(text: str) -> str:
        # Remove BookStack callout blocks e.g. {.callout}, <warning>, etc.
        text = re.sub(r"<(warning|info|danger|success|callout)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove BookStack image attachment references
        text = re.sub(r"!\[.*?\]\(.*?attachment.*?\)", "", text)
        # Clean up trailing whitespace on lines
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()
