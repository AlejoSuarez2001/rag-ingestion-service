import logging
import os
import re
import tempfile
from html import unescape

logger = logging.getLogger(__name__)


_FENCED_CODE_BLOCK_RE = re.compile(r"(^```[\s\S]*?^```[ \t]*$)", re.MULTILINE)


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
                return self._normalize_markdown(self._docling_clean(markdown_text))
            except Exception:
                logger.warning("Docling cleaning failed, using fallback cleaner", exc_info=True)
        else:
            logger.debug("Docling no disponible, usando limpieza por regex como fallback")
        return self._normalize_markdown(self._regex_clean(markdown_text))

    # Docling path

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

    # Regex fallback

    @staticmethod
    def _regex_clean(text: str) -> str:
        # Remove all markdown image references
        text = re.sub(r"!\[.*?\]\([^)]*\)", "", text)
        return text

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        segments = _FENCED_CODE_BLOCK_RE.split(text)

        normalized_segments: list[str] = []
        for index, segment in enumerate(segments):
            if not segment.strip():
                continue
            if index % 2 == 1:
                normalized_segments.append(segment.strip("\n"))
            else:
                cleaned = DoclingCleaner._normalize_text_segment(segment)
                if cleaned:
                    normalized_segments.append(cleaned)

        text = "\n\n".join(normalized_segments)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _normalize_text_segment(text: str) -> str:
        text = unescape(text)
        text = DoclingCleaner._repair_mojibake(text)

        # Remove BookStack callout blocks e.g. <warning>, <info>, etc.
        text = re.sub(
            r"<(warning|info|danger|success|callout)[^>]*>.*?</\1>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Preserve block structure before stripping remaining HTML.
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|section|article|blockquote|h[1-6])>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<li[^>]*>", "- ", text, flags=re.IGNORECASE)

        # Strip formatting tags while keeping their text content.
        text = re.sub(r"</?(span|font|strong|em|b|i|u|mark|small|sup|sub|code)[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"<a [^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", r"\2 (\1)", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)

        # Recover common markdown structure when content arrives flattened.
        text = re.sub(r"```[ \t]*(.*?)[ \t]*```", lambda match: f"```\n{match.group(1).strip()}\n```", text, flags=re.DOTALL)
        text = re.sub(r"([^\n])\s*(```)", r"\1\n\n\2", text)
        text = re.sub(r"(```)\s*([^\n])", r"\1\n\n\2", text)
        text = re.sub(r"```\n{2,}", "```\n", text)
        text = re.sub(r"\n{2,}```", "\n```", text)
        text = re.sub(r"\s+>\s*", "\n> ", text)
        text = re.sub(r">\s+>", ">\n>", text)
        text = re.sub(
            r"(?im)^\s*image-\d+\.(png|jpe?g|gif|webp|svg)\s*$",
            "[Imagen referenciada en esta sección]",
            text,
        )

        # Normalize whitespace without crushing markdown semantics.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = "\n".join(
            line.rstrip()
            for line in text.splitlines()
            if not DoclingCleaner._is_noise_line(line)
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _repair_mojibake(text: str) -> str:
        if not any(marker in text for marker in ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d", "â€“", "â€”")):
            return text

        try:
            repaired = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            return text

        if DoclingCleaner._mojibake_score(repaired) >= DoclingCleaner._mojibake_score(text):
            return text

        return repaired

    @staticmethod
    def _mojibake_score(text: str) -> int:
        return sum(text.count(marker) for marker in ("Ã", "Â", "â€", "â€™", "â€œ", "â€\x9d", "â€“", "â€”"))

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        marker_stripped = re.sub(r"^[-*+>\]\['\",\.\s]+", "", stripped)
        marker_stripped = re.sub(r"[-*+<>\]\['\",\.;:\s]+$", "", marker_stripped)

        alnum_count = sum(char.isalnum() for char in stripped)
        symbol_count = sum(not char.isalnum() and not char.isspace() for char in stripped)
        marker_alnum_count = sum(char.isalnum() for char in marker_stripped)

        if alnum_count == 0 and symbol_count >= 4:
            return True

        if alnum_count == 0 and len(stripped) <= 3 and symbol_count >= 1:
            return True

        if marker_alnum_count == 0 and symbol_count >= 2:
            return True

        if marker_alnum_count <= 1 and len(marker_stripped) <= 2 and symbol_count >= 2:
            return True

        noisy_chars = sum(char in "[]<>{}|\\/'`\",;:" for char in stripped)
        if alnum_count <= 2 and noisy_chars >= max(4, len(stripped) // 2):
            return True

        return False
