import logging
import re
from io import BytesIO

import httpx
from PIL import Image

from rag_ingestion.config.settings import Settings

logger = logging.getLogger(__name__)

# Patrón real del export de BookStack: linked-image anidada
#   [![alt](URL_thumbnail)](URL_original)
# Se usa el grupo 3 (URL original, sin /scaled-840-0/) para el OCR.
LINKED_IMG_RE = re.compile(r"\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)")
# Imagen markdown suelta (fallback): ![alt](url)
BARE_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# Alt que termina en extensión de imagen (image3.png, 2.png, foo.jpg) → nombre de archivo,
# inútil como descripción; se trata como vacío.
FILENAME_ALT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|bmp)$", re.IGNORECASE)
# Caracteres CJK (kanji/kana/fullwidth): en docs en español son alucinaciones del modelo OCR.
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")


class ImageTextExtractor:
    """
    Reemplaza las imágenes del markdown de BookStack por su transcripción de texto vía OCR.

    Corre en tiempo de ingesta (offline, CPU), dentro del propio servicio: NO usa la VM de
    inferencia ni agrega ninguna API HTTP. El texto resultante queda embebido + indexado
    full-text como cualquier párrafo, sin tocar `rag-api` ni el query-time.
    """

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.ocr_enabled
        self._text_score = settings.ocr_text_score
        self._box_thresh = settings.ocr_box_thresh
        self._unclip_ratio = settings.ocr_unclip_ratio
        self._max_bytes = settings.ocr_max_image_mb * 1024 * 1024
        self._client = httpx.Client(timeout=settings.ocr_download_timeout)
        self._engine = None  # RapidOCR, cargado lazy en el primer uso

    # API pública

    def resolve(self, markdown: str) -> str:
        """Devuelve el markdown con cada imagen reemplazada por su texto OCR."""
        if not self._enabled or not markdown:
            return markdown

        cache: dict[str, str] = {}

        def handle(alt: str, url: str) -> str:
            if url not in cache:
                cache[url] = self._extract_one(alt, url)
            return cache[url]

        # Linked-images primero (reemplaza todo el bloque), luego imágenes sueltas.
        text = LINKED_IMG_RE.sub(lambda m: handle(m.group(1), m.group(3)), markdown)
        text = BARE_IMG_RE.sub(lambda m: handle(m.group(1), m.group(2)), text)
        return text

    def close(self) -> None:
        self._client.close()

    # Privados

    def _extract_one(self, alt: str, url: str) -> str:
        """Descarga + OCR de una imagen. Nunca lanza: ante cualquier fallo cae al alt/vacío."""
        try:
            if not url.lower().startswith(("http://", "https://")):
                return self._build_replacement(alt, "")

            resp = self._client.get(url)
            resp.raise_for_status()
            content = resp.content

            ctype = resp.headers.get("content-type", "").lower()
            if "svg" in ctype or url.lower().split("?")[0].endswith(".svg"):
                logger.debug("OCR: imagen vectorial (SVG) salteada: %s", url)
                return self._build_replacement(alt, "")

            if len(content) > self._max_bytes:
                logger.info("OCR: imagen excede %d MB, salteada: %s", self._max_bytes // (1024 * 1024), url)
                return self._build_replacement(alt, "")

            # Valida que sea un raster decodable (SVG/corruptos lanzan acá).
            Image.open(BytesIO(content)).verify()

            engine = self._get_engine()
            result, _ = engine(
                content,
                text_score=self._text_score,
                box_thresh=self._box_thresh,
                unclip_ratio=self._unclip_ratio,
            )
            ocr_text = self._ocr_to_text(result)
            if ocr_text:
                logger.debug("OCR ok (%d frags) en %s", ocr_text.count(" · ") + 1, url)
            return self._build_replacement(alt, ocr_text)

        except Exception:
            logger.warning("OCR falló para imagen %s, se descarta", url, exc_info=True)
            return self._build_replacement(alt, "")

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            logger.info("RapidOCR engine cargado (OCR de imágenes activo)")
        return self._engine

    @staticmethod
    def _ocr_to_text(result) -> str:
        """
        Convierte las detecciones sueltas de RapidOCR (list[[box, text, score]]) en un string.
        El text_score ya filtró las lecturas dudosas dentro del engine.

        Reordena en orden de lectura (band-sort): agrupa por banda vertical (~20px) y dentro
        de cada banda ordena por X. Captura el 100% del texto; en layouts multi-columna
        interleava columnas, pero eso no afecta el retrieval (vector + full-text van por
        presencia de tokens, no por orden global).
        """
        if not result:
            return ""

        def key(det):
            box = det[0]                  # 4 esquinas [[x, y], ...]
            x, y = box[0][0], box[0][1]
            return (round(y / 20), x)

        dets = sorted(result, key=key)
        fragments: list[str] = []
        for _, text, _ in dets:
            # Quita caracteres CJK alucinados y normaliza espacios; descarta lo que quede vacío.
            frag = CJK_RE.sub("", str(text))
            frag = re.sub(r"\s{2,}", " ", frag).strip()
            if frag:
                fragments.append(frag)
        return " · ".join(fragments)

    @staticmethod
    def _build_replacement(alt: str, ocr_text: str) -> str:
        """Arma el texto que reemplaza a la imagen. Vacío → se descarta la línea (sin ruido)."""
        alt = (alt or "").strip()
        if alt and FILENAME_ALT_RE.search(alt):
            alt = ""  # nombre de archivo, no es descripción

        if ocr_text and alt:
            return f"[Imagen: {alt}. Texto en pantalla: {ocr_text}]"
        if ocr_text:
            return f"[Imagen con texto: {ocr_text}]"
        if alt:
            return f"[Imagen: {alt}]"
        return ""
