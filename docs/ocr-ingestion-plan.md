# Plan — OCR de imágenes en la ingesta

> Estado: **Fase 1 implementada** (código escrito y validado en sintaxis + regex + API de
> RapidOCR). Pendiente: build de Docker y validación con 1 página real (Fase 2).
> Alcance: solo OCR (sin VLM). Modifica únicamente `rag_ingestion_service`.
>
> Archivos: `ingest/image_ocr.py` (nuevo), `services/ingestion_service.py` (+resolve),
> `config/settings.py` (+flags OCR), `requirements.txt`, `Dockerfile`, `.env.example`.

## 1. Objetivo

Hoy las imágenes de los docs de BookStack se **descartan** durante la limpieza, así que
los pasos-a-paso con screenshots (formularios, menús, botones) no se pueden recuperar
desde el chat. El objetivo es **transcribir el texto de cada imagen vía OCR en tiempo de
ingesta** e inyectarlo como texto plano en el chunk, para que quede **embebido + indexado
full-text** sin tocar `rag-api` ni el query-time.

No se sube ninguna imagen a Qdrant ni al chat: solo su **transcripción en texto**.

## 2. Relevamiento — hechos confirmados en el código

| Hecho | Dónde | Implicancia |
|---|---|---|
| Las imágenes mueren en la limpieza | `cleaner.py:65` (regex fallback borra `![...](...)`) y `cleaner.py:120-124` (Docling → placeholder vacío `image-NN.png`) | Hay que interceptar **antes** de `clean()` |
| El markdown crudo trae las URLs reales | `bookstack.py:65` `get_page_markdown()` → export de BookStack | Las URLs están disponibles antes de limpiar |
| El `content` del chunk va al campo indexado full-text | `db.py:122` (`text` = Título + Contenido) + índice en `db.py:89-99` | El texto OCR **será buscable por keyword** ✓ |
| El `content` también se embebe | `embeddings.py:67-71` | El texto OCR **será buscable por vector** ✓ |
| La ingesta corre en un thread daemon | `ingestion.py:40-46` | `httpx` sync + OCR sync **no bloquean** el event loop ✓ |
| Incrementalidad por hash de markdown crudo | `ingestion_service.py:127` + skip checks `:118`/`:131` | El OCR corre **solo en páginas nuevas/cambiadas** ✓ |
| Soporta ingesta de una sola página | `ingestion_service.py:73,103` (`page_id`) | Permite validar sin tocar toda la colección |

## 3. Formato real de las imágenes (de un export de muestra)

Ejemplo real de `GET {bookstack}/api/pages/{id}/export/markdown`:

```markdown
**[![image3.png](https://docs.frba.utn.edu.ar/uploads/images/gallery/2019-05-May/scaled-840-0/jLLrnAIHY0QsXEGJ-image3.png)](https://docs.frba.utn.edu.ar/uploads/images/gallery/2019-05-May/jLLrnAIHY0QsXEGJ-image3.png)**
```

Características clave (definen el regex):

1. **Linked-image anidada**: el patrón es `[![alt](THUMB_URL)](FULL_URL)`, no un `![]()` simple.
   - `THUMB_URL` lleva `/scaled-840-0/` → miniatura de 840px (la que se muestra).
   - `FULL_URL` es el original sin `scaled-840-0` → **mejor para OCR** (más resolución).
2. **URLs absolutas y públicas** (`https://docs.frba.utn.edu.ar/uploads/images/...`). Sin auth.
3. **El `alt` es solo el nombre de archivo** (`image3.png`) → **inútil como descripción**.
   Hay que filtrarlo como vacío (no inyectar `[Imagen: image3.png]`).
4. Las imágenes pueden estar envueltas en `**...**` u otros markers → el regex extrae solo
   el bloque de imagen; los `**` que rodean quedan y dan `**[Imagen: ...]**` (inofensivo).
5. Mucho `<span style>` y entidades (`&gt;`) alrededor → los maneja el cleaner **después**
   de nuestro paso; no nos afectan porque los patrones de imagen son markdown limpio.

### Variantes a cubrir en el regex (en orden)

```python
# 1. Linked-image (caso real observado): [![alt](thumb)](full) → usar FULL para OCR
LINKED_IMG_RE = re.compile(r"\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)")
# 2. Imagen suelta (fallback): ![alt](url)
BARE_IMG_RE   = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Aplicar LINKED primero (reemplaza todo el bloque), luego BARE sobre el resto.
# Filtrar alt tipo nombre de archivo:
FILENAME_ALT_RE = re.compile(r"^image[-_]?\d*\.\w+$", re.IGNORECASE)
```

> Para linked-images se usa `FULL_URL` (grupo 3, la URL **original** sin `scaled-840-0`):
> es la que siempre existe (el archivo subido). La URL interna `scaled-840-0` es un thumbnail
> que BookStack genera para mostrar, pero usamos la original para no depender de él.

## 4. Componente nuevo: `rag_ingestion/ingest/image_ocr.py`

Clase `ImageTextExtractor`, instanciada una vez en `IngestionService.__init__` (al lado de
`self._cleaner`), con engine **lazy-loaded** (no bloquea el arranque del servicio).

```python
from rapidocr_onnxruntime import RapidOCR
engine = RapidOCR()                      # instanciar una vez (lazy)
# los parámetros van en la LLAMADA (verificado en v1.4.4):
result, elapse = engine(
    img, text_score=0.7, box_thresh=0.5, unclip_ratio=1.6
)
```

> ✅ **Verificado (v1.4.4, Python 3.12)**: `text_score`/`box_thresh`/`unclip_ratio` se pasan
> como **kwargs en `__call__`** (no en el constructor). Output = `list[[box, text, score]]`
> con `box[0] = [x, y]`. Defaults del paquete ya son `box_thresh=0.5`, `unclip_ratio=1.6`;
> solo subimos `text_score` de 0.5 → 0.7. Texto en español reconocido perfecto con los
> modelos default (`ch_PP-OCRv4`).
>
> **Nota de arquitectura**: RapidOCR es una **librería de Python que corre dentro del
> contenedor de `rag_ingestion_service`** (CPU), NO en la VM `10.2.8.25` y NO es un
> servicio de red. Es la única pieza de "inferencia" local al servicio; el LLM, embeddings
> y reranker siguen corriendo remotos en la VM. El OCR no agrega ninguna API HTTP nueva.

### `resolve(markdown) -> markdown`

1. `LINKED_IMG_RE.sub(...)` → por cada match: alt, thumb_url, full_url.
2. `BARE_IMG_RE.sub(...)` → por cada match: alt, url.
3. Por cada imagen (con cache/dedup de URL dentro de la página):
   - `httpx GET` (URL pública, sin auth) con `ocr_download_timeout`.
   - Guard de tamaño (`ocr_max_image_mb`) y de content-type.
   - **Si no es raster (SVG/otros): saltear** (queda solo alt, que acá es vacío → línea fuera).
   - Decode con Pillow → numpy.
   - `result, _ = engine(img)`.
   - `_ocr_to_text(result)` (§5) → string.
   - `_build_replacement(alt, ocr_text)` (§5) → texto `[Imagen: ...]`.

**Robustez**: todo en try/except por imagen. Descarga falla / timeout / no-raster / OCR vacío
→ se descarta la línea. **Nunca tira la página.** Log por imagen.

## 5. Interpretación del output del OCR

RapidOCR devuelve detecciones sueltas `[box, texto, score]` (el `text_score` ya filtró las
de baja confianza dentro del engine). Reordenado **band-sort** (simple, no column-aware):

```python
def _ocr_to_text(result) -> str:
    if not result:
        return ""
    def key(det):
        box = det[0]                       # 4 esquinas [[x,y],...]
        x, y = box[0][0], box[0][1]
        return (round(y / 20), x)          # agrupa por banda vertical ~20px, luego x
    dets = sorted(result, key=key)
    return " · ".join(t.strip() for _, t, _ in dets if t.strip())
```

**Decisión: band-sort simple.** Captura el 100% del texto; en layouts multi-columna
interleava las columnas, pero eso **no afecta el retrieval** (vector + full-text van por
presencia de tokens, no por orden global). El column-aware queda como mejora futura solo si
se observa necesidad real.

### Armado del reemplazo

```python
def _build_replacement(alt, ocr_text):
    alt = "" if (not alt or FILENAME_ALT_RE.match(alt.strip())) else alt.strip()
    if ocr_text and alt:  return f"[Imagen: {alt}. Texto en pantalla: {ocr_text}]"
    if ocr_text:          return f"[Imagen con texto: {ocr_text}]"
    if alt:               return f"[Imagen: {alt}]"
    return ""             # nada útil → línea descartada (no se inyecta ruido)
```

### Parámetros recomendados (justificados con datos reales)

En una prueba real sobre un screenshot de formulario, todo el texto legítimo salió con
score **> 0.85** y la única basura (texto mal leído / reversado) cayó en **0.65**. Por eso:

| Parámetro | Valor | Razón |
|---|---|---|
| `text_score` | `0.7` | Mata la basura (~0.65), conserva lo legítimo (>0.85) |
| `box_thresh` | `0.5` | Default razonable de detección |
| `unclip_ratio` | `1.6` | Agranda un poco la caja → no corta texto pegado a bordes (botones) |

## 6. Punto de inserción: `ingestion_service.py`

Entre línea 139 y 140, **antes** de `clean()`:

```python
version  = tracker.get_version(page.id)
resolved = self._image_resolver.resolve(markdown)   # ← NUEVO
cleaned  = self._cleaner.clean(resolved)
chunks   = self._chunker.chunk_page(page, cleaned, version=version)
```

## 7. Dependencias e infra

**`requirements.txt`** (versión verificada en Python 3.12):
```
rapidocr-onnxruntime==1.4.4
```
ONNX + CPU, sin PyTorch; modelos embebidos en el paquete. Arrastra como deps transitivas
`onnxruntime`, `opencv-python`, `Pillow`, `numpy`, `shapely`, `pyclipper` (no hay que pinarlas
a mano). **No hace falta agregar `Pillow` aparte** — viene con rapidocr.

**`Dockerfile`** — agregar a la línea `apt-get install`:
```
libgl1 libglib2.0-0
```
(OpenCV, que RapidOCR usa por debajo.)

**`settings.py`** — flags ajustables sin redeploy (con defaults → no requiere tocar `.env`):
```python
ocr_enabled: bool          = True
ocr_download_timeout: int  = 15
ocr_text_score: float      = 0.7
ocr_box_thresh: float      = 0.5
ocr_unclip_ratio: float    = 1.6
ocr_max_image_mb: int      = 10
```
Si `ocr_enabled=False`, `resolve()` es no-op (comportamiento actual). Documentar en `.env.example`.

## 8. Decisiones tomadas (cerradas)

- **SVG / diagramas drawio / no-raster**: se **saltean** por ahora (Pillow no los decodifica).
  Quedan sin transcripción. Aceptado.
- **Imágenes públicas**: descarga sin auth. Confirmado.
- **Sin VLM**: no se implementa ningún modelo de visión. El ~20% de imágenes sin texto
  (diagramas, flechas sin labels) queda sin transcribir. Aceptado.
- **Band-sort simple** sobre column-aware. Aceptado.
- **Imágenes que el OCR lee muy mal** (score < `text_score`) se pierden en vez de inyectarse
  como gibberish. Decisión deliberada (mejor perder que ensuciar el vector).

## 9. A verificar / hacer ANTES de implementar

1. ✅ **Firma de RapidOCR — VERIFICADO** (v1.4.4, Python 3.12): kwargs en `__call__`, output
   `list[[box, text, score]]`, español OK con modelos default. Ver §4.
2. **Build del Docker** — pendiente: confirmar que `rapidocr-onnxruntime==1.4.4` instala en la
   imagen y que `opencv-python` levanta con `libgl1` / `libglib2.0-0` agregados. (RapidOCR pulló
   `opencv-python` —no headless— así que esas libs de sistema son necesarias.)

> Resueltos: formato de imagen **siempre** linked-image `[![](thumb)](full)`; OCR sobre la
> **URL original** (grupo 3, sin `scaled-840-0`); firma de la librería verificada.

## 10. Validación (sin riesgo para la colección)

1. Build del Docker con las deps nuevas.
2. Ingesta de **una sola página** con imágenes vía `page_id` (`ingestion_service.py:103`).
3. Verificar en Qdrant que el chunk contiene el texto `[Imagen: ...]`.
4. Probar en el chat una query que dependa del texto de una imagen.

## 11. Riesgos asumidos

- Ingesta `force=true` completa tarda más (~100-300 ms/imagen, CPU, en background → no afecta
  usuarios).
- +~300-500 MB RAM en `rag_ingestion_service` con el engine cargado (no en la VM de inferencia).
- Imágenes muy densas de texto → bloque grande unido por `·`; el `SentenceSplitter` puede
  partirlo de forma rara. No rompe nada.
