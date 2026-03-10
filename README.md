# RAG Ingestion Service

Servicio de ingestión de documentación desde [BookStack](https://www.bookstackapp.com/) hacia [Qdrant](https://qdrant.tech/) para alimentar el sistema **hybrid-rag-backend**.

## Stack

| Componente | Tecnología |
|---|---|
| API Source | BookStack REST API |
| Parsing / Limpieza | Docling |
| Chunking | LlamaIndex (`MarkdownNodeParser` + `SentenceSplitter`) |
| Embeddings | Sentence Transformers |
| Vector DB | Qdrant (misma instancia que hybrid-rag-backend) |
| State Tracking | SQLite (local) |
| CLI | Typer + Rich |

---

## Estructura del proyecto

```
rag_ingestion_service/
├── main.py                        # CLI principal (ingest / status / stats)
├── rag_ingestion/
│   ├── config/
│   │   └── settings.py            # Variables de entorno (pydantic-settings)
│   └── ingest/
│       ├── models.py              # Dataclass BookStackPage
│       ├── bookstack.py           # Cliente HTTP BookStack API
│       ├── cleaner.py             # Limpieza de markdown con Docling
│       ├── chunking.py            # Chunking con LlamaIndex
│       ├── embeddings.py          # Embeddings con Sentence Transformers
│       └── db.py                  # QdrantStore + PageTracker (SQLite)
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## DER Vectorial

### Colección Qdrant: `tech_manuals`

Cada punto almacena el siguiente payload:

```json
{
  "chunk_id": "123_2",
  "position": 2,
  "page_id": 123,
  "title": "Crear usuario",
  "book": "Manual de Usuario",
  "chapter": "Gestión de usuarios",
  "page": "Usuarios FRBA",
  "content": "...",
  "url": "https://docs.example.com/books/manual/page/usuarios-frba",
  "tokens": 320,
  "version": 1,
  "text": "...",     // alias de content (compatibilidad hybrid-rag-backend)
  "source": "..."    // alias de url (compatibilidad hybrid-rag-backend)
}
```

### Tabla SQLite: `pages`

```sql
page_id     INTEGER PK
updated_at  TEXT
content_hash TEXT      -- MD5 del markdown original
version     INTEGER    -- se incrementa en cada re-ingestión
chunk_count INTEGER
ingested_at TEXT
```

---

## Flujo de ingestión

```
BookStack API
     │ markdown
     ▼
DoclingCleaner         ← limpia HTML, callouts, normaliza formato
     │ cleaned_markdown
     ▼
MarkdownNodeParser     ← split prioritario por headings (LlamaIndex)
     │ heading_nodes
     ▼
SentenceSplitter       ← si node > 500 tokens → sub-chunks con overlap 100
     │ chunks (≤500 tokens cada uno)
     ▼
EmbeddingService       ← sentence-transformers (batch, normalized)
     │ (chunk, vector)[]
     ▼
QdrantStore            ← delete_page_chunks() + upsert_chunks()
     │
     ▼
PageTracker (SQLite)   ← guarda hash + version + chunk_count
```

---

## Configuración de embeddings ⚠️

> **IMPORTANTE:** El modelo de embeddings usado aquí debe coincidir
> con el configurado en `hybrid-rag-backend` para que la búsqueda vectorial funcione.

| Servicio | Variable | Valor recomendado | Dimensión |
|---|---|---|---|
| **Este servicio** | `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | 768 |
| **hybrid-rag-backend** | `OLLAMA_EMBED_MODEL` | mismo modelo vía Ollama, o actualizar a sentence-transformers | 768 |

---

## Inicio rápido

```bash
# 1. Configurar entorno
cp .env.example .env
# Editar .env con tus credenciales de BookStack y Qdrant

# 2. Instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Ingerir toda la documentación
python main.py ingest

# 4. Solo re-ingerir una página específica
python main.py ingest --page-id 42

# 5. Forzar re-ingestión completa (ignora caché)
python main.py ingest --force

# 6. Ver estado de ingestión
python main.py status

# 7. Ver estadísticas de la colección en Qdrant
python main.py stats
```

---

## Ejecución con Docker

```bash
docker build -t rag-ingestion .

# Ingestión completa
docker run --env-file .env --network host rag-ingestion ingest

# Re-ingestión forzada
docker run --env-file .env --network host rag-ingestion ingest --force
```

---

## Extensión futura: job de sincronización

El `PageTracker` (SQLite) ya almacena el `content_hash` y `updated_at` de cada página.
Para implementar el job de detección de cambios:

```python
# Pseudocódigo del job (se puede agregar como main.py sync)
for page in bookstack.get_all_pages():
    page_hash = content_hash(page.content_markdown)
    if tracker.needs_reingestion(page.id, page.updated_at, page_hash):
        ingest_single_page(page)  # delete + re-ingest
```

Se puede programar con **cron**, **Celery Beat**, o un **APScheduler** dentro del mismo proceso.

---

## Créditos de arquitectura

Este servicio es independiente de `hybrid-rag-backend` y comparte únicamente la instancia de Qdrant. El contrato entre ambos es el schema de payload almacenado en cada punto vectorial.
