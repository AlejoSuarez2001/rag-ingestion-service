"""
RAG Ingestion Service — API

Endpoints:
    POST /ingestion/full          — lanza ingestion completa en background
    GET  /ingestion/status        — estado del último job
"""
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_ingestion.config.settings import get_settings
from rag_ingestion.api.routes.ingestion import router as ingestion_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import rag_ingestion.api.deps as deps
    from rag_ingestion.services.ingestion_service import IngestionService

    settings = get_settings()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    logger.info("Inicializando IngestionService...")
    deps.ingestion_service = IngestionService(settings)
    logger.info("IngestionService listo.")
    yield


app = FastAPI(
    title="RAG Ingestion Service",
    description="Ingesta páginas de BookStack en Qdrant para el pipeline RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)

