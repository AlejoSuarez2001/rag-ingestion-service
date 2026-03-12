import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, status

from rag_ingestion.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


def get_ingestion_service() -> IngestionService:
    from rag_ingestion.api.deps import ingestion_service
    return ingestion_service


@router.post(
    "/full",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lanzar ingestion completa de BookStack",
)
async def full_ingestion(
    force: bool = Query(False, description="Re-ingestar todas las páginas ignorando cambios"),
    service: IngestionService = Depends(get_ingestion_service),
):
    """
    Inicia el pipeline completo: BookStack → clean → chunk → embed → Qdrant.
    Retorna 202 inmediatamente. Consultá GET /ingestion/status para el progreso.
    """
    if service.is_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya hay una ingestion en curso. Consultá /ingestion/status.",
        )

    thread = threading.Thread(
        target=service.run_full_ingestion,
        args=(force, None),
        daemon=True,
        name="ingestion-worker",
    )
    thread.start()

    return {"message": "Ingestion iniciada en background.", "force": force}


@router.get(
    "/status",
    summary="Estado del último job de ingestion",
)
async def ingestion_status(
    service: IngestionService = Depends(get_ingestion_service),
):
    """Devuelve el estado actual (idle / running / completed / failed) y estadísticas."""
    return service.job.to_dict()
