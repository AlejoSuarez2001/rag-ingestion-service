"""
Singleton de IngestionService compartido por toda la app.
Se inicializa en el lifespan de main.py.
"""
from typing import Optional
from rag_ingestion.services.ingestion_service import IngestionService

ingestion_service: Optional[IngestionService] = None
