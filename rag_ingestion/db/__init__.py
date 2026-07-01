"""Capa de base de datos: modelado del esquema (SQLModel) y constantes compartidas.

La tabla de tracking de este servicio vive en su propio schema de Postgres
(`ingestion`) para no pisar las de rag-api (`analytics`), que comparte la misma base.
Las queries en runtime siguen usando psycopg2 crudo; estos modelos son la fuente de
verdad del esquema y la base sobre la que Alembic genera las migraciones.
"""

SCHEMA = "ingestion"

__all__ = ["SCHEMA"]
