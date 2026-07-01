"""Modelos SQLModel — la fuente de verdad del esquema de la base.

Mapea la tabla `ingestion.pages` que antes se auto-creaba en PageTracker. No se usa
para consultar (eso sigue en psycopg2 crudo); existe para que Alembic versione el
esquema y autogenere migraciones a partir de su diff.

Para evolucionar el esquema: cambiá el modelo acá y corré
`alembic revision --autogenerate -m "descripcion"` seguido de `alembic upgrade head`.
"""

from sqlalchemy import Column, Integer, Text, text
from sqlmodel import Field, SQLModel

from rag_ingestion.db import SCHEMA


class Page(SQLModel, table=True):
    """Tracking de páginas de BookStack ingestadas, para re-ingesta incremental.

    page_id es el id de BookStack (no autoincremental): por eso es PK explícita.
    """

    __tablename__ = "pages"
    __table_args__ = {"schema": SCHEMA}

    page_id: int = Field(
        sa_column=Column(Integer, primary_key=True, autoincrement=False),
    )
    updated_at: str = Field(sa_column=Column(Text, nullable=False))
    content_hash: str = Field(sa_column=Column(Text, nullable=False))
    version: int = Field(
        default=1,
        sa_column=Column(Integer, nullable=False, server_default=text("1")),
    )
    chunk_count: int = Field(
        default=0,
        sa_column=Column(Integer, nullable=False, server_default=text("0")),
    )
    ingested_at: str = Field(sa_column=Column(Text, nullable=False))
