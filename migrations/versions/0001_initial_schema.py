"""initial schema: ingestion.pages

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-30

Crea el schema propio del servicio de ingesta y la tabla `pages` que antes se
auto-creaba en PageTracker. De acá en más el esquema se evoluciona con nuevas
revisiones de Alembic, no con CREATE TABLE IF NOT EXISTS.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SCHEMA = "ingestion"


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    op.create_table(
        "pages",
        sa.Column("page_id", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("ingested_at", sa.Text(), nullable=False),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("pages", schema=SCHEMA)
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
