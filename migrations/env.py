"""Entorno de Alembic para rag_ingestion_service.

La URL de la base y el schema se toman de rag_ingestion.config.settings.Settings
(una sola fuente de verdad). El target_metadata es el de SQLModel, así que
autogenerate ve los modelos de rag_ingestion/db/models.py. Se crea el schema
`ingestion` antes de migrar para que las tablas (y la tabla de versiones de Alembic)
puedan vivir ahí.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text
from sqlmodel import SQLModel

from rag_ingestion.config.settings import get_settings
from rag_ingestion.db import SCHEMA
import rag_ingestion.db.models  # noqa: F401  (registra las tablas en SQLModel.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

target_metadata = SQLModel.metadata


def _include_object(obj, name, type_, reflected, compare_to):
    # Solo gestionamos objetos de nuestro schema; ignoramos lo de rag-api (analytics).
    if type_ == "table" and getattr(obj, "schema", None) not in (SCHEMA, None):
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        version_table_schema=SCHEMA,
        include_object=_include_object,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # El schema debe existir antes de que Alembic cree su tabla de versiones ahí.
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema=SCHEMA,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
