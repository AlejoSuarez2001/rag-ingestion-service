#!/bin/sh
# Corre las migraciones pendientes y recién después levanta la app. Separa
# "evolucionar el esquema" de "arrancar el servicio". Idempotente: si no hay
# migraciones pendientes, `upgrade head` no hace nada.
set -e

echo "Aplicando migraciones (alembic upgrade head)..."
alembic upgrade head

echo "Iniciando uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8001 --reload
