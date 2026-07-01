FROM python:3.12-slim

WORKDIR /app

# Dependencias necesarias para "docling"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

# El entrypoint corre `alembic upgrade head` antes de levantar uvicorn.
# Se invoca con `sh` para no depender del bit +x (en dev el bind-mount pisa permisos).
ENTRYPOINT ["sh", "entrypoint.sh"]
