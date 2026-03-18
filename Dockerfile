FROM python:3.12-slim

WORKDIR /app

ARG TORCH_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# Dependencias necesarias para el funcionamiento de "sentence-transformers" y "torch"
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
