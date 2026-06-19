FROM python:3.12-slim

WORKDIR /app

# Dependencias de sistema: "docling" (build-essential) y OpenCV que usa RapidOCR
# (libgl1, libglib2.0-0) para el OCR de imágenes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
