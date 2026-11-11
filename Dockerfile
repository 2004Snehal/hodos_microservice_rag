# Hodos RAG microservice
#   docker compose up --build          (API + Redis; LLM via Ollama on the host)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    HF_HUB_DISABLE_TELEMETRY=1

# Tesseract for OCR of printed bills (English language data).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the local models into the image so the container never downloads at
# runtime: bge-small (embeddings, semantic cache) and ResNet-50 (bill router CNN).
RUN python -c "from fastembed import TextEmbedding, ImageEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); ImageEmbedding('Qdrant/resnet50-onnx')"

COPY app ./app
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 hodos \
    && mkdir -p app/data/policies app/data/vector_store app/data/probabilistic \
    && chown -R hodos:hodos /app/app/data /opt/models
USER hodos

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# One worker: ingested RAPTOR trees and the L1 cache live in process memory.
# Scale with more containers behind a load balancer, not more workers per container
# (see README — trees are persisted to the shared data volume and reload on start).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
