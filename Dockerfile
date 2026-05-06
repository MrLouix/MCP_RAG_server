# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

# System deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec & install
COPY pyproject.toml ./
RUN pip install --no-cache-dir "uv>=0.5" && \
    uv pip install --system -e ".[dev]" --no-cache

# --- production image ---
FROM python:3.11-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code
COPY mcp_rag/ ./mcp_rag/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md config.example.yaml docker-entrypoint.sh ./

# Default env
ENV PYTHONUNBUFFERED=1 \
    RAG_LOG_LEVEL=INFO \
    RAG_TRANSPORT=stdio \
    TRANSPORT=stdio \
    PYTHONDONTWRITEBYTECODE=1

# Data volumes
VOLUME ["/app/rag_index", "/app/models", "/app/documents"]

EXPOSE 3000

ENTRYPOINT ["./docker-entrypoint.sh"]
