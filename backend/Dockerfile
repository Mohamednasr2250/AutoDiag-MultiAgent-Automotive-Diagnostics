# ── Stage 1: Builder ──────────────────────────────────────
# Separate build stage — installs all deps
# Final image only copies what's needed → smaller image
FROM python:3.10-slim AS builder

# Build-time args
ARG PYTHON_VERSION=3.10
ARG ENV=production

# Labels for metadata
LABEL maintainer="AutoDiag Pro"
LABEL version="2.0.0"
LABEL description="Multi-Agent Automotive Diagnostic Platform"

WORKDIR /app

# ENV for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — layer caching
COPY requirements.txt .

# BuildKit cache mount — dramatically faster rebuilds
# pip cache reused across builds instead of re-downloading
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────
# Separate runtime stage — only what's needed to run
FROM python:3.10-slim AS runtime

WORKDIR /app

# Same ENV in runtime stage
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

# Copy only installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# System deps needed at runtime only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Running as root inside container is a security risk
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

EXPOSE 80

# HEALTHCHECK — Docker knows if container is actually healthy
# Hits /health endpoint every 30s
# 3 failures → container marked unhealthy → orchestrator can restart it
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]