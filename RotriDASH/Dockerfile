# ============================================================
# Rotrix Dashboard — Multi-stage Docker build
# ============================================================
# Build from this directory:
#   docker build -t rotrix-dashboard .
#   docker run -p 8501:8501 --env-file .env rotrix-dashboard
# ============================================================

# ---------- Stage 1: build dependencies ----------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: production image ----------
FROM python:3.10-slim

# Runtime system deps for pymupdf, kaleido (headless Chromium), reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy all app code (context is multi_param_app_clone_db/)
COPY . .

# Use the repo's Streamlit config (theme, logging, etc.)
RUN mkdir -p /root/.streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Railway injects PORT at runtime; default to 8501
ENV PORT=8501

ENTRYPOINT ["sh", "-c", \
    "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
