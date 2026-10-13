# Aqar.ai API - production image (Render "Docker" runtime, or any container host).
# Lean by design: no PyTorch / OCR / CV stack (see requirements-ingest.txt), so
# the image stays small and the process fits a 512 MB free instance.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Dependencies first, so code-only changes reuse the cached layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts

# Never run as root; the app only needs to read its own files (and, outside
# production, write the local SQLite development database under data/).
RUN useradd --create-home --uid 10001 aqar \
    && mkdir -p /app/data \
    && chown -R aqar:aqar /app/data
USER aqar

EXPOSE 8000

# Liveness only (no dependencies are touched).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)"

# sh, not ./start.sh: the executable bit does not survive every checkout.
CMD ["sh", "scripts/start.sh"]
