# Complaint Intelligence API - container image (CPU only).
# Models, the regulation index and logs are NOT baked in: they are mounted from your project folder at run time
# (see docker-compose.yml), so the image stays small and retraining never needs a rebuild.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# 1. dependencies first (this layer is cached until the version files change)
COPY docker/torch.txt docker/requirements-serve.txt docker/
RUN pip install -r docker/torch.txt --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r docker/requirements-serve.txt

# 2. our code
COPY src/ src/

# run as a normal user, not root
RUN useradd --create-home app && mkdir -p /app/models /app/logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "cfpb.api:app", "--host", "0.0.0.0", "--port", "8000"]
