# Slim image, pinned Python, CPU-only wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs libgomp for OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY models ./models

ENV MODEL_BUNDLE=/srv/models/bundle.joblib \
    PORT=8000

EXPOSE 8000

# Single worker: keeps the ~3 MB model loaded once, per-process cache stays hot.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
