FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc libmagic-dev poppler-utils tesseract-ocr libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --upgrade pip && pip install --retries 5 -e .

COPY . .

FROM base AS production
RUN addgroup --system app && adduser --system --group --home /home/app app
RUN mkdir -p /images /home/app && chown -R app:app /app /images /home/app

ENV HOME=/home/app
ENV NUMBA_CACHE_DIR=/tmp/numba
ENV MPLCONFIGDIR=/tmp/matplotlib

USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]