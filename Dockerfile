# ── Build stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ────────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user
RUN groupadd -r nestor && useradd -r -g nestor -m nestor

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Data volume (SQLite DB, Google tokens, etc.)
VOLUME /data
RUN mkdir -p /data && chown nestor:nestor /data

# Config via environment
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

USER nestor

CMD ["python", "main.py"]
