# stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# stage 2: runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (e.g. for psycopg2, opencv if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source
COPY api/ /app/api/
COPY model/ /app/model/
COPY nlp/ /app/nlp/
COPY requirements.txt /app/
COPY README.md /app/

# Set Python Path to include /app for cross-module imports
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
