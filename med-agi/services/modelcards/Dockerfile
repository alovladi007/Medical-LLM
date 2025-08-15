FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 80$(echo $service | cut -c1 | od -An -N1 -i | tr -d " ")

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80$(echo $service | cut -c1 | od -An -N1 -i | tr -d " ")"]