FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies first (for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding the large model file)
COPY api.py .
COPY *.py .

# Download the large model file from GitHub Releases
# Replace YOUR_USERNAME and YOUR_REPO with actual values
# Replace tag_1 with your actual release tag
RUN echo "📥 Downloading 700MB model file..." && \
    curl -L -o model_compressed.pkl.gz \
    "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/tag_1/model_compressed.pkl.gz" && \
    echo "✅ Model download completed" && \
    ls -lh model_compressed.pkl.gz

# Verify model file
RUN echo "🔍 Verifying model file:" && \
    ls -la *.pkl* *.gz 2>/dev/null || echo "No model files found!" && \
    echo "📏 Model file size:" && \
    du -h model_compressed.pkl.gz

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Increase memory limits for model loading
ENV PYTHONHASHSEED=random

EXPOSE 8000

# Use longer timeout for large model loading (10 minutes)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "600", "--preload", "--max-requests", "1000", "--max-requests-jitter", "100", "api:app"]