FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs && git lfs install

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Simple check for model files
RUN ls -la && echo "Checking for model files:" && ls -la *.pkl* *.gz 2>/dev/null || echo "No .pkl/.gz files found"

# Pull LFS files after clone
RUN git lfs pull

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "600", "--preload", "api:app"]