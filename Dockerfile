FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЁ одной командой
COPY . .

# Проверяем что скопировалось
RUN echo "=== Files in /app ===" && ls -la

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "api:app"]