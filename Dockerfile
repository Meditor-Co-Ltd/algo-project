FROM python:3.9-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    git-lfs \
    curl \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Клонируем репозиторий заново, чтобы LFS сработал (вставь свой репозиторий и токен)
ARG GIT_TOKEN
ARG GIT_REPO=https://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@github.com/Meditor-Co-Ltd/algo-project

RUN git clone ${GIT_REPO} . && git lfs pull

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Проверка модели
RUN ls -la *.pkl* *.gz 2>/dev/null || echo "No .pkl/.gz files found"

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "600", "--preload", "api:app"]
