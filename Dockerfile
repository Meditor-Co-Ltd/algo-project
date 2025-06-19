# Production Dockerfile for Glucose Prediction API
# Model embedded: 700MB compressed
# Total image size: ~1.2GB
# Startup time: ~10 seconds

FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Добавляем метаданные
LABEL maintainer="glucose-api"
LABEL description="Glucose Prediction API with embedded 700MB model"
LABEL version="4.0.0"

# Обновляем систему и устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# КРИТИЧЕСКИ ВАЖНО: Копируем модель В образ Docker
# Убедитесь что файл model_compressed.pkl.gz (700MB) находится в той же папке что и Dockerfile
COPY model_compressed.pkl.gz .

# Копируем код приложения
COPY app.py .

# Создаем непривилегированного пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Переключаемся на непривилегированного пользователя
USER appuser

# Открываем порт 8000
EXPOSE 8000

# Настройки окружения для оптимизации Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8

# Healthcheck для мониторинга
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Команда запуска
CMD ["python", "app.py"]

# ================================================================================
# ИНФОРМАЦИЯ О ОБРАЗЕ:
# ================================================================================
# Размер образа: ~1.2GB (Python 3.9-slim ~150MB + зависимости ~350MB + модель 700MB)
# Время сборки: 5-10 минут (зависит от интернета)
# Время запуска: ~10 секунд (загрузка модели из диска)
# RAM требования: ~2GB (модель 700MB + Python + буферы)

# ================================================================================
# КОМАНДЫ ДЛЯ ЛОКАЛЬНОЙ РАЗРАБОТКИ:
# ================================================================================

# Сборка образа:
# docker build -t glucose-api:latest .

# Проверка размера:
# docker images glucose-api

# Локальный запуск:
# docker run -p 8000:8000 glucose-api:latest

# Запуск с ограничением памяти:
# docker run -p 8000:8000 --memory=2g glucose-api:latest

# Запуск в фоне:
# docker run -d -p 8000:8000 --name glucose-api glucose-api:latest

# Просмотр логов:
# docker logs glucose-api

# Остановка:
# docker stop glucose-api

# ================================================================================
# ОПТИМИЗИРОВАННЫЙ DOCKERFILE (многоступенчатая сборка):
# ================================================================================

# FROM python:3.9-slim as builder
# 
# WORKDIR /build
# 
# # Устанавливаем зависимости в изолированную папку
# COPY requirements.txt .
# RUN pip install --user --no-cache-dir -r requirements.txt
# 
# # Финальный образ
# FROM python:3.9-slim
# 
# WORKDIR /app
# 
# # Устанавливаем curl для healthcheck
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# 
# # Копируем установленные пакеты из builder этапа
# COPY --from=builder /root/.local /root/.local
# ENV PATH=/root/.local/bin:$PATH
# 
# # Копируем модель и код
# COPY model_compressed.pkl.gz .
# COPY app.py .
# 
# # Создаем пользователя
# RUN groupadd -r appuser && useradd -r -g appuser appuser
# RUN chown -R appuser:appuser /app
# USER appuser
# 
# EXPOSE 8000
# 
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
# 
# CMD ["python", "app.py"]

# ================================================================================
# DOCKER COMPOSE для разработки:
# ================================================================================

# version: '3.8'
# services:
#   glucose-api:
#     build: .
#     ports:
#       - "8000:8000"
#     environment:
#       - PORT=8000
#     volumes:
#       # Для разработки - монтируем только код (НЕ модель!)
#       - ./app.py:/app/app.py:ro
#     restart: unless-stopped
#     mem_limit: 2g
#     healthcheck:
#       test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
#       interval: 30s
#       timeout: 10s
#       retries: 3
#       start_period: 60s

# Для запуска: docker-compose up -d

# ================================================================================
# .dockerignore для оптимизации:
# ================================================================================

# *.pyc
# __pycache__/
# .git/
# .gitignore
# README.md
# tests/
# *.log
# .env
# .DS_Store
# node_modules/
# 
# # Исключаем большие файлы, которые НЕ нужны в образе
# random_forest_model_0740_rmse_17.pkl  # Оригинальная несжатая модель
# *.pkl                                  # Все несжатые модели
# !model_compressed.pkl.gz               # НО включаем нашу сжатую модель
# 
# # Исключаем файлы разработки
# test_*.py
# compress_model.py
# *.ipynb

# ================================================================================
# DEPLOYMENT НА РАЗЛИЧНЫХ ПЛАТФОРМАХ:
# ================================================================================

# RAILWAY:
# 1. railway login
# 2. railway link (или railway init)  
# 3. railway up
# Railway автоматически обнаружит Dockerfile и соберет образ

# RENDER:
# 1. Подключите GitHub репозиторий к Render
# 2. Выберите "Docker" как Build Command
# 3. Render автоматически соберет и разместит образ
# 4. Убедитесь что план поддерживает 2GB RAM

# GOOGLE CLOUD RUN:
# gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/glucose-api
# gcloud run deploy glucose-api \
#   --image gcr.io/YOUR_PROJECT_ID/glucose-api \
#   --platform managed \
#   --memory 2Gi \
#   --cpu 2 \
#   --timeout 300s \
#   --max-instances 10

# HEROKU (с Container Registry):
# heroku container:login
# heroku container:push web --app your-app-name
# heroku container:release web --app your-app-name

# AWS ECS/Fargate:
# 1. Push образ в ECR
# 2. Создайте Task Definition с 2GB памяти
# 3. Создайте ECS Service

# ================================================================================
# МОНИТОРИНГ И ОТЛАДКА:
# ================================================================================

# Подключение к запущенному контейнеру:
# docker exec -it glucose-api /bin/bash

# Проверка использования ресурсов:
# docker stats glucose-api

# Просмотр логов в реальном времени:
# docker logs -f glucose-api

# Проверка health check:
# docker inspect --format='{{.State.Health.Status}}' glucose-api