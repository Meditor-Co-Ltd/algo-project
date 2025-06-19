FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY . .

# Загрузка модели вручную (замени ссылку на актуальную!)
RUN curl -L https://github.com/Meditor-Co-Ltd/algo-project/releases/tag/tag_1/model_compressed.pkl.gz
 -o model_compressed.pkl.gz

# Проверка
RUN ls -la && echo "Checking for model files:" && ls -la *.pkl* *.gz 2>/dev/null || echo "No .pkl/.gz files found"

# Переменные среды
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Запуск
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "600", "--preload", "api:app"]
