#!/usr/bin/env python3
"""
Простой тест загрузки модели с Google Drive
Запустите этот скрипт отдельно для диагностики проблем
"""

import os
import requests
import joblib
import time

def test_model_download():
    """Тестирует только загрузку и проверку модели"""
    
    print("🔍 ТЕСТ ЗАГРУЗКИ МОДЕЛИ С GOOGLE DRIVE")
    print("=" * 50)
    
    # Настройки
    file_id = "1CqaltVpa7azklS_orDxzxg6Oj3Pgv_33"
    model_path = "test_model.pkl"
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    print(f"📋 FILE_ID: {file_id}")
    print(f"📋 Download URL: {download_url}")
    print(f"📋 Local path: {model_path}")
    
    # Удаляем старый файл если есть
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"🗑️ Removed existing file: {model_path}")
    
    try:
        print(f"\n📥 Начинаем загрузку...")
        start_time = time.time()
        
        # Первый запрос для проверки
        print("🔍 Проверяем доступность URL...")
        head_response = requests.head(download_url, timeout=30)
        print(f"   Status: {head_response.status_code}")
        print(f"   Headers:")
        
        important_headers = ['content-type', 'content-length', 'content-disposition']
        for header in important_headers:
            value = head_response.headers.get(header, 'Not found')
            print(f"     {header}: {value}")
        
        # Основная загрузка
        print(f"\n📥 Загружаем файл...")
        response = requests.get(download_url, stream=True, timeout=180)
        response.raise_for_status()
        
        print(f"   Response status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
        
        # Проверяем на HTML ответ
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            print("❌ ОШИБКА: Google Drive вернул HTML вместо файла!")
            print("   Возможные причины:")
            print("   - Файл слишком большой (>25MB)")
            print("   - Требуется подтверждение вирусной проверки")
            print("   - Неправильные права доступа")
            return False
        
        # Сохраняем файл
        total_size = 0
        chunk_count = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    chunk_count += 1
                    
                    # Показываем прогресс
                    if chunk_count % 100 == 0:
                        mb_size = total_size / 1024 / 1024
                        print(f"     📊 Загружено: {mb_size:.1f} MB")
        
        download_time = time.time() - start_time
        mb_size = total_size / 1024 / 1024
        
        print(f"✅ Загрузка завершена!")
        print(f"   Размер: {total_size:,} bytes ({mb_size:.1f} MB)")
        print(f"   Время: {download_time:.1f} секунд")
        print(f"   Скорость: {mb_size/download_time:.1f} MB/s")
        
        # Проверяем размер
        if total_size < 1000:
            print("❌ ОШИБКА: Файл слишком маленький!")
            print("   Вероятно, это страница ошибки Google Drive")
            
            # Покажем содержимое
            with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(500)
                print(f"   Содержимое файла:")
                print(f"   {content[:200]}...")
            return False
        
        # Проверяем первые байты
        print(f"\n🔍 Проверяем содержимое файла...")
        with open(model_path, 'rb') as f:
            first_bytes = f.read(100)
            print(f"   Первые байты: {first_bytes[:20]}...")
            
            # Проверяем на HTML
            f.seek(0)
            try:
                text_content = f.read(200).decode('utf-8', errors='ignore')
                if text_content.strip().startswith(('<!DOCTYPE', '<html', '<HTML')):
                    print("❌ ОШИБКА: Файл содержит HTML!")
                    print(f"   Начало: {text_content[:100]}...")
                    return False
                else:
                    print("✅ Файл содержит бинарные данные (не HTML)")
            except:
                print("✅ Файл содержит бинарные данные")
        
        # Пытаемся загрузить модель
        print(f"\n🤖 Проверяем загрузку модели...")
        try:
            model_data = joblib.load(model_path)
            print("✅ Модель успешно загружена!")
            
            # Показываем информацию о модели
            print(f"   Ключи в модели: {list(model_data.keys())}")
            
            if 'model_name' in model_data:
                print(f"   Имя модели: {model_data['model_name']}")
            
            if 'model' in model_data:
                model = model_data['model']
                print(f"   Тип модели: {type(model).__name__}")
                if hasattr(model, 'n_estimators'):
                    print(f"   n_estimators: {model.n_estimators}")
            
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                print(f"   Метрики: {metrics}")
            
            if 'feature_names' in model_data:
                features = model_data['feature_names']
                print(f"   Количество признаков: {len(features)}")
                print(f"   Первые 3 признака: {features[:3]}")
            
            print(f"\n🎉 УСПЕХ! Модель готова к использованию.")
            return True
            
        except Exception as e:
            print(f"❌ ОШИБКА при загрузке модели:")
            print(f"   {type(e).__name__}: {e}")
            
            # Дополнительная диагностика
            file_size = os.path.getsize(model_path)
            print(f"   Размер файла: {file_size} bytes")
            
            if file_size > 0:
                print("   💡 Файл скачался, но поврежден или не является моделью joblib")
                print("   Попробуйте:")
                print("   1. Проверить файл модели на Google Drive")
                print("   2. Загрузить файл вручную из браузера")
                print("   3. Проверить совместимость версий joblib/scikit-learn")
            
            return False
    
    except requests.exceptions.Timeout:
        print("❌ ОШИБКА: Превышено время ожидания")
        print("   Попробуйте позже или проверьте интернет соединение")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ОШИБКА сети: {e}")
        return False
        
    except Exception as e:
        print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        return False
    
    finally:
        # Очистка
        if os.path.exists(model_path):
            print(f"\n🗑️ Удаляем тестовый файл: {model_path}")
            os.remove(model_path)

def test_alternative_methods():
    """Тестирует альтернативные способы загрузки"""
    
    print(f"\n🔄 ТЕСТИРОВАНИЕ АЛЬТЕРНАТИВНЫХ МЕТОДОВ")
    print("=" * 50)
    
    file_id = "1CqaltVpa7azklS_orDxzxg6Oj3Pgv_33"
    
    # Альтернативные URL
    alternative_urls = [
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.google.com/uc?id={file_id}",
        f"https://docs.google.com/uc?export=download&id={file_id}",
    ]
    
    for i, url in enumerate(alternative_urls, 1):
        print(f"\n{i}. Тестируем: {url}")
        
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"   Content-Length: {response.headers.get('content-length', 'Unknown')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                if 'application' in content_type or 'octet-stream' in content_type:
                    print(f"   ✅ Этот URL может работать")
                else:
                    print(f"   ⚠️ Подозрительный Content-Type")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

def main():
    """Основная функция"""
    
    print("🚀 Запуск теста загрузки модели...")
    print("   Этот скрипт проверит загрузку с Google Drive отдельно от API")
    print()
    
    # Основной тест
    success = test_model_download()
    
    if not success:
        # Пробуем альтернативные методы
        test_alternative_methods()
        
        print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ РЕШЕНИЯ ПРОБЛЕМ:")
        print("1. 🔗 Проверьте ссылку в браузере:")
        print("   https://drive.google.com/uc?id=1CqaltVpa7azklS_orDxzxg6Oj3Pgv_33&export=download")
        print()
        print("2. 🔐 Убедитесь что файл публичный:")
        print("   - Откройте файл в Google Drive")
        print("   - Нажмите 'Поделиться'")
        print("   - Выберите 'Доступ для всех, у кого есть ссылка'")
        print()
        print("3. 📁 Проверьте размер файла:")
        print("   - Если файл >25MB, Google Drive может требовать подтверждения")
        print("   - Попробуйте сжать файл или использовать другой сервис")
        print()
        print("4. 🔄 Попробуйте загрузить файл заново:")
        print("   - Возможно файл поврежден при загрузке в Drive")
        print()
        print("5. 🛠️ Альтернативы:")
        print("   - Загрузите файл на Dropbox или другой сервис")
        print("   - Используйте GitHub LFS")
        print("   - Разместите файл на своем сервере")
    
    else:
        print(f"\n🎉 ВСЕ ОТЛИЧНО!")
        print("   Загрузка модели работает корректно.")
        print("   Теперь можно запускать основной API.")

if __name__ == "__main__":
    main()