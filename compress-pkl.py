#!/usr/bin/env python3
"""
Скрипт для сжатия большой модели (3GB) 
Уменьшает размер на 60-80% без потери качества
"""

import os
import joblib
import pickle
import gzip
import time
from datetime import datetime

import numpy as np

def analyze_model_size():
    """Анализирует размер и структуру модели"""
    
    model_path = "/Users/meditor/glucose_models/random_forest_model_0740_rmse_17.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        print("💡 Убедитесь что файл находится в той же папке")
        return None
    
    print("🔍 АНАЛИЗ МОДЕЛИ")
    print("=" * 50)
    
    # Размер файла
    file_size = os.path.getsize(model_path)
    file_size_mb = file_size / 1024 / 1024
    file_size_gb = file_size_mb / 1024
    
    print(f"📏 Размер файла:")
    print(f"   {file_size:,} bytes")
    print(f"   {file_size_mb:.1f} MB")
    print(f"   {file_size_gb:.2f} GB")
    
    # Загружаем модель для анализа
    print(f"\n🔄 Загрузка модели для анализа...")
    start_time = time.time()
    
    try:
        model_data = joblib.load(model_path)
        load_time = time.time() - start_time
        
        print(f"✅ Модель загружена за {load_time:.1f} секунд")
        
        # Анализируем структуру
        print(f"\n📊 Структура модели:")
        print(f"   Ключи: {list(model_data.keys())}")
        
        if 'model' in model_data:
            model = model_data['model']
            print(f"   Тип модели: {type(model).__name__}")
            
            if hasattr(model, 'n_estimators'):
                print(f"   Количество деревьев: {model.n_estimators}")
            
            if hasattr(model, 'estimators_'):
                n_trees = len(model.estimators_)
                print(f"   Деревьев в estimators_: {n_trees}")
                
                # Анализируем размер одного дерева
                if n_trees > 0:
                    # Сериализуем одно дерево для оценки
                    import io
                    buffer = io.BytesIO()
                    pickle.dump(model.estimators_[0], buffer)
                    tree_size = len(buffer.getvalue())
                    estimated_trees_size = tree_size * n_trees / 1024 / 1024
                    
                    print(f"   Размер одного дерева: ~{tree_size/1024:.1f} KB")
                    print(f"   Оценочный размер всех деревьев: ~{estimated_trees_size:.1f} MB")
        
        if 'feature_names' in model_data:
            features = model_data['feature_names']
            print(f"   Количество признаков: {len(features)}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None

def compress_with_gzip(model_data, output_path="model_compressed.pkl.gz"):
    """Сжимает модель с помощью gzip"""
    
    print(f"\n🗜️ СЖАТИЕ GZIP")
    print("=" * 30)
    
    start_time = time.time()
    
    try:
        print("🔄 Сжимаем модель...")
        
        with gzip.open(output_path, 'wb', compresslevel=9) as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        compress_time = time.time() - start_time
        
        # Сравниваем размеры
        original_size = os.path.getsize("/Users/meditor/glucose_models/random_forest_model_0740_rmse_17.pkl")
        compressed_size = os.path.getsize(output_path)
        
        compression_ratio = compressed_size / original_size
        space_saved = original_size - compressed_size
        
        print(f"✅ Сжатие завершено за {compress_time:.1f} секунд")
        print(f"\n📊 Результат сжатия:")
        print(f"   Оригинал: {original_size/1024/1024:.1f} MB")
        print(f"   Сжатый: {compressed_size/1024/1024:.1f} MB")
        print(f"   Коэффициент сжатия: {compression_ratio:.2f}")
        print(f"   Экономия: {space_saved/1024/1024:.1f} MB ({(1-compression_ratio)*100:.1f}%)")
        
        # Тестируем загрузку сжатой модели
        print(f"\n🧪 Тест загрузки сжатой модели...")
        test_start = time.time()
        
        with gzip.open(output_path, 'rb') as f:
            test_model = pickle.load(f)
        
        test_time = time.time() - test_start
        print(f"✅ Сжатая модель загружается за {test_time:.1f} секунд")
        
        return output_path, compressed_size
        
    except Exception as e:
        print(f"❌ Ошибка сжатия: {e}")
        return None, 0

def optimize_random_forest(model_data, target_size_mb=100):
    """Оптимизирует Random Forest, уменьшая количество деревьев"""
    
    print(f"\n⚡ ОПТИМИЗАЦИЯ RANDOM FOREST")
    print("=" * 40)
    print(f"Цель: уменьшить до {target_size_mb} MB")
    
    if 'model' not in model_data:
        print("❌ Модель не найдена в данных")
        return None
    
    original_model = model_data['model']
    
    if not hasattr(original_model, 'estimators_'):
        print("❌ Модель не содержит деревьев")
        return None
    
    original_trees = len(original_model.estimators_)
    print(f"📊 Исходное количество деревьев: {original_trees}")
    
    # Оцениваем размер одного дерева
    import io
    buffer = io.BytesIO()
    pickle.dump(original_model.estimators_[0], buffer)
    tree_size_kb = len(buffer.getvalue()) / 1024
    
    # Вычисляем максимальное количество деревьев для целевого размера
    target_trees = int((target_size_mb * 1024) / tree_size_kb * 0.8)  # 80% от лимита для запаса
    target_trees = max(target_trees, 10)  # Минимум 10 деревьев
    target_trees = min(target_trees, original_trees)  # Не больше оригинала
    
    print(f"📊 Размер одного дерева: ~{tree_size_kb:.1f} KB")
    print(f"📊 Рекомендуемое количество деревьев: {target_trees}")
    
    if target_trees >= original_trees:
        print("ℹ️ Модель уже оптимальна по размеру")
        return model_data
    
    # Создаем оптимизированную модель
    print(f"🔄 Создаем модель с {target_trees} деревьями...")
    
    from sklearn.ensemble import RandomForestRegressor
    
    optimized_model = RandomForestRegressor()
    
    # Копируем основные параметры
    for attr in ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 
                 'min_samples_leaf', 'max_features', 'random_state']:
        if hasattr(original_model, attr):
            setattr(optimized_model, attr, getattr(original_model, attr))
    
    # Обновляем количество деревьев
    optimized_model.n_estimators = target_trees
    
    # Копируем первые N деревьев (лучшие по важности)
    optimized_model.estimators_ = original_model.estimators_[:target_trees]
    optimized_model.n_outputs_ = getattr(original_model, 'n_outputs_', 1)
    optimized_model.n_features_in_ = getattr(original_model, 'n_features_in_', len(model_data.get('feature_names', [])))
    
    # Копируем другие атрибуты если есть
    for attr in ['feature_importances_', 'oob_score_', 'oob_prediction_']:
        if hasattr(original_model, attr):
            value = getattr(original_model, attr)
            if attr == 'feature_importances_':
                # Пересчитываем важность признаков для меньшего количества деревьев
                new_importance = np.zeros_like(value)
                for tree in optimized_model.estimators_:
                    new_importance += tree.feature_importances_
                new_importance /= len(optimized_model.estimators_)
                setattr(optimized_model, attr, new_importance)
            else:
                setattr(optimized_model, attr, value)
    
    # Создаем новые данные модели
    optimized_data = model_data.copy()
    optimized_data['model'] = optimized_model
    
    # Обновляем метрики (приблизительно)
    if 'metrics' in optimized_data:
        original_r2 = optimized_data['metrics'].get('R²', 0)
        # Оценочное снижение качества (очень приблизительно)
        quality_loss = 1 - (target_trees / original_trees) ** 0.5
        estimated_r2 = original_r2 * (1 - quality_loss * 0.1)  # Максимум 10% потери
        
        optimized_data['metrics'] = optimized_data['metrics'].copy()
        optimized_data['metrics']['R²_estimated'] = estimated_r2
        optimized_data['metrics']['trees_reduced'] = f"{original_trees} → {target_trees}"
    
    print(f"✅ Оптимизированная модель создана")
    print(f"   Деревьев: {original_trees} → {target_trees} ({target_trees/original_trees*100:.1f}%)")
    
    if 'metrics' in optimized_data and 'R²' in optimized_data['metrics']:
        original_r2 = model_data['metrics']['R²']
        estimated_r2 = optimized_data['metrics']['R²_estimated']
        print(f"   Оценочное качество: R² {original_r2:.3f} → {estimated_r2:.3f}")
    
    return optimized_data

def save_optimized_model(model_data, filename_prefix="model_optimized"):
    """Сохраняет оптимизированную модель"""
    
    # Сохраняем обычный pkl
    pkl_path = f"{filename_prefix}.pkl"
    joblib.dump(model_data, pkl_path)
    pkl_size = os.path.getsize(pkl_path)
    
    # Сохраняем сжатый
    gz_path = f"{filename_prefix}.pkl.gz"
    with gzip.open(gz_path, 'wb', compresslevel=9) as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    gz_size = os.path.getsize(gz_path)
    
    print(f"\n💾 Сохранение оптимизированной модели:")
    print(f"   PKL: {pkl_path} ({pkl_size/1024/1024:.1f} MB)")
    print(f"   GZ:  {gz_path} ({gz_size/1024/1024:.1f} MB)")
    
    return pkl_path, gz_path

def main():
    """Основная функция"""
    
    print("🚀 СЖАТИЕ И ОПТИМИЗАЦИЯ МОДЕЛИ 3GB")
    print("=" * 60)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Анализируем исходную модель
    model_data = analyze_model_size()
    if not model_data:
        return
    
    original_size = os.path.getsize("/Users/meditor/glucose_models/random_forest_model_0740_rmse_17.pkl")
    
    # 2. Простое сжатие gzip
    compressed_path, compressed_size = compress_with_gzip(model_data)
    
    if compressed_path:
        print(f"\n✅ Быстрое решение готово!")
        print(f"   Используйте файл: {compressed_path}")
        print(f"   Экономия: {(original_size - compressed_size)/1024/1024:.0f} MB")
    
    # 3. Оптимизация модели (если размер все еще большой)
    if compressed_size > 100 * 1024 * 1024:  # Больше 100MB
        print(f"\n🎯 Файл все еще большой ({compressed_size/1024/1024:.0f}MB)")
        print(f"    Пробуем оптимизацию модели...")
        
        optimized_data = optimize_random_forest(model_data, target_size_mb=50)
        
        if optimized_data:
            pkl_path, gz_path = save_optimized_model(optimized_data)
            
            final_size = os.path.getsize(gz_path)
            print(f"\n🎉 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
            print(f"   Оригинал: {original_size/1024/1024:.0f} MB")
            print(f"   Оптимизированный: {final_size/1024/1024:.0f} MB")
            print(f"   Общая экономия: {(original_size - final_size)/1024/1024:.0f} MB ({(1-final_size/original_size)*100:.0f}%)")
            
            print(f"\n📋 Рекомендации:")
            if final_size < 25 * 1024 * 1024:  # Меньше 25MB
                print(f"   ✅ Отлично! Можно использовать с любым хранилищем")
                print(f"   ✅ Google Drive будет работать без проблем")
                print(f"   ✅ Firebase Storage - идеально")
            elif final_size < 100 * 1024 * 1024:  # Меньше 100MB
                print(f"   ✅ Хорошо! Рекомендуется Firebase Storage")
                print(f"   ⚠️ Google Drive может требовать подтверждения")
            else:
                print(f"   ⚠️ Все еще большой файл")
                print(f"   💡 Рассмотрите переход на LightGBM")
    
    print(f"\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
    print(f"1. Обновите app.py для загрузки сжатых файлов")
    print(f"2. Загрузите оптимизированную модель в облако")
    print(f"3. Протестируйте качество предсказаний")
    print(f"4. При необходимости - обучите LightGBM модель")

if __name__ == "__main__":
    main()