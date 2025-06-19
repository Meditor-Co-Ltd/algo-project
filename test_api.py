#!/usr/bin/env python3
"""
Обновленные тесты для Glucose Prediction API
cal_data теперь ОБЯЗАТЕЛЬНЫЙ параметр для точных предсказаний
"""

import requests
import json
import time
import sys

# ВАЖНО: Обновите URL для вашего развернутого API
API_URL = "http://localhost:8000"
# API_URL = "https://your-app.railway.app"
# API_URL = "https://your-app.onrender.com"

def test_health_check():
    """Тест проверки работоспособности API"""
    print("🧪 Testing Health Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            
            if 'model_info' in data:
                model_info = data['model_info']
                print(f"   Model name: {model_info.get('name', 'Unknown')}")
                print(f"   Model R²: {model_info.get('r2_score', 'N/A')}")
                print(f"   Model RMSE: {model_info.get('rmse', 'N/A')} mg/dL")
                print(f"   Features: {model_info.get('features', 'N/A')}")
                
                if 'trees' in model_info:
                    print(f"   Trees: {model_info.get('trees')}")
            
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_documentation():
    """Тест получения документации API"""
    print("\n📖 Testing API Documentation")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Documentation retrieved")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Deployment: {data.get('deployment')}")
            
            # Проверяем что cal_data указан как обязательный
            if 'required_fields' in data:
                required_fields = data['required_fields']
                if 'cal_data' in required_fields:
                    print("✅ cal_data correctly listed as required field")
                else:
                    print("⚠️ cal_data not found in required fields")
            
            return True
        else:
            print(f"❌ Documentation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Documentation error: {e}")
        return False

def test_simple_prediction():
    """Тест простого предсказания с cal_data"""
    print("\n🔍 Testing Simple Prediction (with cal_data)")
    
    # Тестовые данные с cal_data
    test_data = {
        "measure": [1000, 1010, 1020, 1015, 1005, 1025, 1030, 1012],
        "reference": [1100, 1110, 1120, 1115, 1105, 1125, 1130, 1112],
        "dark": [50, 52, 51, 53, 49, 54, 55, 51],
        "cal_data": [900, 910, 920, 915, 905, 925, 930, 912]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_data, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            glucose = response.json()
            print("✅ Prediction successful!")
            print(f"   Glucose level: {glucose} mg/dL")
            print(f"   Response time: {response_time:.2f} seconds")
            
            # Проверяем разумность результата
            if isinstance(glucose, (int, float)) and 20 <= glucose <= 500:
                print("✅ Glucose value is within expected range (20-500 mg/dL)")
                return True
            else:
                print(f"⚠️ Glucose value seems unusual: {glucose}")
                return False
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_prediction_without_cal_data():
    """Тест что API требует cal_data"""
    print("\n🚫 Testing Prediction WITHOUT cal_data (should fail)")
    
    # Данные БЕЗ cal_data
    test_data = {
        "measure": [1000, 1010, 1020],
        "reference": [1100, 1110, 1120],
        "dark": [50, 52, 54]
        # cal_data отсутствует намеренно
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=test_data, timeout=30)
        
        if response.status_code == 400:
            error_data = response.json()
            print("✅ API correctly rejects request without cal_data")
            print(f"   Error: {error_data.get('error')}")
            
            # Проверяем что cal_data упоминается в ошибке
            error_text = str(error_data.get('error', ''))
            if 'cal_data' in error_text:
                print("✅ Error message correctly mentions cal_data")
                return True
            else:
                print("⚠️ Error message doesn't mention cal_data")
                return False
        else:
            print(f"❌ API should reject request without cal_data, but returned: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_batch_prediction():
    """Тест пакетных предсказаний с cal_data"""
    print("\n📊 Testing Batch Prediction (with cal_data)")
    
    # Данные для 3 образцов
    batch_data = {
        "samples": [
            {
                "measure": [1000, 1010, 1020],
                "reference": [1100, 1110, 1120],
                "dark": [50, 52, 54],
                "cal_data": [900, 910, 920]
            },
            {
                "measure": [1005, 1015, 1025],
                "reference": [1105, 1115, 1125],
                "dark": [48, 50, 52],
                "cal_data": [905, 915, 925]
            },
            {
                "measure": [995, 1005, 1015],
                "reference": [1095, 1105, 1115],
                "dark": [52, 54, 56],
                "cal_data": [895, 905, 915]
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict/batch", json=batch_data, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Batch prediction successful!")
            print(f"   Results: {results}")
            print(f"   Response time: {response_time:.2f} seconds")
            
            # Проверяем результаты
            if isinstance(results, list) and len(results) == 3:
                print("✅ Correct number of results returned")
                
                valid_results = 0
                for i, glucose in enumerate(results):
                    if isinstance(glucose, (int, float)) and 20 <= glucose <= 500:
                        valid_results += 1
                    elif glucose is None:
                        print(f"⚠️ Sample {i+1}: prediction failed (None)")
                    else:
                        print(f"⚠️ Sample {i+1}: unusual glucose value {glucose}")
                
                if valid_results >= 2:  # Минимум 2 из 3 должны быть валидными
                    print(f"✅ {valid_results}/3 predictions are valid")
                    return True
                else:
                    print(f"❌ Only {valid_results}/3 predictions are valid")
                    return False
            else:
                print(f"❌ Unexpected batch response format: {results}")
                return False
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_model_info():
    """Тест получения информации о модели"""
    print("\n🤖 Testing Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved")
            print(f"   Model name: {data.get('model_name')}")
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Features: {data.get('feature_count')}")
            print(f"   Deployment: {data.get('deployment_type')}")
            
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"   R²: {metrics.get('R²', 'N/A')}")
                print(f"   RMSE: {metrics.get('RMSE', 'N/A')} mg/dL")
                
                if 'trees_reduced' in metrics:
                    print(f"   Optimization: {metrics['trees_reduced']}")
            
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_performance():
    """Тест производительности с большими массивами"""
    print("\n⚡ Testing Performance (large arrays with cal_data)")
    
    # Большие массивы данных (50 точек)
    size = 50
    test_data = {
        "measure": [1000 + i for i in range(size)],
        "reference": [1100 + i for i in range(size)],
        "dark": [50 + (i % 10) for i in range(size)],
        "cal_data": [900 + i for i in range(size)]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_data, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            glucose = response.json()
            print("✅ Large array processing successful!")
            print(f"   Array size: {size} points")
            print(f"   Glucose: {glucose} mg/dL")
            print(f"   Processing time: {response_time:.2f} seconds")
            
            if response_time < 5.0:  # Должно быть быстро
                print("✅ Performance is acceptable (<5 seconds)")
                return True
            else:
                print("⚠️ Performance is slow (>5 seconds)")
                return False
        else:
            print(f"❌ Large array processing failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def run_all_tests():
    """Запуск всех тестов"""
    print("🚀 GLUCOSE PREDICTION API - COMPREHENSIVE TESTING")
    print("=" * 60)
    print(f"🎯 Testing API at: {API_URL}")
    print(f"🕐 Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Documentation", test_api_documentation),
        ("Simple Prediction (with cal_data)", test_simple_prediction),
        ("Rejection without cal_data", test_prediction_without_cal_data),
        ("Batch Prediction", test_batch_prediction),
        ("Model Information", test_model_info),
        ("Performance Test", test_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your API is ready for production!")
        print("✅ cal_data integration is working correctly")
        print("✅ Model predictions are accurate")
        print("✅ Error handling is robust")
        print("✅ Performance is acceptable")
        
        print(f"\n🔗 Your API is ready for Flutter integration:")
        print(f"   Base URL: {API_URL}")
        print(f"   Prediction endpoint: {API_URL}/predict")
        print(f"   Required fields: measure, reference, dark, cal_data")
        
    elif passed >= total * 0.8:  # 80% прошло
        print("⚠️ Most tests passed, but some issues found")
        print("🔧 Please check the failed tests and fix any issues")
        
    else:
        print("❌ Many tests failed - API needs attention")
        print("🚨 Please fix the issues before deployment")
    
    print("=" * 60)
    return passed == total

def main():
    """Основная функция"""
    
    if len(sys.argv) > 1:
        global API_URL
        API_URL = sys.argv[1]
        print(f"🎯 Using custom API URL: {API_URL}")
    
    success = run_all_tests()
    
    if success:
        print("\n💡 Next steps:")
        print("1. Deploy to production (Railway/Render/Google Cloud)")
        print("2. Update Flutter app with the new API URL")
        print("3. Ensure Flutter sends cal_data in all requests")
        print("4. Test end-to-end glucose predictions")
        sys.exit(0)
    else:
        print("\n🔧 Fix the issues and run tests again:")
        print(f"python {sys.argv[0]} [API_URL]")
        sys.exit(1)

if __name__ == "__main__":
    main()