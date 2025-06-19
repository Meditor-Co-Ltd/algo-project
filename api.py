from flask import Flask, request, jsonify
import numpy as np
import joblib
import json
import logging
from datetime import datetime
import os
import traceback
import gzip
import pickle
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the loaded model
model_data = None

def load_compressed_model(file_path):
    """Загружает сжатую модель (.pkl.gz)"""
    try:
        logger.info(f"📂 Loading compressed model from: {file_path}")
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load compressed model: {e}")
        return None

def load_model_on_startup():
    """
    Загрузка модели ТОЛЬКО из локальных файлов
    Никаких загрузок из интернета!
    """
    global model_data
    
    logger.info("🤖 Starting LOCAL model loading...")
    
    # Список файлов в порядке приоритета
    model_files = [
        "model_compressed.pkl.gz",                    # Ваш сжатый файл (приоритет)
        "model_optimized.pkl.gz",                     # Оптимизированная версия
        "random_forest_model_0740_rmse_17.pkl.gz",   # Альтернативное имя
        "model.pkl.gz",                               # Общее имя
        "random_forest_model_0740_rmse_17.pkl",      # Несжатая версия (запасной)
        "model.pkl"                                   # Альтернативная несжатая
    ]
    
    # Показываем содержимое папки для диагностики
    logger.info("📁 Files in current directory:")
    try:
        files_in_dir = os.listdir('.')
        for file in sorted(files_in_dir):
            if os.path.isfile(file):
                size = os.path.getsize(file)
                size_mb = size / 1024 / 1024
                if file.endswith(('.pkl', '.gz')) or size > 1024*1024:  # Показываем модели или файлы >1MB
                    logger.info(f"   📄 {file} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"⚠️ Could not list directory: {e}")
    
    # Пробуем загрузить файлы по порядку
    for model_file in model_files:
        if os.path.exists(model_file):
            logger.info(f"✅ Found model file: {model_file}")
            
            try:
                # Проверяем размер файла
                file_size = os.path.getsize(model_file)
                file_size_mb = file_size / 1024 / 1024
                logger.info(f"📏 File size: {file_size:,} bytes ({file_size_mb:.1f} MB)")
                
                # Загружаем модель
                start_time = time.time()
                
                if model_file.endswith('.gz'):
                    logger.info("🗜️ Loading compressed model...")
                    model_data = load_compressed_model(model_file)
                else:
                    logger.info("📄 Loading uncompressed model...")
                    model_data = joblib.load(model_file)
                
                load_time = time.time() - start_time
                
                if model_data is not None:
                    logger.info(f"🎉 Model loaded successfully in {load_time:.1f} seconds!")
                    
                    # Показываем информацию о модели
                    logger.info(f"   Model name: {model_data.get('model_name', 'Unknown')}")
                    
                    if 'metrics' in model_data:
                        metrics = model_data['metrics']
                        r2 = metrics.get('R²', 'N/A')
                        rmse = metrics.get('RMSE', 'N/A')
                        logger.info(f"   Model R²: {r2}")
                        logger.info(f"   Model RMSE: {rmse} mg/dL")
                        
                        # Информация об оптимизации
                        if 'trees_reduced' in metrics:
                            logger.info(f"   Trees optimization: {metrics['trees_reduced']}")
                    
                    if 'feature_names' in model_data:
                        feature_count = len(model_data['feature_names'])
                        logger.info(f"   Features: {feature_count}")
                    
                    # Проверяем структуру модели
                    if 'model' in model_data:
                        model = model_data['model']
                        model_type = type(model).__name__
                        logger.info(f"   Model type: {model_type}")
                        
                        if hasattr(model, 'n_estimators'):
                            logger.info(f"   Trees: {model.n_estimators}")
                    
                    logger.info("✅ Model is ready for predictions!")
                    return True
                else:
                    logger.error(f"❌ Model data is None after loading {model_file}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                continue
        else:
            logger.debug(f"⏭️ File not found: {model_file}")
    
    # Если ничего не нашли
    logger.error("❌ No model files found!")
    logger.error("💡 Make sure you have one of these files in the project directory:")
    for model_file in model_files[:3]:  # Показываем топ-3
        logger.error(f"   - {model_file}")
    
    logger.error("🔧 To fix this:")
    logger.error("   1. Copy your model_compressed.pkl.gz to the project directory")
    logger.error("   2. Or update the model_files list with your file name")
    logger.error("   3. Make sure the file is not corrupted")
    
    return False

def calculate_absorbance_features(measure, reference, dark, cal_data):
    """Calculate absorbance features for one sample - cal_data is required"""
    try:
        # Convert to numpy arrays
        measure = np.array(measure)
        reference = np.array(reference)
        dark = np.array(dark)
        cal_data = np.array(cal_data)  # Теперь обязательно
        
        # Dark signal correction
        measure_corrected = np.maximum(measure - dark, 1e-10)
        reference_corrected = np.maximum(reference - dark, 1e-10)
        
        # Transmittance
        transmittance = np.maximum(measure_corrected / reference_corrected, 1e-6)
        
        # Absorbance
        absorbance = -np.log10(transmittance)
        
        # Derivatives
        first_deriv = np.gradient(absorbance)
        second_deriv = np.gradient(first_deriv)
        
        # Feature extraction
        n_points = len(absorbance)
        quarter_size = n_points // 4
        
        features = {
            # Basic statistical features
            'abs_mean': np.mean(absorbance),
            'abs_std': np.std(absorbance),
            'abs_max': np.max(absorbance),
            'abs_min': np.min(absorbance),
            'abs_median': np.median(absorbance),
            'abs_range': np.max(absorbance) - np.min(absorbance),
            'abs_q25': np.percentile(absorbance, 25),
            'abs_q75': np.percentile(absorbance, 75),
            'abs_iqr': np.percentile(absorbance, 75) - np.percentile(absorbance, 25),
            
            # Integral characteristics
            'abs_area': np.trapz(absorbance),
            'abs_area_positive': np.trapz(np.maximum(absorbance, 0)),
            'abs_area_negative': np.trapz(np.minimum(absorbance, 0)),
            
            # Spectral ranges
            'abs_quarter1': np.mean(absorbance[:quarter_size]),
            'abs_quarter2': np.mean(absorbance[quarter_size:2*quarter_size]),
            'abs_quarter3': np.mean(absorbance[2*quarter_size:3*quarter_size]),
            'abs_quarter4': np.mean(absorbance[3*quarter_size:]),
            
            # Derivative features
            'abs_deriv1_mean': np.mean(np.abs(first_deriv)),
            'abs_deriv1_std': np.std(first_deriv),
            'abs_deriv2_mean': np.mean(np.abs(second_deriv)),
            'abs_deriv2_std': np.std(second_deriv),
            
            # Gradients between ranges
            'abs_gradient_start_end': (absorbance[-1] - absorbance[0]) / len(absorbance),
            'abs_gradient_q1_q4': np.mean(absorbance[3*quarter_size:]) - np.mean(absorbance[:quarter_size]),
            
            # Peaks and valleys
            'abs_n_peaks': len([i for i in range(1, len(absorbance)-1) 
                              if absorbance[i] > absorbance[i-1] and absorbance[i] > absorbance[i+1]]),
            'abs_n_valleys': len([i for i in range(1, len(absorbance)-1) 
                                if absorbance[i] < absorbance[i-1] and absorbance[i] < absorbance[i+1]]),
            
            # Additional raw features
            'measure_mean': np.mean(measure),
            'measure_std': np.std(measure),
            'reference_mean': np.mean(reference),
            'reference_std': np.std(reference),
            'dark_mean': np.mean(dark),
            'signal_to_noise': np.mean(measure) / np.std(dark) if np.std(dark) > 0 else 0,
        }
        
        # ОБЯЗАТЕЛЬНЫЕ признаки из cal_data для точного предсказания
        features.update({
            'cal_mean': np.mean(cal_data),
            'cal_std': np.std(cal_data),
            'cal_range': np.max(cal_data) - np.min(cal_data),
            'cal_median': np.median(cal_data),
            'cal_max': np.max(cal_data),
            'cal_min': np.min(cal_data),
            'cal_q25': np.percentile(cal_data, 25),
            'cal_q75': np.percentile(cal_data, 75),
            'cal_iqr': np.percentile(cal_data, 75) - np.percentile(cal_data, 25),
        })
        
        # Дополнительные признаки на основе калибровочных данных
        if len(cal_data) > 1:
            cal_deriv = np.gradient(cal_data)
            features.update({
                'cal_deriv_mean': np.mean(np.abs(cal_deriv)),
                'cal_deriv_std': np.std(cal_deriv),
                'cal_gradient': (cal_data[-1] - cal_data[0]) / len(cal_data) if len(cal_data) > 1 else 0,
            })
        
        # Соотношения между калибровочными и измерительными данными
        features.update({
            'measure_to_cal_ratio': np.mean(measure) / np.mean(cal_data) if np.mean(cal_data) != 0 else 0,
            'reference_to_cal_ratio': np.mean(reference) / np.mean(cal_data) if np.mean(cal_data) != 0 else 0,
            'cal_to_abs_ratio': np.mean(cal_data) / np.mean(absorbance) if np.mean(absorbance) != 0 else 0,
        })
        
        return features
        
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        return None

def predict_glucose(spectral_data):
    """Make glucose prediction from spectral data - returns only the glucose value"""
    global model_data
    
    if model_data is None:
        return None
    
    try:
        # Extract spectral arrays - cal_data теперь обязательно
        measure = spectral_data.get('measure', [])
        reference = spectral_data.get('reference', [])
        dark = spectral_data.get('dark', [])
        cal_data = spectral_data.get('cal_data', [])
        
        # Validate input data
        if not measure or not reference or not dark or not cal_data:
            logger.error("Missing required spectral data arrays")
            return None
        
        if len(measure) == 0 or len(reference) == 0 or len(dark) == 0 or len(cal_data) == 0:
            logger.error("Empty spectral data arrays")
            return None
        
        if not (len(measure) == len(reference) == len(dark)):
            logger.error(f"Spectral arrays length mismatch: measure={len(measure)}, reference={len(reference)}, dark={len(dark)}")
            return None
        
        # cal_data может иметь другой размер, но не должен быть пустым
        if len(cal_data) == 0:
            logger.error("cal_data array is empty")
            return None
        
        logger.debug(f"Processing spectral data: measure={len(measure)}, reference={len(reference)}, dark={len(dark)}, cal_data={len(cal_data)} points")
        
        # Calculate features
        features = calculate_absorbance_features(measure, reference, dark, cal_data)
        
        if features is None:
            logger.error("Feature calculation failed")
            return None
        
        # Prepare features for model
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        feature_names = model_data.get('feature_names', [])
        
        if not feature_names:
            logger.error("No feature names found in model data")
            return None
        
        # Create feature vector in correct order
        try:
            X = np.array([features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
        except Exception as e:
            logger.error(f"Feature vector creation failed: {e}")
            return None
        
        # Apply scaling if needed
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X)
        glucose_value = float(prediction[0])
        
        logger.debug(f"Prediction successful: {glucose_value:.1f} mg/dL")
        return round(glucose_value, 1)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_data
    
    status = {
        "status": "healthy" if model_data is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_data is not None,
        "service": "Glucose Prediction API (Local Model)",
        "version": "4.0.0",
        "deployment": "local-embedded"
    }
    
    if model_data is not None:
        status["model_info"] = {
            "name": model_data.get('model_name', 'Unknown'),
            "r2_score": round(model_data['metrics'].get('R²', 0), 3),
            "rmse": round(model_data['metrics'].get('RMSE', 0), 1)
        }
        
        # Информация о модели
        if 'model' in model_data:
            model = model_data['model']
            if hasattr(model, 'n_estimators'):
                status["model_info"]["trees"] = model.n_estimators
        
        if 'feature_names' in model_data:
            status["model_info"]["features"] = len(model_data['feature_names'])
        
        # Информация об оптимизации
        if 'trees_reduced' in model_data.get('metrics', {}):
            status["model_info"]["optimization"] = model_data['metrics']['trees_reduced']
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - returns only glucose value"""
    try:
        if model_data is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # cal_data теперь ОБЯЗАТЕЛЬНОЕ поле для точных предсказаний
        required_fields = ['measure', 'reference', 'dark', 'cal_data']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields,
                "note": "cal_data is required for accurate glucose predictions"
            }), 400
        
        glucose_value = predict_glucose(data)
        
        if glucose_value is None:
            return jsonify({"error": "Prediction failed"}), 400
        
        return jsonify(glucose_value)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint - returns array of glucose values"""
    try:
        if model_data is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({"error": "No samples provided. Expected format: {'samples': [sample1, sample2, ...]}"}), 400
        
        samples = data['samples']
        
        if not isinstance(samples, list):
            return jsonify({"error": "Samples must be a list"}), 400
        
        if len(samples) > 100:
            return jsonify({"error": "Maximum 100 samples allowed per batch"}), 400
        
        results = []
        
        for i, sample in enumerate(samples):
            try:
                glucose_value = predict_glucose(sample)
                results.append(glucose_value)
            except Exception as e:
                results.append(None)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    global model_data
    
    if model_data is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        info = {
            "model_name": model_data.get('model_name', 'Unknown'),
            "model_type": str(type(model_data['model']).__name__),
            "training_date": model_data.get('training_date', 'Unknown'),
            "metrics": model_data.get('metrics', {}),
            "feature_count": len(model_data.get('feature_names', [])),
            "deployment_type": "local-embedded",
            "compressed": True
        }
        
        # Добавляем информацию о модели
        if 'model' in model_data:
            model = model_data['model']
            if hasattr(model, 'n_estimators'):
                info["trees"] = model.n_estimators
            if hasattr(model, 'max_depth'):
                info["max_depth"] = model.max_depth
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    global model_data
    
    doc = {
        "service": "Glucose Prediction API",
        "version": "4.0.0",
        "deployment": "local-embedded-model",
        "status": "ready" if model_data is not None else "model not loaded",
        "description": "Production-ready API with embedded compressed model (700MB)",
        "features": [
            "🚀 Instant startup (no downloads)",
            "🗜️ Compressed model support", 
            "🐳 Docker-ready deployment",
            "⚡ High-performance predictions",
            "📊 Batch processing support"
        ],
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check",
            "POST /predict": "Single glucose prediction (returns number)",
            "POST /predict/batch": "Batch predictions (returns array)",
            "GET /model/info": "Detailed model information"
        },
        "model_info": {
            "size": "700MB compressed",
            "format": "gzip compressed pickle",
            "startup_time": "~10 seconds",
            "no_external_dependencies": True
        },
        "example_usage": {
            "single_prediction": {
                "url": "/predict",
                "method": "POST",
                "body": {
                    "measure": [1000, 1010, 1020],
                    "reference": [1100, 1110, 1120],
                    "dark": [50, 52, 54],
                    "cal_data": [900, 910, 920]
                },
                "response": "145.2"
            },
            "batch_prediction": {
                "url": "/predict/batch", 
                "method": "POST",
                "body": {
                    "samples": [
                        {
                            "measure": [1000, 1010, 1020],
                            "reference": [1100, 1110, 1120],
                            "dark": [50, 52, 54],
                            "cal_data": [900, 910, 920]
                        }
                    ]
                },
                "response": "[145.2]"
            }
        },
        "required_fields": {
            "measure": "Array of measurement values",
            "reference": "Array of reference values", 
            "dark": "Array of dark signal values",
            "cal_data": "Array of calibration data (REQUIRED for accurate predictions)"
        },
        "notes": {
            "arrays_length": "measure, reference, and dark must have the same length",
            "cal_data_importance": "cal_data is required for accurate glucose predictions",
            "glucose_range": "Expected output: 20-500 mg/dL"
        }
    }
    
    return jsonify(doc)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 GLUCOSE PREDICTION API - LOCAL MODEL DEPLOYMENT")
    print("=" * 60)
    print("📦 Model: Embedded 700MB compressed")
    print("🚀 Startup: ~10 seconds (no downloads)")
    print("🎯 Perfect for production deployment")
    print("=" * 60)
    
    # Load model on startup
    logger.info("🔄 Loading embedded model...")
    model_loaded = load_model_on_startup()
    
    if model_loaded:
        logger.info("🎉 API is ready for predictions!")
    else:
        logger.error("❌ Model loading failed!")
        logger.error("   Make sure model_compressed.pkl.gz is in the project directory")
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)