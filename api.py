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
import gc
import sys

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - memory monitoring disabled")

# Set up logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the loaded model
model_data = None

def get_memory_usage():
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        return 0
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def get_system_memory_info():
    """Get system memory information"""
    if not PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}
    
    try:
        memory_info = psutil.virtual_memory()
        return {
            "total_gb": memory_info.total / 1024 / 1024 / 1024,
            "available_gb": memory_info.available / 1024 / 1024 / 1024,
            "used_percent": memory_info.percent,
            "free_gb": memory_info.free / 1024 / 1024 / 1024
        }
    except Exception as e:
        return {"error": str(e)}

def detailed_model_diagnostics():
    """Comprehensive model loading diagnostics"""
    logger.info("🔍 DETAILED MODEL DIAGNOSTICS")
    logger.info("=" * 50)
    
    # Check current working directory
    current_dir = os.getcwd()
    logger.info(f"📂 Current directory: {current_dir}")
    
    # System memory info
    memory_info = get_system_memory_info()
    if "error" not in memory_info:
        logger.info(f"💾 System memory: {memory_info['total_gb']:.1f} GB total")
        logger.info(f"💾 Available memory: {memory_info['available_gb']:.1f} GB")
        logger.info(f"💾 Memory usage: {memory_info['used_percent']:.1f}%")
    
    # List ALL files in directory
    try:
        all_files = os.listdir('.')
        logger.info(f"📁 Total files in directory: {len(all_files)}")
        
        # Show model-related files
        model_files = [f for f in all_files if '.pkl' in f or '.gz' in f or 'model' in f.lower()]
        if model_files:
            logger.info("🎯 Found potential model files:")
            for f in model_files:
                try:
                    size = os.path.getsize(f)
                    permissions = oct(os.stat(f).st_mode)[-3:]
                    readable = os.access(f, os.R_OK)
                    size_mb = size / 1024 / 1024
                    logger.info(f"   📄 {f}: {size_mb:.1f} MB, permissions: {permissions}, readable: {readable}")
                except Exception as e:
                    logger.error(f"   ❌ {f}: Error reading - {e}")
        else:
            logger.error("❌ No model files found!")
                
    except Exception as e:
        logger.error(f"❌ Directory listing failed: {e}")
    
    # Check Python environment
    logger.info("🐍 Python environment:")
    logger.info(f"   Python version: {sys.version}")
    logger.info(f"   Current working directory: {os.getcwd()}")
    
    # Check required packages
    required_packages = ['joblib', 'pickle', 'gzip', 'numpy', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"   ✅ {package}: Available")
        except ImportError:
            logger.error(f"   ❌ {package}: Missing!")
    
    logger.info("=" * 50)

def load_model_with_fallback(file_path):
    """Try multiple loading methods with detailed error reporting and memory optimization"""
    logger.info(f"🔄 Attempting to load: {file_path}")
    
    # Check file exists and is readable
    if not os.path.exists(file_path):
        logger.error(f"❌ File does not exist: {file_path}")
        return None
    
    if not os.access(file_path, os.R_OK):
        logger.error(f"❌ File is not readable: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / 1024 / 1024
    logger.info(f"📏 File size: {file_size_mb:.1f} MB")
    
    # Memory optimization for large files
    if file_size_mb > 500:
        logger.info("🚨 Large file detected - applying memory optimizations")
        gc.collect()  # Force garbage collection
        initial_memory = get_memory_usage()
        logger.info(f"💾 Memory before loading: {initial_memory:.1f} MB")
    
    # Method 1: Standard gzip + pickle (for .gz files)
    if file_path.endswith('.gz'):
        try:
            logger.info("🗜️ Trying gzip + pickle...")
            start_time = time.time()
            
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Gzip + pickle succeeded in {load_time:.1f} seconds!")
            
            if file_size_mb > 500:
                final_memory = get_memory_usage()
                memory_increase = final_memory - initial_memory
                logger.info(f"💾 Memory after loading: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
            
            return data
            
        except MemoryError as e:
            logger.error(f"💥 Memory error during gzip+pickle loading: {e}")
            logger.error("💡 Try increasing container memory or using a smaller model")
            return None
        except Exception as e1:
            logger.warning(f"⚠️ Gzip+pickle failed: {type(e1).__name__}: {e1}")
    
    # Method 2: Try joblib (works with many formats)
    try:
        logger.info("🔧 Trying joblib...")
        start_time = time.time()
        data = joblib.load(file_path)
        load_time = time.time() - start_time
        logger.info(f"✅ Joblib succeeded in {load_time:.1f} seconds!")
        return data
    except MemoryError as e:
        logger.error(f"💥 Memory error during joblib loading: {e}")
        return None
    except Exception as e2:
        logger.warning(f"⚠️ Joblib failed: {type(e2).__name__}: {e2}")
    
    # Method 3: Try standard pickle (for uncompressed files)
    if not file_path.endswith('.gz'):
        try:
            logger.info("🥒 Trying standard pickle...")
            start_time = time.time()
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time
            logger.info(f"✅ Standard pickle succeeded in {load_time:.1f} seconds!")
            return data
        except MemoryError as e:
            logger.error(f"💥 Memory error during pickle loading: {e}")
            return None
        except Exception as e3:
            logger.warning(f"⚠️ Standard pickle failed: {type(e3).__name__}: {e3}")
    
    logger.error(f"❌ All loading methods failed for: {file_path}")
    return None

def validate_model_structure(model_data):
    """Validate that loaded model has the expected structure"""
    try:
        if model_data is None:
            logger.error("❌ Model data is None")
            return False
        
        logger.info(f"📊 Model data type: {type(model_data)}")
        
        if isinstance(model_data, dict):
            logger.info(f"🗂️ Model dictionary keys: {list(model_data.keys())}")
            
            # Check for essential components
            if 'model' not in model_data:
                logger.error("❌ Missing 'model' key in model data")
                return False
            
            if 'feature_names' not in model_data:
                logger.warning("⚠️ Missing 'feature_names' - this might cause issues")
            
            # Validate the actual model object
            model = model_data['model']
            if not hasattr(model, 'predict'):
                logger.error("❌ Model object doesn't have 'predict' method")
                return False
            
            logger.info(f"🤖 Model type: {type(model).__name__}")
            
            # Log model details
            if hasattr(model, 'n_estimators'):
                logger.info(f"🌳 Number of trees: {model.n_estimators}")
            
            if 'feature_names' in model_data:
                feature_count = len(model_data['feature_names'])
                logger.info(f"🏷️ Number of features: {feature_count}")
            
            if 'metrics' in model_data:
                metrics = model_data['metrics']
                logger.info(f"📈 Model metrics: {metrics}")
            
            # Check scaler (it might be None, which is OK)
            scaler = model_data.get('scaler', None)
            if scaler is not None:
                logger.info(f"🔧 Scaler type: {type(scaler).__name__}")
            else:
                logger.info("🔧 No scaler (features will be used raw)")
            
            # Test a dummy prediction to make sure everything works
            if 'feature_names' in model_data:
                try:
                    import numpy as np
                    n_features = len(model_data['feature_names'])
                    dummy_X = np.zeros((1, n_features))
                    
                    # Apply scaler only if it exists and is not None
                    if scaler is not None:
                        dummy_X = scaler.transform(dummy_X)
                    
                    pred = model.predict(dummy_X)
                    logger.info(f"✅ Dummy prediction successful: {pred[0]:.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ Dummy prediction failed (non-critical): {e}")
                    # Don't fail validation for this
            
            logger.info("✅ Model structure validation passed!")
            return True
            
        else:
            # Direct model object
            logger.info("🤖 Direct model object detected")
            if hasattr(model_data, 'predict'):
                logger.info("✅ Model has predict method")
                return True
            else:
                logger.error("❌ Direct model object missing predict method")
                return False
                
    except Exception as e:
        logger.error(f"❌ Model validation error: {e}")
        # Don't fail validation for minor issues
        return True  # Let's be more permissive

def load_model_on_startup():
    """
    Enhanced model loading with memory optimization and detailed error handling
    """
    global model_data
    
    logger.info("🚀 Starting MEMORY-OPTIMIZED model loading...")
    logger.info("=" * 60)
    
    # Run comprehensive diagnostics
    detailed_model_diagnostics()
    
    # List of model files to try (in order of priority)
    model_files = [
        "model_compressed.pkl.gz",
        "model_optimized.pkl.gz",
        "model.pkl.gz",
        "random_forest_model_0740_rmse_17.pkl.gz",
        "model.pkl",
        "random_forest_model_0740_rmse_17.pkl",
        "glucose_model.pkl",
        "glucose_model.pkl.gz",
        "trained_model.pkl",
        "trained_model.pkl.gz"
    ]
    
    # Also scan for any additional model files
    try:
        all_files = os.listdir('.')
        discovered_models = [f for f in all_files if 
                           (f.endswith('.pkl') or f.endswith('.pkl.gz')) and 
                           f not in model_files and
                           os.path.getsize(f) > 1024*1024]  # At least 1MB
        if discovered_models:
            logger.info(f"🔍 Discovered additional potential model files: {discovered_models}")
            model_files.extend(discovered_models)
    except Exception as e:
        logger.warning(f"⚠️ Could not scan for additional model files: {e}")
    
    # Find the actual model file
    found_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            file_size_mb = file_size / 1024 / 1024
            logger.info(f"✅ Found: {model_file} ({file_size_mb:.1f} MB)")
            found_model = model_file
            break
    
    if not found_model:
        logger.error("❌ No model file found!")
        logger.error("🔧 Available files:")
        try:
            for f in os.listdir('.'):
                if os.path.isfile(f):
                    size = os.path.getsize(f)
                    if size > 1024:  # Show files > 1KB
                        logger.error(f"   📄 {f}: {size/1024:.1f} KB")
        except:
            pass
        return False
    
    # Memory check for large models
    file_size_mb = os.path.getsize(found_model) / 1024 / 1024
    if file_size_mb > 100:  # For models > 100MB
        memory_info = get_system_memory_info()
        if "error" not in memory_info:
            estimated_uncompressed_gb = file_size_mb * 4 / 1024  # Conservative estimate
            available_gb = memory_info['available_gb']
            
            logger.info(f"📊 Model file size: {file_size_mb:.1f} MB")
            logger.info(f"📊 Estimated uncompressed size: ~{estimated_uncompressed_gb:.1f} GB")
            logger.info(f"📊 Available memory: {available_gb:.1f} GB")
            
            if estimated_uncompressed_gb > available_gb * 0.7:  # Use max 70% of available memory
                logger.warning("⚠️ Model might be large for available memory!")
                logger.warning("💡 Loading will proceed but may be slow...")
    
    # Force garbage collection before loading
    logger.info("🧹 Running garbage collection...")
    gc.collect()
    
    # Load the model
    logger.info(f"🔄 Loading model: {found_model}")
    overall_start_time = time.time()
    
    try:
        model_data = load_model_with_fallback(found_model)
        
        if model_data is not None:
            total_load_time = time.time() - overall_start_time
            logger.info(f"🎉 Model loading completed in {total_load_time:.1f} seconds!")
            
            # Validate the model
            if validate_model_structure(model_data):
                logger.info("✅ Model is ready for predictions!")
                
                # Final memory check
                final_memory = get_memory_usage()
                logger.info(f"💾 Final memory usage: {final_memory:.1f} MB")
                
                logger.info("=" * 60)
                return True
            else:
                logger.error("❌ Model validation failed!")
                model_data = None
                return False
        else:
            logger.error("❌ Model loading returned None!")
            return False
            
    except Exception as e:
        logger.error(f"💥 Critical error during model loading: {e}")
        logger.error(f"📍 Full traceback: {traceback.format_exc()}")
        return False

def calculate_absorbance_features(measure, reference, dark, cal_data):
    """Calculate absorbance features for one sample - cal_data is required"""
    try:
        # Convert to numpy arrays
        measure = np.array(measure)
        reference = np.array(reference)
        dark = np.array(dark)
        cal_data = np.array(cal_data)  # Now required
        
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
        
        # REQUIRED features from cal_data for accurate prediction
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
        
        # Additional features based on calibration data
        if len(cal_data) > 1:
            cal_deriv = np.gradient(cal_data)
            features.update({
                'cal_deriv_mean': np.mean(np.abs(cal_deriv)),
                'cal_deriv_std': np.std(cal_deriv),
                'cal_gradient': (cal_data[-1] - cal_data[0]) / len(cal_data) if len(cal_data) > 1 else 0,
            })
        
        # Ratios between calibration and measurement data
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
        # Extract spectral arrays - cal_data is now required
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
        
        # cal_data can have different size, but must not be empty
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
        scaler = model_data.get('scaler', None)  # This might be None
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
        
        # Apply scaling ONLY if scaler exists and is not None
        if scaler is not None:
            try:
                X = scaler.transform(X)
                logger.debug("Applied feature scaling")
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")
                # Continue without scaling
        else:
            logger.debug("No scaler - using raw features")
        
        # Make prediction
        prediction = model.predict(X)
        glucose_value = float(prediction[0])
        
        logger.debug(f"Prediction successful: {glucose_value:.1f} mg/dL")
        return round(glucose_value, 1)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Prediction traceback: {traceback.format_exc()}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model_data
    
    status = {
        "status": "healthy" if model_data is not None else "unhealthy!",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_data is not None,
        "service": "Glucose Prediction API (Memory-Optimized)",
        "version": "4.1.0",
        "deployment": "local-embedded-optimized"
    }
    
    if model_data is not None:
        status["model_info"] = {
            "name": model_data.get('model_name', 'Unknown'),
            "r2_score": round(model_data['metrics'].get('R²', 0), 3) if 'metrics' in model_data else 'N/A',
            "rmse": round(model_data['metrics'].get('RMSE', 0), 1) if 'metrics' in model_data else 'N/A'
        }
        
        # Model information
        if 'model' in model_data:
            model = model_data['model']
            if hasattr(model, 'n_estimators'):
                status["model_info"]["trees"] = model.n_estimators
        
        if 'feature_names' in model_data:
            status["model_info"]["features"] = len(model_data['feature_names'])
        
        # Optimization info
        if 'metrics' in model_data and 'trees_reduced' in model_data['metrics']:
            status["model_info"]["optimization"] = model_data['metrics']['trees_reduced']
    
    # Memory info
    if PSUTIL_AVAILABLE:
        memory_info = get_system_memory_info()
        if "error" not in memory_info:
            status["memory"] = {
                "process_mb": get_memory_usage(),
                "system_available_gb": memory_info['available_gb'],
                "system_used_percent": memory_info['used_percent']
            }
    
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
        
        # cal_data is now REQUIRED for accurate predictions
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
                logger.error(f"Batch prediction error for sample {i}: {e}")
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
            "deployment_type": "local-embedded-optimized",
            "compressed": True,
            "memory_optimized": True
        }
        
        # Add model details
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

@app.route('/debug/memory', methods=['GET'])
def debug_memory():
    """Debug endpoint to check memory usage"""
    if not PSUTIL_AVAILABLE:
        return jsonify({"error": "psutil not available - install with: pip install psutil"})
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return jsonify({
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
            "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024,
            "system_memory_percent": system_memory.percent,
            "model_loaded": model_data is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to see what files are available"""
    try:
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f or 'model' in f.lower()]
        
        file_info = {}
        for f in model_files:
            if os.path.exists(f):
                size = os.path.getsize(f)
                file_info[f] = {
                    "size_mb": size / 1024 / 1024,
                    "readable": os.access(f, os.R_OK),
                    "is_file": os.path.isfile(f)
                }
        
        return jsonify({
            "current_directory": os.getcwd(),
            "all_files": sorted(files),
            "potential_model_files": model_files,
            "file_details": file_info,
            "model_loaded": model_data is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/admin/reload-model', methods=['POST'])
def reload_model():
    """Admin endpoint to reload the model"""
    global model_data
    logger.info("🔄 Manual model reload requested...")
    
    # Clear current model
    model_data = None
    gc.collect()
    
    # Reload
    success = load_model_on_startup()
    
    return jsonify({
        "success": success,
        "model_loaded": model_data is not None,
        "message": "Model reloaded successfully" if success else "Model reload failed",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    global model_data
    
    doc = {
        "service": "Glucose Prediction API",
        "version": "4.1.0",
        "deployment": "local-embedded-memory-optimized",
        "status": "ready" if model_data is not None else "model not loaded",
        "description": "Production-ready API with memory-optimized compressed model loading",
        "features": [
            "🚀 Memory-optimized startup",
            "🗜️ Large compressed model support (700MB+)",
            "🐳 Docker-ready deployment",
            "⚡ High-performance predictions",
            "📊 Batch processing support",
            "💾 Memory monitoring and diagnostics",
            "🔧 Debug endpoints for troubleshooting"
        ],
        "endpoints": {
            "GET /": "API documentation",
            "GET /health": "Health check with memory info",
            "POST /predict": "Single glucose prediction (returns number)",
            "POST /predict/batch": "Batch predictions (returns array)",
            "GET /model/info": "Detailed model information",
            "GET /debug/memory": "Memory usage information",
            "GET /debug/files": "File system diagnostics",
            "POST /admin/reload-model": "Reload model (admin only)"
        },
        "model_info": {
            "size": "700MB+ compressed",
            "format": "gzip compressed pickle",
            "startup_time": "30-60 seconds for large models",
            "memory_optimized": True,
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
            "glucose_range": "Expected output: 20-500 mg/dL",
            "memory_requirements": "Ensure adequate RAM for large model loading"
        }
    }
    
    return jsonify(doc)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.route('/debug/startup', methods=['GET'])
def debug_startup():
    """Debug endpoint to see startup logs and force model reload with verbose output"""
    global model_data
    
    logger.info("🔍 Manual startup debug requested")
    
    # Clear any existing model
    model_data = None
    gc.collect()
    
    debug_info = {
        "startup_attempt": datetime.now().isoformat(),
        "steps": [],
        "files_found": [],
        "loading_attempts": [],
        "final_status": "unknown"
    }
    
    try:
        # Step 1: Check files
        debug_info["steps"].append("Checking files...")
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f]
        debug_info["files_found"] = model_files
        
        if not model_files:
            debug_info["final_status"] = "no_model_files"
            return jsonify(debug_info)
        
        # Step 2: Try to load each file
        for model_file in model_files:
            attempt = {
                "file": model_file,
                "size_mb": os.path.getsize(model_file) / 1024 / 1024,
                "loading_methods": []
            }
            
            # Method 1: gzip + pickle
            try:
                debug_info["steps"].append(f"Trying gzip+pickle on {model_file}")
                start_time = time.time()
                with gzip.open(model_file, 'rb') as f:
                    data = pickle.load(f)
                load_time = time.time() - start_time
                
                method_result = {
                    "method": "gzip+pickle",
                    "success": True,
                    "load_time": load_time,
                    "data_type": str(type(data)),
                    "validation": "not_tested"
                }
                
                # Quick validation
                if isinstance(data, dict):
                    method_result["keys"] = list(data.keys())
                    if 'model' in data:
                        model = data['model']
                        method_result["model_type"] = str(type(model).__name__)
                        method_result["has_predict"] = hasattr(model, 'predict')
                        
                        # Check scaler specifically
                        scaler = data.get('scaler', 'missing')
                        if scaler is None:
                            method_result["scaler"] = "None"
                        elif scaler == 'missing':
                            method_result["scaler"] = "missing_key"
                        else:
                            method_result["scaler"] = str(type(scaler).__name__)
                        
                        # Try prediction
                        try:
                            if 'feature_names' in data:
                                import numpy as np
                                n_features = len(data['feature_names'])
                                dummy_X = np.zeros((1, n_features))
                                
                                # Handle scaler properly
                                if scaler is not None:
                                    dummy_X = scaler.transform(dummy_X)
                                
                                pred = model.predict(dummy_X)
                                method_result["validation"] = "passed"
                                method_result["dummy_prediction"] = float(pred[0])
                            else:
                                method_result["validation"] = "no_feature_names"
                        except Exception as e:
                            method_result["validation"] = f"failed: {str(e)}"
                
                attempt["loading_methods"].append(method_result)
                
                # If this method worked, set as global model
                if method_result.get("validation") == "passed":
                    model_data = data
                    debug_info["final_status"] = "success"
                    debug_info["successful_method"] = method_result
                    break
                    
            except Exception as e:
                method_result = {
                    "method": "gzip+pickle", 
                    "success": False,
                    "error": str(e),
                    "error_type": str(type(e).__name__)
                }
                attempt["loading_methods"].append(method_result)
            
            debug_info["loading_attempts"].append(attempt)
            
            # If we found a working model, break
            if model_data is not None:
                break
        
        # Final status
        if model_data is not None:
            debug_info["final_status"] = "success"
            debug_info["model_loaded"] = True
        else:
            debug_info["final_status"] = "all_methods_failed"
            debug_info["model_loaded"] = False
            
    except Exception as e:
        debug_info["final_status"] = "critical_error"
        debug_info["error"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
    
    return jsonify(debug_info)

@app.route('/debug/force-load', methods=['POST'])
def force_load_model():
    """Force load model with minimal validation"""
    global model_data
    
    logger.info("🔄 Force loading model with minimal validation...")
    
    # Clear existing model
    model_data = None
    gc.collect()
    
    try:
        # Find model file
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f]
        
        if not model_files:
            return jsonify({"success": False, "error": "No model files found"})
        
        model_file = model_files[0]  # Use first one found
        logger.info(f"🔄 Force loading: {model_file}")
        
        # Load with minimal checks
        with gzip.open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"✅ Loaded data type: {type(data)}")
        
        # Minimal validation - just check if it's a dict with model key
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            if hasattr(model, 'predict'):
                model_data = data
                logger.info("✅ Force load successful!")
                return jsonify({
                    "success": True, 
                    "model_loaded": True,
                    "file": model_file,
                    "model_type": str(type(model).__name__),
                    "keys": list(data.keys())
                })
        
        return jsonify({"success": False, "error": "Model validation failed"})
        
    except Exception as e:
        logger.error(f"❌ Force load failed: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/debug/model-keys', methods=['GET'])
def debug_model_keys():
    """Show what's actually in the model file"""
    try:
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f]
        
        if not model_files:
            return jsonify({"error": "No model files found"})
        
        model_file = model_files[0]
        
        with gzip.open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        result = {
            "file": model_file,
            "data_type": str(type(data)),
            "model_data_loaded": model_data is not None
        }
        
        if isinstance(data, dict):
            result["keys"] = list(data.keys())
            
            for key, value in data.items():
                if key == 'model':
                    result[f"{key}_type"] = str(type(value).__name__)
                    result[f"{key}_has_predict"] = hasattr(value, 'predict')
                elif key == 'scaler':
                    if value is None:
                        result[f"{key}_value"] = "None"
                    else:
                        result[f"{key}_type"] = str(type(value).__name__)
                elif key == 'feature_names':
                    result[f"{key}_count"] = len(value) if value else 0
                else:
                    result[f"{key}_type"] = str(type(value).__name__)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

@app.route('/debug/file-content', methods=['GET'])
def debug_file_content():
    """Check what's actually in the model file"""
    try:
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f]
        
        if not model_files:
            return jsonify({"error": "No model files found"})
        
        result = {}
        
        for model_file in model_files:
            file_info = {
                "size_bytes": os.path.getsize(model_file),
                "size_mb": os.path.getsize(model_file) / 1024 / 1024,
                "exists": os.path.exists(model_file),
                "readable": os.access(model_file, os.R_OK)
            }
            
            # Read first 100 bytes to see what's actually in the file
            try:
                with open(model_file, 'rb') as f:
                    first_bytes = f.read(100)
                    file_info["first_20_bytes_hex"] = first_bytes[:20].hex()
                    file_info["first_50_chars_text"] = str(first_bytes[:50])
                    
                    # Check if it's actually a text file (like a Git LFS pointer)
                    try:
                        text_content = first_bytes.decode('utf-8')
                        file_info["text_content"] = text_content
                        
                        # Check for Git LFS
                        if "version https://git-lfs.github.com" in text_content:
                            file_info["is_git_lfs_pointer"] = True
                        elif "oid sha256:" in text_content:
                            file_info["is_git_lfs_pointer"] = True
                        else:
                            file_info["is_git_lfs_pointer"] = False
                            
                    except UnicodeDecodeError:
                        file_info["is_text_file"] = False
                        
            except Exception as e:
                file_info["read_error"] = str(e)
            
            result[model_file] = file_info
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()})

if __name__ == '__main__':
    print("🚀 GLUCOSE PREDICTION API - MEMORY-OPTIMIZED DEPLOYMENT")
    print("=" * 65)
    print("📦 Model: Embedded 700MB+ compressed with memory optimization")
    print("🚀 Startup: ~30-60 seconds (no downloads)")
    print("💾 Memory: Optimized loading with monitoring")
    print("🎯 Perfect for production deployment")
    print("=" * 65)
    
    # Load model on startup
    logger.info("🔄 Loading embedded model with memory optimization...")
    model_loaded = load_model_on_startup()
    
    if model_loaded:
        logger.info("🎉 API is ready for predictions!")
        print("✅ Model loaded successfully!")
        print("🌐 API endpoints available:")
        print("   📊 http://localhost:8000/health")
        print("   🔧 http://localhost:8000/debug/memory")
        print("   📋 http://localhost:8000/debug/files")
    else:
        logger.error("❌ Model loading failed!")
        print("❌ Model loading failed!")
        print("🔧 Troubleshooting:")
        print("   1. Check if model_compressed.pkl.gz exists")
        print("   2. Verify sufficient memory (4GB+ recommended)")
        print("   3. Check debug endpoints for more info")
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)