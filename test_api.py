#!/usr/bin/env python3
"""
Standalone model loading test script
Run this to diagnose exactly what's wrong with model loading
"""

import os
import sys
import traceback
import time
import gzip
import pickle
import joblib
import gc

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Python imports...")
    
    packages = [
        ('os', os),
        ('sys', sys), 
        ('time', time),
        ('gzip', gzip),
        ('pickle', pickle),
        ('joblib', joblib),
        ('gc', gc)
    ]
    
    for name, module in packages:
        try:
            print(f"   ✅ {name}: {module.__version__ if hasattr(module, '__version__') else 'OK'}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Test optional imports
    try:
        import psutil
        print(f"   ✅ psutil: {psutil.__version__}")
    except ImportError:
        print("   ⚠️ psutil: Not available (non-critical)")
    
    try:
        import numpy as np
        print(f"   ✅ numpy: {np.__version__}")
    except ImportError as e:
        print(f"   ❌ numpy: {e} (CRITICAL)")
    
    try:
        import sklearn
        print(f"   ✅ sklearn: {sklearn.__version__}")
    except ImportError as e:
        print(f"   ❌ sklearn: {e} (CRITICAL)")

def check_file_details(file_path):
    """Check file details"""
    print(f"\n🔍 Analyzing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ❌ File does not exist")
        return False
    
    # File stats
    stat = os.stat(file_path)
    size_mb = stat.st_size / 1024 / 1024
    print(f"   📏 Size: {size_mb:.1f} MB")
    print(f"   🗂️ Permissions: {oct(stat.st_mode)[-3:]}")
    print(f"   📖 Readable: {os.access(file_path, os.R_OK)}")
    
    # Try to read first few bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(20)
            print(f"   📋 File header: {header[:10]}")
            
            # Check if it's gzip
            if header[:2] == b'\x1f\x8b':
                print("   🗜️ Detected: gzip compressed file")
            else:
                print("   📄 Detected: uncompressed file")
                
    except Exception as e:
        print(f"   ❌ Cannot read file header: {e}")
        return False
    
    return True

def test_model_loading_methods(file_path):
    """Test different loading methods"""
    print(f"\n🚀 Testing model loading methods for: {file_path}")
    
    methods = [
        ("gzip + pickle", load_with_gzip_pickle),
        ("joblib", load_with_joblib),
        ("direct pickle", load_with_pickle),
        ("manual gzip", load_with_manual_gzip)
    ]
    
    for method_name, method_func in methods:
        print(f"\n   🔄 Method: {method_name}")
        try:
            start_time = time.time()
            result = method_func(file_path)
            load_time = time.time() - start_time
            
            if result is not None:
                print(f"   ✅ SUCCESS in {load_time:.1f}s")
                print(f"   📊 Result type: {type(result)}")
                
                if isinstance(result, dict):
                    print(f"   🗂️ Dict keys: {list(result.keys())}")
                    if 'model' in result:
                        model = result['model']
                        print(f"   🤖 Model type: {type(model).__name__}")
                        print(f"   🔧 Has predict: {hasattr(model, 'predict')}")
                
                print(f"   ✅ {method_name} WORKS!")
                return result, method_name
            else:
                print(f"   ❌ Returned None")
                
        except Exception as e:
            print(f"   ❌ Failed: {type(e).__name__}: {e}")
    
    return None, None

def load_with_gzip_pickle(file_path):
    """Load with gzip + pickle"""
    if not file_path.endswith('.gz'):
        return None
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def load_with_joblib(file_path):
    """Load with joblib"""
    return joblib.load(file_path)

def load_with_pickle(file_path):
    """Load with standard pickle"""
    if file_path.endswith('.gz'):
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_with_manual_gzip(file_path):
    """Load with manual gzip decompression"""
    if not file_path.endswith('.gz'):
        return None
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = gzip.decompress(compressed_data)
    return pickle.loads(decompressed_data)

def validate_model(model_data):
    """Validate loaded model"""
    print(f"\n🧪 Validating model...")
    
    if model_data is None:
        print("   ❌ Model data is None")
        return False
    
    print(f"   📊 Type: {type(model_data)}")
    
    if isinstance(model_data, dict):
        print(f"   🗂️ Keys: {list(model_data.keys())}")
        
        # Check required components
        required = ['model', 'feature_names']
        for key in required:
            if key in model_data:
                print(f"   ✅ Has {key}")
            else:
                print(f"   ⚠️ Missing {key}")
        
        # Test model object
        if 'model' in model_data:
            model = model_data['model']
            print(f"   🤖 Model: {type(model).__name__}")
            
            if hasattr(model, 'predict'):
                print("   ✅ Model has predict method")
                
                # Try a dummy prediction
                try:
                    import numpy as np
                    if 'feature_names' in model_data:
                        n_features = len(model_data['feature_names'])
                        dummy_X = np.zeros((1, n_features))
                        
                        # Check if scaler exists and is not None
                        scaler = model_data.get('scaler', None)
                        if scaler is not None:
                            dummy_X = scaler.transform(dummy_X)
                            print(f"   🔧 Applied scaler: {type(scaler).__name__}")
                        else:
                            print("   🔧 No scaler - using raw features")
                        
                        pred = model.predict(dummy_X)
                        print(f"   ✅ Dummy prediction works: {pred[0]:.2f}")
                        return True
                    else:
                        print("   ⚠️ Cannot test prediction - no feature_names")
                        return True
                except Exception as e:
                    print(f"   ❌ Prediction test failed: {e}")
                    print(f"   📍 Error details: {type(e).__name__}")
                    return False
            else:
                print("   ❌ Model missing predict method")
                return False
    else:
        print("   🤖 Direct model object")
        if hasattr(model_data, 'predict'):
            print("   ✅ Has predict method")
            return True
        else:
            print("   ❌ Missing predict method")
            return False
    
    return False

def main():
    """Main test function"""
    print("🧪 MODEL LOADING DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test imports
    test_imports()
    
    # Find model files
    print(f"\n📂 Current directory: {os.getcwd()}")
    
    model_files = []
    try:
        files = os.listdir('.')
        model_files = [f for f in files if '.pkl' in f or '.gz' in f]
        print(f"📋 Found {len(model_files)} potential model files: {model_files}")
    except Exception as e:
        print(f"❌ Cannot list directory: {e}")
        return
    
    if not model_files:
        print("❌ No model files found!")
        return
    
    # Test each model file
    for model_file in model_files:
        print(f"\n{'='*20} TESTING {model_file} {'='*20}")
        
        # Check file details
        if not check_file_details(model_file):
            continue
        
        # Test loading
        model_data, successful_method = test_model_loading_methods(model_file)
        
        if model_data is not None:
            # Validate model
            if validate_model(model_data):
                print(f"\n🎉 SUCCESS! {model_file} loaded with {successful_method}")
                print("✅ Model is ready for predictions!")
                return model_file, successful_method
            else:
                print(f"\n❌ Model validation failed for {model_file}")
        else:
            print(f"\n❌ All loading methods failed for {model_file}")
    
    print(f"\n❌ No working model files found!")
    return None, None

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")