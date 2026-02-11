#!/usr/bin/env python3
"""
Diagnostic script for MacBook M1 setup
Run this to identify any issues before running the main app
"""

import sys
import platform
import subprocess

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_python():
    print_header("1. Python Version Check")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_platform():
    print_header("2. Platform Check")
    print(f"System: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    if platform.system() == 'Darwin' and 'arm' in platform.machine().lower():
        print("✅ Detected Apple Silicon (M1/M2)")
        return True
    elif platform.system() == 'Darwin':
        print("⚠️  Detected Intel Mac (use regular TensorFlow)")
        return True
    else:
        print("ℹ️  Not a Mac (use regular requirements.txt)")
        return True

def check_tensorflow():
    print_header("3. TensorFlow Check")
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Check if it's the Mac version
        if 'macos' in tf.__version__ or platform.system() == 'Darwin':
            print("✅ TensorFlow installed")
            
            # Test GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ Metal GPU detected: {len(gpus)} device(s)")
            else:
                print("⚠️  No GPU detected (CPU only)")
            return True
        else:
            print("✅ TensorFlow installed")
            return True
    except ImportError:
        print("❌ TensorFlow not installed")
        print("   Run: pip install tensorflow-macos tensorflow-metal")
        return False
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return False

def check_dependencies():
    print_header("4. Dependencies Check")
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yfinance': 'yfinance',
        'ccxt': 'ccxt',
        'sklearn': 'scikit-learn',
        'openpyxl': 'openpyxl'
    }
    
    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Run: pip install {package}")
            all_ok = False
    
    return all_ok

def check_network():
    print_header("5. Network Connectivity Check")
    
    # Test Binance
    print("Testing Binance API...")
    try:
        import ccxt
        exchange = ccxt.binance({'timeout': 10000})
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ Binance API: BTC/USDT = ${ticker['last']:,.2f}")
    except Exception as e:
        print(f"⚠️  Binance API failed: {e}")
        print("   Will fallback to Yahoo Finance")
    
    # Test Yahoo Finance
    print("\nTesting Yahoo Finance...")
    try:
        import yfinance as yf
        btc = yf.download('BTC-USD', period='1d', interval='1h', progress=False)
        if not btc.empty:
            print(f"✅ Yahoo Finance: Fetched {len(btc)} candles")
        else:
            print("⚠️  Yahoo Finance returned empty data")
    except Exception as e:
        print(f"❌ Yahoo Finance failed: {e}")
        return False
    
    return True

def check_directories():
    print_header("6. Directory Structure Check")
    import os
    
    required_dirs = [
        'data/raw',
        'results',
        'model/saved_models',
        'preprocessing',
        'signal_generation',
        'rolling_window'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Creating...")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"   ✅ Created {dir_path}")
            except Exception as e:
                print(f"   ❌ Failed to create: {e}")
                all_ok = False
    
    return all_ok

def test_data_loader():
    print_header("7. Data Loader Test")
    try:
        from preprocessing.data_loader import DataLoader
        print("Testing data fetch (this may take 30-60 seconds)...")
        loader = DataLoader('BTC/USDT')
        df = loader.get_raw_data()
        print(f"✅ Successfully fetched {len(df)} candles")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        return True
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  🍎 MacBook M1 Diagnostic Tool")
    print("  Bitcoin Futures Intelligence Setup Checker")
    print("="*60)
    
    results = {
        'Python': check_python(),
        'Platform': check_platform(),
        'TensorFlow': check_tensorflow(),
        'Dependencies': check_dependencies(),
        'Network': check_network(),
        'Directories': check_directories(),
        'Data Loader': test_data_loader()
    }
    
    print_header("📊 Summary")
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test}")
    
    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{total} tests passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 All checks passed! You can run: streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("   Refer to MAC_SETUP.md for detailed instructions.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
