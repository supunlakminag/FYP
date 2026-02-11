#!/usr/bin/env python3
"""
Quick API Test - Run this first on Mac to verify connectivity
"""

import time
import sys

print("🧪 Testing API Connectivity...\n")

# Test 1: Binance
print("1️⃣  Testing Binance API...")
try:
    import ccxt
    exchange = ccxt.binance({'timeout': 30000, 'enableRateLimit': True})
    print("   Connecting...")
    start = time.time()
    ticker = exchange.fetch_ticker('BTC/USDT')
    elapsed = time.time() - start
    print(f"   ✅ SUCCESS in {elapsed:.2f}s")
    print(f"   BTC/USDT Price: ${ticker['last']:,.2f}\n")
    binance_ok = True
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")
    binance_ok = False

# Test 2: Yahoo Finance
print("2️⃣  Testing Yahoo Finance API...")
try:
    import yfinance as yf
    print("   Connecting...")
    start = time.time()
    btc = yf.download('BTC-USD', period='5d', interval='1h', progress=False)
    elapsed = time.time() - start
    if not btc.empty:
        print(f"   ✅ SUCCESS in {elapsed:.2f}s")
        print(f"   Fetched {len(btc)} candles")
        print(f"   Latest Price: ${btc['Close'].iloc[-1]:,.2f}\n")
        yahoo_ok = True
    else:
        print("   ❌ FAILED: Empty data\n")
        yahoo_ok = False
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")
    yahoo_ok = False

# Test 3: Data Loader
print("3️⃣  Testing Data Loader (Full Integration)...")
try:
    from preprocessing.data_loader import DataLoader
    print("   This may take 30-60 seconds...")
    start = time.time()
    loader = DataLoader('BTC/USDT')
    df = loader.get_raw_data()
    elapsed = time.time() - start
    print(f"   ✅ SUCCESS in {elapsed:.2f}s")
    print(f"   Fetched {len(df)} candles")
    print(f"   Date Range: {df.index[0]} to {df.index[-1]}\n")
    loader_ok = True
except Exception as e:
    print(f"   ❌ FAILED: {e}\n")
    loader_ok = False

# Summary
print("="*50)
print("📊 SUMMARY")
print("="*50)
print(f"Binance API:     {'✅ PASS' if binance_ok else '❌ FAIL'}")
print(f"Yahoo Finance:   {'✅ PASS' if yahoo_ok else '❌ FAIL'}")
print(f"Data Loader:     {'✅ PASS' if loader_ok else '❌ FAIL'}")
print("="*50)

if loader_ok:
    print("\n🎉 All tests passed! You can run: streamlit run app.py")
    sys.exit(0)
elif yahoo_ok:
    print("\n⚠️  Binance failed but Yahoo works. App will use Yahoo Finance.")
    print("   You can run: streamlit run app.py")
    sys.exit(0)
else:
    print("\n❌ API tests failed. Check your internet connection.")
    print("   Try: ping api.binance.com")
    sys.exit(1)
