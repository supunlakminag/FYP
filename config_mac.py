"""
Configuration file for MacBook M1 compatibility
Add this import at the top of data_loader.py and roll_train.py
"""

import os
import warnings

# M1 Mac optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Increase network timeouts for API calls
BINANCE_TIMEOUT = 30000  # 30 seconds
YFINANCE_TIMEOUT = 60    # 60 seconds

# Memory optimization for 8GB RAM
TF_MEMORY_GROWTH = True
BATCH_SIZE_M1 = 16  # Reduced from 32

warnings.filterwarnings('ignore')
