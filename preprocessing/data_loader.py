import ccxt
import yfinance as yf
import pandas as pd
import os
import time

class DataLoader:
    def __init__(self, ticker="BTC/USDT", data_path="data/bitcoin_hourly.csv"):
        self.ticker_binance = ticker.replace("-", "/")  # BTC/USDT
        self.ticker_yahoo = ticker.replace("/", "-")    # BTC-USD
        self.data_path = data_path
        
        # Initialize Binance with timeout settings (M1 Mac compatible)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,  # 30 seconds timeout
            'options': {'defaultType': 'spot'}
        })

    def get_raw_data(self):
        """
        Robust Fetcher: Binance -> YFinance -> Local CSV
        """
        # --- ATTEMPT 1: BINANCE ---
        print(f"🔄 Attempting Binance Connection for {self.ticker_binance}...")
        try:
            all_candles = []
            now = self.exchange.milliseconds()
            since = now - (250 * 24 * 60 * 60 * 1000) # 250 Days ago

            while since < now:
                print(f"   Fetching batch from {pd.to_datetime(since, unit='ms')}...")
                ohlcv = self.exchange.fetch_ohlcv(self.ticker_binance, timeframe='1h', since=since, limit=1000)
                if not ohlcv: break
                all_candles += ohlcv
                since = ohlcv[-1][0] + 3600000 # Move forward 1 hour
                time.sleep(0.5)  # Rate limiting for M1 Mac
                if len(ohlcv) < 1000: break
            
            if len(all_candles) > 0:
                df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].astype(float)
                df = df.iloc[:-1] # Drop unfinished candle
                print(f"✅ Success: Fetched {len(df)} candles from BINANCE.")
                return df
            else:
                raise ValueError("Binance returned empty data.")

        except Exception as e:
            print(f"⚠️ Binance Failed ({e}). Switching to Backup...")

        # --- ATTEMPT 2: YFINANCE (The Reliable Backup) ---
        print(f"🔄 Attempting Yahoo Finance for {self.ticker_yahoo}...")
        try:
            # Fetch 2 Years (730 days) to be safe. YFinance is fast.
            df = yf.download(self.ticker_yahoo, period="2y", interval="1h", progress=False)
            
            if df.empty: raise ValueError("YFinance returned empty.")
            
            # Clean YFinance formatting
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [col.lower() for col in df.columns]
            df.index = df.index.tz_localize(None) # Remove timezone
            
            print(f"✅ Success: Fetched {len(df)} candles from YAHOO FINANCE.")
            return df

        except Exception as e:
            print(f"⚠️ YFinance Failed ({e}). Switching to Offline CSV...")

        # --- ATTEMPT 3: LOCAL CSV (The Fail-Safe) ---
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            df.columns = [col.lower() for col in df.columns]
            print(f"✅ Loaded {len(df)} candles from LOCAL CSV.")
            return df
        else:
            raise FileNotFoundError("❌ CRITICAL FAILURE: No Internet & No Local Data found.")
            return None
