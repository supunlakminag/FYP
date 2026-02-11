from preprocessing.data_loader import DataLoader

# 1. Configuration (5 Years)
TICKER = "BTC-USD"
START_DATE = "2020-01-01"
END_DATE = "2025-01-01" 

# 2. Initialize
loader = DataLoader(TICKER, START_DATE, END_DATE)

# 3. Download & Save Raw
df_raw = loader.fetch_and_save_raw()

# 4. Inspect for Issues
if df_raw is not None:
    loader.inspect_data(df_raw)