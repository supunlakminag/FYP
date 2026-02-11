import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional

# --- CONFIGURATION ---
DATA_DIR = Path("data/raw")
TICKERS = ["BTC-USD", "ETH-USD"]
PERIOD = "730d"  # Max lookback for 1h interval on free tier
INTERVAL = "1h"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Cleans and standardizes the Yahoo Finance DataFrame.
    """
    try:
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns in downloaded data.")
            return None

        df = df[required_cols].copy()
        df.columns = [col.lower() for col in required_cols]
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        return df

    except Exception as e:
        logger.error(f"Error processing dataframe: {e}")
        return None

def download_hourly_data(tickers: List[str], save_dir: Path) -> None:
    """
    Downloads hourly crypto data and saves it to CSV.
    
    Args:
        tickers: List of ticker symbols (e.g., ['BTC-USD']).
        save_dir: Path object for the directory to save files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory verified: {save_dir}")

    total_rows = 0

    for ticker in tickers:
        logger.info(f"📥 Initiating download for {ticker}...")
        
        try:
            # Download data
            df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
            
            if df.empty:
                logger.warning(f"⚠️ No data found for {ticker}. Skipping.")
                continue

            # Process data
            clean_df = process_dataframe(df)
            
            if clean_df is not None and not clean_df.empty:
                # Save to CSV
                file_path = save_dir / f"{ticker}_hourly.csv"
                clean_df.to_csv(file_path)
                
                rows = len(clean_df)
                total_rows += rows
                logger.info(f"✅ Successfully saved {ticker} ({rows} rows) to {file_path}")
            else:
                logger.warning(f"❌ Data processing failed for {ticker}")

        except Exception as e:
            logger.error(f"❌ Critical error downloading {ticker}: {e}")

    logger.info(f"🎉 Batch complete. Total data points collected: {total_rows}")

if __name__ == "__main__":
    download_hourly_data(TICKERS, DATA_DIR)


