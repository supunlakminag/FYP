import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Import your existing modules
from preprocessing.data_loader import DataLoader
from rolling_window.roll_train import RollingWindowTrainer


def _fallback_latest_prediction(df):
    """
    Returns a safe fallback prediction row when model output is unavailable.
    """
    if df is None or df.empty or "close" not in df.columns:
        current_price = 0.0
    else:
        current_price = float(df["close"].iloc[-1])

    return pd.Series(
        {
            "Date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "Actual_Close": current_price,
            "Predicted_Close": current_price,
        }
    )

def get_market_data(ticker):
    """
    Backend function to fetch raw data.
    This works dynamically for any ticker supported by DataLoader.
    """
    loader = DataLoader(ticker)
    return loader.get_raw_data()

def run_ai_training(df, window_hours):
    """
    Orchestrates the AI Training.
    Returns the latest prediction row.
    """
    # Slice relevant range for training
    needed_data = window_hours + 12
    df_subset = df.iloc[-needed_data:].copy()

    # Guardrails for short/unavailable data
    if df_subset.empty or "close" not in df_subset.columns:
        return _fallback_latest_prediction(df)

    # Ensure window size is valid for current data volume
    # (rolling trainer needs enough rows to produce at least one prediction).
    adaptive_window = min(window_hours, max(72, len(df_subset) - 24))
    if len(df_subset) <= adaptive_window:
        return _fallback_latest_prediction(df_subset)

    # Run Trainer
    trainer = RollingWindowTrainer(df_subset, window_size=adaptive_window, debug_mode=False)
    trainer.run()

    # Read the results it just saved
    if not os.path.exists("results/predictions.csv"):
        return _fallback_latest_prediction(df_subset)

    results_df = pd.read_csv("results/predictions.csv")
    if results_df.empty:
        return _fallback_latest_prediction(df_subset)

    latest_prediction = results_df.iloc[-1]
    if "Actual_Close" not in latest_prediction or "Predicted_Close" not in latest_prediction:
        return _fallback_latest_prediction(df_subset)

    return latest_prediction

# Updated to accept 'ticker' argument
def calculate_signal_logic(current_price, predicted_price, df_history, leverage=10, ticker="BTC/USDT"):
    """
    Pure math logic for the signal card.
    """
    # 1. Determine Drift
    drift_pct = ((predicted_price - current_price) / current_price) * 100
    
    # Calculate Validity
    valid_until = (datetime.now() + timedelta(hours=1)).strftime("%H:%M UTC")

    # 2. Market Noise Filter (Threshold 0.1%)
    if abs(drift_pct) < 0.1:
        return {
            "action": "HOLD",
            "position": "NEUTRAL",
            "coin": ticker,  # <--- FIXED: Uses the dynamic ticker
            "entry": current_price,
            "stop_loss": 0.0,
            "target": 0.0,
            "leverage": leverage,
            "risk_raw": 0.0,
            "risk_lev": 0.0,
            "reward_raw": 0.0,
            "reward_lev": 0.0,
            "confidence": abs(drift_pct),
            "validity": valid_until
        }

    # --- Standard BUY/SELL Logic ---
    direction = "LONG" if predicted_price > current_price else "SHORT"
    action = "BUY" if direction == "LONG" else "SELL"
    
    # ATR Calculation
    high_low = df_history['high'] - df_history['low']
    atr = high_low.rolling(14).mean().iloc[-1]
    
    entry_price = current_price
    
    if direction == "LONG":
        stop_loss = entry_price - (1.5 * atr)
        risk_amount = entry_price - stop_loss
        target = entry_price + (risk_amount * 3.0) 
    else: # SHORT
        stop_loss = entry_price + (1.5 * atr)
        risk_amount = stop_loss - entry_price
        target = entry_price - (risk_amount * 3.0)

    # Percentage Calculations
    risk_pct_raw = (abs(entry_price - stop_loss) / entry_price) * 100
    reward_pct_raw = (abs(target - entry_price) / entry_price) * 100
    
    return {
        "action": action,
        "position": direction,
        "coin": ticker, 
        "entry": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "leverage": leverage,
        "risk_raw": risk_pct_raw,
        "risk_lev": risk_pct_raw * leverage,
        "reward_raw": reward_pct_raw,
        "reward_lev": reward_pct_raw * leverage,
        "confidence": abs(drift_pct),
        "validity": valid_until
    }
