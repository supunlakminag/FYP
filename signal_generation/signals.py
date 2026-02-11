import pandas as pd
import numpy as np
from datetime import datetime
import os

def calculate_rsi(series, period=14):
    """
    Calculates Relative Strength Index (RSI) manually 
    to avoid needing extra libraries like pandas-ta.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_signal(current_price, predicted_price, df_history, leverage=10):
    """
    The Master Logic Function.
    Inputs: Price data & AI Prediction.
    Output: A dictionary with the Decision, Stop Loss, Target, and Reason.
    """
    # --- 1. PREPARE DATA ---
    df = df_history.copy()
    # Normalize column names to lowercase just in case
    df.columns = [c.lower() for c in df.columns] 
    
    # --- 2. CALCULATE INDICATORS ---
    # RSI (Momentum)
    df['rsi'] = calculate_rsi(df['close'])
    current_rsi = df['rsi'].iloc[-1]
    
    # 200 Moving Average (Trend)
    df['ma_200'] = df['close'].rolling(window=200).mean()
    # If not enough data for 200MA, use current price as fallback to avoid crash
    ma_200 = df['ma_200'].iloc[-1] if not pd.isna(df['ma_200'].iloc[-1]) else current_price
    
    # --- 3. DETERMINE DIRECTION ---
    drift_pct = ((predicted_price - current_price) / current_price) * 100
    ai_direction = "LONG" if drift_pct > 0 else "SHORT"
    
    # --- 4. TIME FILTERS (UTC) ---
    today = datetime.utcnow().weekday() # 0=Monday, 4=Friday
    hour = datetime.utcnow().hour
    
    # --- 5. DECISION MATRIX ---
    action = "HOLD"
    reason = "Low Volatility / Neutral"
    
    # BUY SCENARIOS
    if ai_direction == "LONG" and abs(drift_pct) > 0.15:
        # A. Trend Follow: Price is above 200MA + RSI is not too high
        if current_price > ma_200 and current_rsi < 70:
            action = "BUY"
            reason = "Uptrend Confirmation"
        
        # B. Dip Buy: RSI is super low (Oversold)
        elif current_rsi < 30:
            action = "BUY"
            reason = "Oversold Reversal (Dip Buy)"
            
        # C. Monday Morning Reset
        elif today == 0 and hour < 12:
            action = "BUY"
            reason = "Monday Market Reset"
            
        else:
            action = "HOLD"
            reason = "Weak Buy Signal (Risk High)"

    # SELL SCENARIOS
    elif ai_direction == "SHORT" and abs(drift_pct) > 0.15:
        # A. Downtrend Follow: Price is below 200MA + RSI is not too low
        if current_price < ma_200 and current_rsi > 30:
            action = "SELL"
            reason = "Downtrend Confirmation"
            
        # B. Top Rejection: RSI is super high (Overbought)
        elif current_rsi > 70:
            action = "SELL"
            reason = "Overbought Rejection"
            
        # C. Friday Sell-off
        elif today == 4 and hour > 12:
            action = "SELL"
            reason = "Weekend Risk Off"
            
        else:
            action = "HOLD"
            reason = "Weak Sell Signal"

    # --- 6. RISK MANAGEMENT (ATR) ---
    high_low = df['high'] - df['low']
    atr = high_low.rolling(14).mean().iloc[-1]
    
    if action == "BUY":
        stop_loss = current_price - (1.5 * atr)
        target = current_price + (4.5 * atr) # 1:3 Risk/Reward
    elif action == "SELL":
        stop_loss = current_price + (1.5 * atr)
        target = current_price - (4.5 * atr)
    else:
        stop_loss = 0.0
        target = 0.0

    # Calculate PnL Percentages
    risk_pct_raw = (abs(current_price - stop_loss) / current_price) * 100 if current_price else 0
    reward_pct_raw = (abs(target - current_price) / current_price) * 100 if current_price else 0
    
    # --- 7. RETURN CARD DATA ---
    return {
        "action": action,
        "position": ai_direction,
        "coin": "BTC/USDT",
        "entry": current_price,
        "stop_loss": stop_loss,
        "target": target,
        "leverage": leverage,
        "risk_lev": risk_pct_raw * leverage,
        "reward_lev": reward_pct_raw * leverage,
        "confidence": min(abs(drift_pct) * 10, 99.9), 
        "validity": (datetime.utcnow()).strftime("%H:00 UTC next hour"),
        "reason": reason,
        "rsi": current_rsi,
        "trend": "BULL" if current_price > ma_200 else "BEAR"
    }

class SignalGenerator:
    def __init__(self, input_file="results/predictions.csv", threshold=0.005):
        self.input_file = input_file
        self.threshold = threshold

    def generate_signals(self):
        print("🚦 Generating Trading Signals with Explanations...")
        
        if not os.path.exists(self.input_file):
            print(f"❌ Error: {self.input_file} not found. Run compare_models.py first.")
            return None

        df = pd.read_csv(self.input_file)

        # 1. Calculate Drift
        df['Drift_Pct'] = (df['Predicted_Close'] - df['Actual_Close']) / df['Actual_Close']
        
        # 2. Add Threshold Column for reference
        df['Threshold_Used'] = self.threshold

        # 3. Define Logic for Signal AND Reasoning
        def analyze_row(row):
            drift = row['Drift_Pct']
            drift_display = round(drift * 100, 2) # Convert to % for readable text
            thresh_display = self.threshold * 100
            
            if drift > self.threshold:
                return pd.Series([
                    "BUY", 
                    f"Strong Upside: Predicted +{drift_display}% exceeds +{thresh_display}% threshold"
                ])
            elif drift < -self.threshold:
                return pd.Series([
                    "SELL", 
                    f"Strong Downside: Predicted {drift_display}% exceeds -{thresh_display}% threshold"
                ])
            else:
                return pd.Series([
                    "HOLD", 
                    f"Neutral: Move ({drift_display}%) is inside safety band (+/-{thresh_display}%)"
                ])

        # Apply logic to create two new columns
        df[['Signal', 'Reasoning']] = df.apply(analyze_row, axis=1)
        
        # 4. Save to Excel (Better than CSV for reading text)
        output_path = "results/Detailed_Signals.xlsx"
        df.to_excel(output_path, index=False)
        
        print(f"✅ Detailed signals saved to {output_path}")
        print("-" * 30)
        print("SAMPLE OUTPUT:")
        print(df[['Date', 'Signal', 'Reasoning']].tail(3))
        
        return df

if __name__ == "__main__":
    # 0.5% Threshold
    gen = SignalGenerator(threshold=0.005) 
    gen.generate_signals()