import gc
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import logging
import os
import platform

from model.lstm_model import build_lstm_model

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# M1 Mac Memory Optimization
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU config: {e}")

class RollingWindowTrainer:
    def __init__(self, df, window_size=180, debug_mode=True):
        self.df = df
        self.window_size = window_size
        self.debug_mode = debug_mode
        self.results = []

    def calculate_metrics(self, actuals, predictions):
        """
        Calculates the 3 key accuracy metrics for the report.
        """
        if not actuals or not predictions:
            return {
                "RMSE (Error Cost)": 0.0,
                "MAPE (Error %)": "0.0%",
                "MAE (Avg Error $)": 0.0
            }

        # Convert to numpy arrays for math operations
        y_true = np.array(actuals)
        y_pred = np.array(predictions)

        # 1. RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 2. MAPE (Mean Absolute Percentage Error)
        # Add epsilon to avoid division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        # 3. MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)

        return {
            "RMSE (Error Cost)": round(rmse, 2),
            "MAPE (Error %)": f"{round(mape, 2)}%",
            "MAE (Avg Error $)": round(mae, 2)
        }

    def run(self):
        print(f"🚀 Running EVALUATION Model (Rolling Window)...")
        
        # Scenario Description for Excel
        preprocessing_desc = "Linear Interpolation + Feature Engineering"
        scaling_desc = "Dynamic Window Scaling (Fit per Window)"
        split_desc = "Rolling Walk-Forward Validation"
        novelty_desc = "Adaptive Retraining on Non-Stationary Data"
        
        # Data Selection
        if self.debug_mode:
            # Last 500 hours for quick test
            df_subset = self.df.iloc[-500:].copy()
        else:
            df_subset = self.df.copy()

        total_steps = len(df_subset)
        start_index = self.window_size
        
        predictions = []
        actuals = []
        dates = []
        
        # Start Rolling Window Loop
        for t in range(start_index, total_steps):
            # 1. Window Selection
            window_data = df_subset.iloc[t - self.window_size : t + 1].values
            
            # 2. Dynamic Preprocessing (Per Window!)
            train_data_raw = window_data[:-1]
            target_actual = window_data[-1, 3] # Close price
            current_date = df_subset.index[t]

            # 3. Dynamic Normalization (Crucial Step)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train_data_raw)
            train_data_scaled = scaler.transform(train_data_raw)
            
            # 4. Create Sub-sequences
            internal_win = 60
            X_sub, y_sub = [], []
            for i in range(internal_win, len(train_data_scaled)):
                X_sub.append(train_data_scaled[i-internal_win:i])
                y_sub.append(train_data_scaled[i, 3])
            
            X_sub, y_sub = np.array(X_sub), np.array(y_sub)
            
            # --- Skip if window is too small for sequences ---
            if len(X_sub) == 0:
                if t % 50 == 0:
                    print(f"   ⚠️ Skipping step {t}: Insufficient data for lookback.")
                continue

            # 5. Train Model (Fine-tune)
            try:
                # Build model inside loop (Adaptive Retraining)
                model = build_lstm_model((internal_win, 5))
                
                # verbose=0 silences the progress bar for speed
                # Use smaller batch size for M1 Mac (8GB RAM)
                batch_size = 16 if platform.system() == 'Darwin' else 32
                model.fit(X_sub, y_sub, epochs=1, batch_size=batch_size, verbose=0)
                
                # 6. Predict Next Step
                last_seq = train_data_scaled[-internal_win:].reshape(1, internal_win, 5)
                pred_scaled = model.predict(last_seq, verbose=0)
                
                # 7. Inverse Transform
                dummy = np.zeros((1, 5))
                dummy[0, 3] = pred_scaled[0, 0]
                pred_inv = scaler.inverse_transform(dummy)[0, 3]
                
                predictions.append(pred_inv)
                actuals.append(target_actual)
                dates.append(current_date)
                
            except Exception as e:
                print(f"❌ Error at step {t}: {e}")
                continue

            # --- MEMORY CLEANUP ---
            if t % 20 == 0:
                K.clear_session()
                gc.collect()

            # Print progress every 50 steps
            if t % 50 == 0:
                 print(f"   Step {t}/{total_steps} | Pred: {pred_inv:.2f} | Actual: {target_actual:.2f}")

        # --- Save Raw Predictions ---
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual_Close': actuals,
            'Predicted_Close': predictions
        })
        os.makedirs("results", exist_ok=True)
        results_df.to_csv("results/predictions.csv", index=False)

        # --- Calculate Final Metrics ---
        if len(predictions) == 0:
            print("❌ No predictions made. Check Window Size.")
            return {}

        metrics_dict = self.calculate_metrics(actuals, predictions)
        
        print(f"   Done. RMSE: {metrics_dict['RMSE (Error Cost)']} | MAPE: {metrics_dict['MAPE (Error %)']}")
        
        K.clear_session()
        gc.collect()
        
        return metrics_dict