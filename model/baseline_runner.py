import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from model.lstm_model import build_lstm_model
import math

class BaselineRunner:
    def __init__(self, df):
        self.df = df
        self.look_back = 60 

    def run(self):
        print("📉 Running BASELINE Model (Raw Data / No Normalization)...")
        
        # Scenario Description for Excel
        preprocessing_desc = "None (Raw Data)"
        scaling_desc = "None (Raw Prices Used)"  # explicitly stating no scaling
        split_desc = "Static Hold-Out (80% Train / 20% Test)"
        novelty_desc = "Baseline (Control Group)"
        
        # 1. Prepare Data (NO SCALING)
        data = self.df[['close']].values.astype(float)
        
        # 2. Static Split
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[0:train_size], data[train_size:]
        
        # 3. Create Sequences
        def create_dataset(dataset, look_back):
            X, Y = [], []
            for i in range(len(dataset) - look_back - 1):
                X.append(dataset[i:(i+look_back), 0])
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        X_train, y_train = create_dataset(train_data, self.look_back)
        X_test, y_test = create_dataset(test_data, self.look_back)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 4. Train Model
        print("   (Training on raw values - this may result in high loss...)")
        model = build_lstm_model((self.look_back, 1))
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        # 5. Predict
        test_predict = model.predict(X_test, verbose=0)
        
        # 6. Metrics
        if np.isnan(test_predict).any():
            print("   ⚠️ WARNING: Model produced NaN predictions due to lack of normalization.")
            rmse, mae, mape = 999999, 999999, 999999 # Dummy bad values
            obs = "Model FAILED (NaN outputs) due to lack of normalization."
        else:
            rmse = math.sqrt(mean_squared_error(y_test, test_predict[:,0]))
            mae = mean_absolute_error(y_test, test_predict[:,0])
            mape = mean_absolute_percentage_error(y_test, test_predict[:,0]) * 100
            obs = "High error/instability due to unscaled inputs."
        
        print(f"   Done. RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        
        return {
            "Model_Type": "Baseline (Raw/Unscaled)",
            "Preprocessing": preprocessing_desc,
            "Normalization": scaling_desc,
            "Training_Strategy": split_desc,
            "Novelty_Contribution": novelty_desc,
            "RMSE (Error Cost)": round(rmse, 2),
            "MAE (Avg Error $)": round(mae, 2),
            "MAPE (Error %)": f"{round(mape, 2)}%",
            "Observation": obs
        }