import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from model.lstm_model import build_lstm_model
import math

class IntermediateRunner:
    def __init__(self, df, mode='drop'):
        """
        mode: 'drop' (B1.1) or 'interpolate' (B1.2)
        """
        self.raw_df = df.copy()
        self.mode = mode
        self.look_back = 60

    def preprocess(self):
        # 1. Handle Missing Data
        if self.mode == 'drop':
            # B1.1: Simply remove rows with missing values
            df_clean = self.raw_df.dropna()
            desc = "Drop Missing Rows"
        else:
            # B1.2: Interpolate missing values
            df_clean = self.raw_df.interpolate(method='linear')
            desc = "Linear Interpolation"
            
        # 2. Global Normalization
        data = df_clean[['close']].values.astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        return data_scaled, scaler, desc

    def run(self):
        print(f"📉 Running Experiment B1.{'1' if self.mode == 'drop' else '2'} ({self.mode})...")
        
        data_scaled, scaler, preprocessing_desc = self.preprocess()
        
        # 3. Static Split (80% Train, 20% Test) - NO Rolling Window
        train_size = int(len(data_scaled) * 0.8)
        train_data, test_data = data_scaled[0:train_size], data_scaled[train_size:]
        
        # 4. Create Sequences
        def create_dataset(dataset, look_back):
            X, Y = [], []
            for i in range(len(dataset) - look_back - 1):
                X.append(dataset[i:(i+look_back), 0])
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)

        X_train, y_train = create_dataset(train_data, self.look_back)
        X_test, y_test = create_dataset(test_data, self.look_back)
        
        # Reshape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 5. Train Standard LSTM
        model = build_lstm_model((self.look_back, 1))
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        # 6. Predict
        test_predict = model.predict(X_test, verbose=0)
        
        # 7. Inverse Transform
        test_predict_inv = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform([y_test])
        
        # 8. Metrics
        rmse = math.sqrt(mean_squared_error(y_test_inv[0], test_predict_inv[:,0]))
        mae = mean_absolute_error(y_test_inv[0], test_predict_inv[:,0])
        mape = mean_absolute_percentage_error(y_test_inv[0], test_predict_inv[:,0]) * 100
        
        print(f"   Done. RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        
        return {
            "Model_Type": f"B1.{'1' if self.mode == 'drop' else '2'} ({self.mode.capitalize()})",
            "Preprocessing": preprocessing_desc,
            "Normalization": "Global MinMax (Standard)",
            "Training_Strategy": "Static Split (80/20)",
            "Novelty_Contribution": "None (Standard LSTM)",
            "RMSE (Error Cost)": round(rmse, 2),
            "MAE (Avg Error $)": round(mae, 2),
            "MAPE (Error %)": f"{round(mape, 2)}%",
            "Observation": "Better than B0, but suffers from non-stationarity."
        }