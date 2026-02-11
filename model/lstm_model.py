import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    """
    Constructs the LSTM model based on Section 6 of the specs.
    
    Structure:
    Input (180 x 5) -> LSTM(Ret=True) -> Dropout(0.2) -> LSTM -> Dropout(0.2) -> Dense(1)
    """
    model = Sequential([
        # LSTM Layer 1
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        
        # LSTM Layer 2
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        # Output Layer
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model