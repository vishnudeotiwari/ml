import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"yfinance version: {yf.__version__}")

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # Convert bytes to KB

# Fetch stock data with fixed yfinance settings & robust RSI
def fetch_nifty50_data():
    nifty = yf.download('SBIN.NS', start='2023-01-01', end='2025-02-19', auto_adjust=False, progress=True)

    if nifty.empty:
        raise ValueError("Failed to fetch data. Check internet connection or ticker.")

    # Fill missing values properly
    nifty.ffill(inplace=True)
    nifty.bfill(inplace=True)

    # Technical Indicators
    nifty['SMA_10'] = nifty['Close'].rolling(window=10).mean()
    nifty['EMA_10'] = nifty['Close'].ewm(span=10, adjust=False).mean()
    
    # RSI Calculation Fix
    delta = nifty['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    nifty['RSI'] = 100 - (100 / (1 + rs))

    # Normalize features
    features = ['Close', 'SMA_10', 'EMA_10', 'RSI']
    nifty = nifty[features].dropna()
    min_vals, max_vals = nifty.min(), nifty.max()
    normalized_data = (nifty - min_vals) / (max_vals - min_vals)

    return normalized_data, min_vals, max_vals

# Prepare data for LSTM
def prepare_data(prices, look_back=5):
    X, y = [], []
    for i in range(len(prices) - look_back):
        X.append(prices[i:i + look_back])
        y.append(prices[i + look_back]['Close'])  # Ensure y is 1D by selecting 'Close' column
    
    X = np.array(X)
    y = np.array(y).flatten()  # Ensure y is 1D
    
    print(f"Debug: X shape={X.shape}, y shape={y.shape}")  # Add debug prints
    return X, y

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Multi-day prediction function
def predict_next_days(model, last_sequence, min_vals, max_vals, num_days=5):
    future_prices = []
    for _ in range(num_days):
        pred_normalized = model.predict(last_sequence.reshape(1, *last_sequence.shape), verbose=0)[0][0]
        pred_price = (pred_normalized * (max_vals['Close'] - min_vals['Close'])) + min_vals['Close']
        future_prices.append(pred_price)
        new_row = np.append(last_sequence[0, 1:], [[pred_normalized, 0, 0, 0]], axis=0)
        last_sequence = new_row.reshape(1, last_sequence.shape[1], last_sequence.shape[2])
    return future_prices

def main():
    start_time = time.perf_counter()
    initial_memory = get_memory_usage()

    try:
        data, min_vals, max_vals = fetch_nifty50_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    look_back = 5
    X, y = prepare_data(data, look_back)

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build & train model
    model = build_lstm_model((look_back, X.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict next 5 days
    last_sequence = X[-1].reshape(1, look_back, X.shape[2])
    predicted_prices = predict_next_days(model, last_sequence, min_vals, max_vals)

    print(f"Predicted prices for next 5 days: {predicted_prices}")

    end_time = time.perf_counter()
    memory_used_kb = get_memory_usage() - initial_memory
    print(f"Execution Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"Memory Used: {memory_used_kb:.0f} KB")

if __name__ == "__main__":
    main()
