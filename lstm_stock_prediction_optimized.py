import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import os

# Print versions for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"yfinance version: {yf.__version__}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # Convert bytes to KB

def fetch_nifty50_data():
    # Explicitly set auto_adjust=False if you want unadjusted prices
    nifty = yf.download('SBIN.NS', start='2023-01-01', end='2025-02-19', progress=False, auto_adjust=False)
    if nifty.empty:
        raise ValueError("Failed to fetch Nifty 50 data.")
    closing_prices = nifty['Close'].dropna().values.astype('float32')
    if len(closing_prices) < 6:
        raise ValueError(f"Insufficient data points: {len(closing_prices)}")
    # Normalize prices to 0-1 range
    min_price, max_price = closing_prices.min(), closing_prices.max()
    normalized_prices = (closing_prices - min_price) / (max_price - min_price)
    last_close = float(closing_prices[-1])  # Convert to Python float
    print(f"Debug: last_close type={type(last_close)}, value={last_close}")
    return normalized_prices, last_close, min_price, max_price

def prepare_data(prices, look_back=5):
    X, y = [], []
    for i in range(len(prices) - look_back):
        X.append(prices[i:i + look_back])
        y.append(prices[i + look_back])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    print(f"Debug: X shape={X.shape}, y shape={y.shape}")
    return X, y

def main():
    start_time = time.perf_counter()
    initial_memory = get_memory_usage()

    # Fetch and normalize data
    try:
        normalized_prices, last_close, min_price, max_price = fetch_nifty50_data()
        # Convert last_close to a Python scalar explicitly for formatting
        print(f"Last Nifty 50 closing price (as of latest data): {last_close:.2f}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Prepare data for LSTM
    look_back = 5
    X, y = prepare_data(normalized_prices, look_back)

    # Define LSTM model
    model = Sequential([
        LSTM(10, activation='tanh', input_shape=(look_back, 1)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Predict the next price
    last_sequence = normalized_prices[-look_back:].reshape(1, look_back, 1)
    pred_normalized = model.predict(last_sequence, verbose=0)[0][0]  # Extract scalar
    pred_price = (pred_normalized * (max_price - min_price)) + min_price
    # Convert pred_price to a Python scalar for formatting
    print(f"Predicted next Nifty 50 closing price: {float(pred_price):.2f}")

    # Record metrics
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000
    final_memory = get_memory_usage()
    memory_used_kb = final_memory - initial_memory

    print(f"Execution Time: {elapsed_time_ms:.2f} ms")
    print(f"Memory Used: {memory_used_kb:.0f} KB")

if __name__ == "__main__":
    main()