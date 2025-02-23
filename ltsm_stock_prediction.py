import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # Convert bytes to KB

def main():
    # Record start time and initial memory
    start_time = time.perf_counter()
    initial_memory = get_memory_usage()

    # Mock Nifty 50 data: 10 days of closing prices (normalized 0-1)
    prices = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
    
    # Prepare data for LSTM: [samples, timesteps, features]
    X, y = [], []
    look_back = 1  # Use 1 previous day to predict the next
    for i in range(len(prices) - look_back):
        X.append(prices[i:i + look_back])
        y.append(prices[i + look_back])
    X = np.array(X).reshape(-1, look_back, 1)  # [samples, timesteps, features]
    y = np.array(y)

    # Define LSTM model
    model = Sequential([
        LSTM(10, activation='tanh', input_shape=(look_back, 1)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=1, verbose=0)

    # Predict the next price
    last_price = np.array([0.55]).reshape(1, look_back, 1)
    prediction = model.predict(last_price, verbose=0)
    print(f"Predicted next Nifty 50 value (normalized): {prediction[0][0]:.2f}")

    # Record end time and final memory
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    final_memory = get_memory_usage()
    memory_used_kb = final_memory - initial_memory

    # Print metrics
    print(f"Execution Time: {elapsed_time_ms:.2f} ms")
    print(f"Memory Used: {memory_used_kb:.0f} KB")

if __name__ == "__main__":
    main()