import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import GRU, Dense

# Fetch AAPL historical stock price data from Yahoo Finance
ticker_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2021-12-31"

# Download historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Extract the 'Close' prices
data = data[['Close']].values

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Create sequences of historical data for training
def create_sequences(data, seq_length, horizon):
    sequences = []
    for i in range(len(data) - seq_length - horizon):
        X = data[i:i + seq_length]
        y = data[i + seq_length + horizon]  # Adjust the horizon here
        sequences.append((X, y))
    return np.array(sequences)

seq_length = 10  # Length of historical sequences
prediction_horizon = 1 # 30 trading days = 1 month (assuming daily data)

train_sequences = create_sequences(train_data, seq_length, prediction_horizon)
test_sequences = create_sequences(test_data, seq_length, prediction_horizon)

# Prepare data for training
X_train = np.array([seq[0] for seq in train_sequences])
y_train = np.array([seq[1] for seq in train_sequences])
X_test = np.array([seq[0] for seq in test_sequences])
y_test = np.array([seq[1] for seq in test_sequences])

# Convert NumPy arrays to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Build a GRU-based model
model = Sequential()
model.add(GRU(units=50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')

# Add labels and legend
plt.title(f'{ticker_symbol} Stock Price Prediction (1 Month Horizon)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.axvline(x=len(y_train), color='gray', linestyle='--', label='Training End')
plt.legend()

# Show gridlines
plt.grid(True)

# Highlight the prediction period
plt.fill_between(
    range(len(y_train), len(y_train) + len(y_test)),
    min(y_test.min(), y_pred.min()) - 10,
    max(y_test.max(), y_pred.max()) + 10,
    color='lightgray',
    alpha=0.5,
    label='Prediction Period'
)

plt.show()
