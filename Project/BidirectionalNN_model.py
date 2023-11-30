import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
import matplotlib.pyplot as plt
from keras.regularizers import l2

# Fetch AAPL historical stock price data from Yahoo Finance
ticker_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2021-12-31"

# Download historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Preprocess the data
data['Date'] = pd.to_datetime(data.index)  # Set the index as 'Date'
data = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Create sequences and labels
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 30  # Adjust as needed
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the Bi-RNN model
model = Sequential()
model.add(Bidirectional(LSTM(50, input_shape=(seq_length, 1))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get actual stock prices
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with Bi-RNN')
plt.show()

# Predict the next 30 days
next_30_days_data = data[-seq_length:]
scaled_data = scaler.transform(next_30_days_data.reshape(-1, 1))
predictions = []

# Iterate through each day
for day in range(30):
    # Make a prediction for the next day
    prediction = model.predict(scaled_data.reshape(1, seq_length, 1))

    # Append the prediction to the list
    predictions.append(prediction[0][0])

    # Update the input data for the next iteration
    scaled_data = np.roll(scaled_data, shift=-1)
    scaled_data[-1] = prediction

# Inverse transform the predictions to get actual stock prices
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Optimize (if needed)
model_optimized = Sequential()
model_optimized.add(Bidirectional(LSTM(50, input_shape=(seq_length, 1), kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01))))
model_optimized.add(Dense(1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted')
plt.legend()
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price (Dollars)')
plt.title('Stock Price Prediction with Bi-RNN for Next 30 Days')
plt.show()
