import math
# from pymongo.mongo_client import MongoClient
# from gridfs import GridFS
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
from keras.models import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

class StockUtilities:

    @staticmethod
    # Get original data from Yahoo Finance for use in plotting
    def get_original(ticker, start='10y'):
        original = yf.Ticker(ticker)
        original = original.history(period=start)
        original = original.filter(['Close'])
        original = original.reset_index(drop=True)
        original = original.loc[1:]
        return original
    
    @staticmethod
    # Get data from Yahoo Finance and calculate daily returns
    def get_data(ticker, start):
        df = yf.Ticker(ticker)
        df = df.history(period=start)
        df['Return'] = df['Close'].shift(-1) - df['Close']
        df = df.filter(['Return'])
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df
    
    @staticmethod
    # Scale data using StandardScaler
    def scale_data(df):
        cols = list(df)[0:1]
        data = df[cols].astype(float)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaler, scaled_data
    
    @staticmethod
    # Get training data for LSTM model
    def get_training_data(scaled_data, future, past):
        n_future = future
        n_past = past
        split = len(scaled_data) - n_future
        trainX = []
        trainY = []

        for i in range(n_past, split):
            trainX.append(scaled_data[i - n_past:i, 0:scaled_data.shape[1]])
            trainY.append(scaled_data[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)
        return trainX, trainY

    @staticmethod
    # Get testing data for predictions
    def get_testing_data(scaled_data, past, split):
        testX = []

        for i in range(split, len(scaled_data)):
            testX.append(scaled_data[i - past:i, 0:scaled_data.shape[1]])

        testX = np.array(testX)
        return testX
    

class NewStock(StockUtilities):

    def __init__(self, ticker, future, start='10y'):
        self.original = self.get_original(ticker, start)
        self.df = self.get_data(ticker, start)
        self.scaler, self.scaled_data = self.scale_data(self.df)
        self.split = len(self.scaled_data) - future
        self.n_future = future
        self.n_past = 45
        self.train_dataX, self.train_dataY = self.get_training_data(self.scaled_data, self.n_future, self.n_past)
        self.test_data = self.get_testing_data(self.scaled_data, self.n_past, self.split)
    
    # Get model for predictions
    def get_model(self):
        model = Sequential()
        model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.train_dataX.shape[1], self.train_dataX.shape[2])))
        model.add(LSTM(32, activation='tanh', return_sequences=False))
        model.add(Dense(self.train_dataY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    # Fit model to training data
    def fit_model(self, model):
        model.fit(self.train_dataX, self.train_dataY, epochs=48, batch_size=10, verbose=0)
        return model

    # Predict future returns
    def predict(self, model):
        raw_predictions = model.predict(self.test_data)
        return raw_predictions

    # Reshape for plotting
    def reshape(self, raw_pred):
        raw_pred = np.repeat(raw_pred, self.scaled_data.shape[1], axis=1)
        predictions = self.scaler.inverse_transform(raw_pred)[:,0]

        predictions[0] = self.original['Close'][self.split + self.n_future] + predictions[0]
        predictions = np.cumsum(predictions)

        predictions = pd.DataFrame(predictions)

        predictions.rename(columns={0:'Predicted'}, inplace=True)

        last_price = self.original['Close'][self.split + self.n_future]
        new_row = pd.DataFrame({'Predicted': [last_price]})
        predictions = pd.concat([new_row, predictions], ignore_index=True)

        predictions.index = predictions.index + self.split + self.n_future

        merged_df = pd.merge(self.original, predictions, left_index=True, right_index=True, how='inner')

        new_rows = pd.DataFrame([None] * self.n_future, columns=['Close'])
        merged_df = pd.concat([merged_df, new_rows], ignore_index=True)
        merged_df.index += self.split + self.n_future
        merged_df['Predicted'] = predictions['Predicted']

        return merged_df
    
    # Create and show plot
    def display_predictions(self, merged_df):
        start = math.ceil(self.split * 0.975)
        plt.title('Predicted Stock Price')
        plt.xlabel('Trading Days Since 10y Ago')
        plt.ylabel('Price')
        plt.plot(merged_df.index, merged_df['Predicted'], label='Predicted')
        plt.plot(self.original.index[start:], self.original['Close'][start:], label='Actual')

        st.pyplot(plt)


st.title('Stock Price Prediction')
ticker = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
n_future = st.number_input('Enter Number of Days to Predict', 10)

if st.button("Submit"):
    # How to use:
    stock = NewStock(ticker, n_future)
    model = stock.get_model()
    model = stock.fit_model(model)
    raw_pred = stock.predict(model)
    merged_df = stock.reshape(raw_pred)
    stock.display_predictions(merged_df)