import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import timedelta
from keras.optimizers import Adam
from keras.models import Sequential
import database_operations as db_ops
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.preprocessing import StandardScaler
from keras.optimizers.schedules import PiecewiseConstantDecay

# Non-unique functions for getting necessary data for predictions
class StockUtilities:

    @staticmethod
    # Get original data from Yahoo Finance for use in plotting
    def get_original(ticker, start='10y'):
        original = yf.Ticker(ticker)
        original = original.history(period=start)
        original = original.filter(['Close'])
        original = original.reset_index(drop=False)
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
            trainX.append(scaled_data[i - n_past:i, 0])
            trainY.append(scaled_data[i + n_future - 1:i + n_future, 0])

        trainX, trainY = np.array(trainX), np.array(trainY)
        return trainX, trainY

    @staticmethod
    # Get testing data for predictions
    def get_testing_data(scaled_data, past, split):
        testX = []

        for i in range(split, len(scaled_data)):
            testX.append(scaled_data[i - past:i, 0])

        testX = np.array(testX)
        return testX
    
    @staticmethod
    # Predict future returns
    def predict(model, test_data):
        raw_predictions = model.predict(test_data)
        return raw_predictions
    
    @staticmethod
    # Reshape predictions for plotting
    def reshape(raw_pred, scaler, scaled_data, original):
        original_copy = original.copy()
        original_copy.index = original_copy['Date']
        original_copy = original_copy.filter(['Close'])
        original_copy.index = original_copy.index.strftime('%Y-%m-%d')

        raw_pred = np.repeat(raw_pred, scaled_data.shape[1], axis=1)
        predictions = scaler.inverse_transform(raw_pred)[:,0]

        last_price = original['Close'][original.index[-1]]
        predictions[0] = last_price + predictions[0]
        predictions = np.cumsum(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.rename(columns={0:'Predicted'}, inplace=True)
        new_row = pd.DataFrame({'Predicted': [last_price]})
        predictions = pd.concat([new_row, predictions], ignore_index=True)
        predictions.index = predictions.index + original.index[-1]

        pickup = pd.to_datetime(original_copy.index[-1])
        date_range = [pickup + timedelta(days=i) for i in range(0, 11)]
        predictions.index = date_range
        predictions.index = predictions.index.strftime('%Y-%m-%d')

        merged_df = pd.concat([original_copy, predictions], ignore_index=False)

        return merged_df
    
    @staticmethod
    # Plot predictions
    def display_predictions(original, n_future, merged_df):
        start = math.ceil((original.index[-1] - n_future)  * 0.975)
        end = len(original)

        plt.title('Predicted Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.plot(merged_df.index[start:end], merged_df['Close'][start:end], label='Predicted')
        plt.plot(merged_df.index[end - 1:], merged_df['Predicted'][end - 1:], label='Actual')

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=8))
        plt.xticks(fontsize=8)
        plt.xticks(rotation=45)

        st.pyplot(plt)


# Class for new stocks that are not in the database
class NewStock(StockUtilities):

    def __init__(self, ticker, future, start='10y'):
        self.original = self.get_original(ticker, start)
        self.now_index = self.original.index[-1]
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
        model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.train_dataX.shape[1], 1)))
        model.add(LSTM(32, activation='tanh', return_sequences=False))
        model.add(Dense(self.train_dataY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    # Fit model to training data
    def fit_model(self, model):
        model.fit(self.train_dataX, self.train_dataY, epochs=48, batch_size=10, verbose=0)
        return model
    

# Class for stocks that are in the database
class OldStock(StockUtilities):
    
    def __init__(self, ticker, future, start='10y'):
        self.n_past = 45
        self.n_future = future
        self.ticker = ticker
        self.original = self.get_original(ticker, start)
        self.now_index = self.original.index[-1]
        self.last_updated = db_ops.get_last_updated(f'{ticker}.h5')
        self.df = self.get_data(ticker, start)[(self.last_updated - self.n_future - self.n_past):]
        self.df = self.df.reset_index(drop=True)
        self.scaler, self.scaled_data = self.scale_data(self.df)
        self.split = len(self.scaled_data) - self.n_future
        self.train_dataX, self.train_dataY = self.get_training_data(self.scaled_data, future, self.n_past)
        self.test_data = self.get_testing_data(self.scaled_data, self.n_past, self.split)


    # Retrieve model from database
    def get_model(self):
        model = db_ops.get_model_from_db(f'{self.ticker}.h5')
        return model
    
    # Recompile model with new learning rate and fit to new training data
    def fit_model(self, model):
        if self.now_index - self.last_updated > 0:
            new_lr = PiecewiseConstantDecay(
                boundaries=[math.ceil((self.now_index - self.last_updated) * (1/2))],
                values=[0.00066, 0.00033]
            )
            new_optimizer = Adam(learning_rate=new_lr)
            model.compile(optimizer=new_optimizer, loss='mse')
            model.fit(self.train_dataX, self.train_dataY, epochs=5, batch_size=12, verbose=0)

        return model
    
# Example of how to implement using Streamlit
st.title('Stock Price Prediction')
ticker = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
n_future = st.number_input('Enter Number of Days to Predict', 10)

if st.button("Submit"):
    stock = None
    in_db = db_ops.check_if_exists(f'{ticker}.h5')
    if in_db:
        stock = OldStock(ticker, n_future)
    else:
        stock = NewStock(ticker, n_future)
    model = stock.get_model()
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, f'{ticker}.h5', in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data)
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.scaled_data, stock.original)
    stock.display_predictions(stock.original, stock.n_future, merged_df)
