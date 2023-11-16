import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from keras.optimizers import Adam
from keras.models import Sequential
import database_operations as db_ops
import plotly.graph_objects as go
from keras.layers import LSTM, Dense, GRU, Bidirectional
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sklearn.preprocessing import StandardScaler
from keras.optimizers.schedules import PiecewiseConstantDecay

# Non-unique functions for getting necessary data for predictions
class StockUtilities:

    @staticmethod
    # Get original data from Yahoo Finance for use in plotting
    def get_original(ticker: str, start: str ='10y') -> pd.DataFrame:
        original: pd.DataFrame = yf.Ticker(ticker)
        original = original.history(period=start)
        original = original.filter(['Close'])
        original = original.reset_index(drop=False)
        # new_rows = pd.DataFrame({'Close': [201.66, 210.88, 208.97, 222.18]})
        # original = pd.concat([new_rows, original], ignore_index=True)
        original = original.loc[:len(original) - 1]
        return original
    
    @staticmethod
    # Get data from Yahoo Finance and calculate daily returns
    def get_data(ticker: str, start: str ='10y') -> pd.DataFrame:
        df: pd.DataFrame = yf.Ticker(ticker)
        df = df.history(period=start)
        df['Return'] = df['Close'].shift(-1) - df['Close']
        df = df.filter(['Return'])
        df = df.dropna()
        df = df.reset_index(drop=True)
        # new_rows = pd.DataFrame({'Return': [6.5, -4.5, -1.1, 3.4]})
        # df = pd.concat([new_rows, df], ignore_index=True)
        return df
    
    @staticmethod
    # Scale data using StandardScaler
    def scale_data(df : pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
        cols: list = list(df)[0:1]
        data: pd.DataFrame = df[cols].astype(float)
        scaler = StandardScaler()
        scaled_data: np.ndarray = scaler.fit_transform(data)
        return scaler, scaled_data
    
    @staticmethod
    # Get training data for LSTM model
    def get_training_data(scaled_data: np.ndarray, future: int, past: int) -> tuple[np.ndarray, np.ndarray]:
        n_future: int = future
        n_past: int = past
        split: int = len(scaled_data) - n_future
        trainX: list = []
        trainY: list = []

        for i in range(n_past, split):
            trainX.append(scaled_data[i - n_past:i, 0])
            trainY.append(scaled_data[i + n_future - 1:i + n_future, 0])

        train_X: np.ndarray
        train_Y: np.ndarray
        train_X, train_Y = np.array(trainX), np.array(trainY)
        return train_X, train_Y

    @staticmethod
    # Get testing data for predictions
    def get_testing_data(scaled_data: np.ndarray, past: int, split: int) -> np.ndarray:
        testX: list = []

        for i in range(split, len(scaled_data)):
            testX.append(scaled_data[i - past:i, 0])

        
        test_X: np.ndarray = np.array(testX)
        return test_X
    
    @staticmethod
    # Predict future returns
    def predict(model: Sequential, test_data: np.ndarray) -> np.ndarray:
        raw_predictions: np.ndarray = model.predict(test_data)
        return raw_predictions
    
    @staticmethod
    # Reshape predictions for plotting
    def reshape(raw_pred: np.ndarray, scaler: StandardScaler, scaled_data: np.ndarray, original: pd.DataFrame, n_future: int) -> pd.DataFrame:
        original_copy: pd.DataFrame = original.copy()
        original_copy.index = original_copy['Date']
        original_copy = original_copy.filter(['Close'])
        original_copy.index = original_copy.index.strftime('%Y-%m-%d')

        # raw_pred = np.repeat(raw_pred, scaled_data.shape[1], axis=1)
        predictions: np.ndarray | pd.DataFrame = scaler.inverse_transform(raw_pred)[:,0]

        last_price: float = original['Close'][original.index[-1]]
        predictions[0] = last_price + predictions[0]
        predictions = np.cumsum(predictions)
        predictions = pd.DataFrame(predictions)
        predictions.rename(columns={0:'Predicted'}, inplace=True)
        new_row: pd.DataFrame = pd.DataFrame({'Predicted': [last_price]})
        predictions = pd.concat([new_row, predictions], ignore_index=True)
        predictions.index = predictions.index + original.index[-1]

        pickup: pd.Timestamp = pd.to_datetime(original_copy.index[-1])
        date_range = [pickup + timedelta(days=i) for i in range(0, n_future + 1)]
        predictions.index = date_range
        predictions.index = predictions.index.strftime('%Y-%m-%d')

        merged_df = pd.concat([original_copy, predictions], ignore_index=False)

        return merged_df
    
    @staticmethod
    # Plot predictions
    def display_predictions(original: pd.DataFrame, n_future: int, merged_df: pd.DataFrame) -> plt.figure:
        start = math.ceil((original.index[-1] - n_future)  * 0.975)
        end = len(original)

        trace1 = go.Scatter(x=merged_df.index[start:end], y=merged_df['Close'][start:end], mode='lines+markers', name='Actual', line=dict(color='blue'))
        trace2 = go.Scatter(x=merged_df.index[end - 1:], y=merged_df['Predicted'][end - 1:], mode='lines+markers', name='Predicted', line=dict(color='red'))

        # Create the layout
        layout = go.Layout(title='Predicted Stock Price', 
                           xaxis=dict(title='Date'), 
                           yaxis=dict(title='Close Price'),
                           )

        # Create the figure
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        return fig


# Class for new stocks that are not in the database
class NewStock(StockUtilities):

    def __init__(self, ticker: str, future: int, start: str='10y') -> None:
        self.original: pd.DataFrame = self.get_original(ticker, start)
        self.now_index: int = self.original.index[-1]
        self.df: pd.DataFrame = self.get_data(ticker, start)
        self.scaler: StandardScaler
        self.scaled_data: np.ndarray
        self.scaler, self.scaled_data = self.scale_data(self.df)
        self.split: int = len(self.scaled_data) - future
        self.n_future: int = future
        self.n_past: int = 45
        self.train_dataX: np.ndarray
        self.train_dataY: np.ndarray
        self.train_dataX, self.train_dataY = self.get_training_data(self.scaled_data, self.n_future, self.n_past)
        self.test_data: np.ndarray = self.get_testing_data(self.scaled_data, self.n_past, self.split)
    
    # Get model for predictions
    def get_model(self, type: str) -> Sequential:
        model: Sequential = Sequential()
        if type == 'LSTM':
            model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.train_dataX.shape[1], 1)))
            model.add(LSTM(32, activation='tanh', return_sequences=False))
            model.add(Dense(self.train_dataY.shape[1]))
            model.compile(optimizer='adam', loss='mse')
        elif type == 'GRU':
            model.add(GRU(64, activation='tanh', return_sequences=True, input_shape=(self.train_dataX.shape[1], 1)))
            model.add(GRU(32, activation='tanh', return_sequences=False))
            model.add(Dense(self.train_dataY.shape[1]))
            model.compile(optimizer='adam', loss='mse')
        else:
            self.train_dataX = np.expand_dims(self.train_dataX, axis=-1)
            model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.train_dataX.shape[1], 1))))
            model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=False)))
            model.add(Dense(self.train_dataY.shape[1]))
            model.compile(optimizer='adam', loss='mse')
        return model
    
    # Fit model to training data
    def fit_model(self, model: Sequential) -> Sequential:
        model.fit(self.train_dataX, self.train_dataY, epochs=48, batch_size=10, verbose=0)
        return model
    

# Class for stocks that are in the database
class OldStock(StockUtilities):
    
    def __init__(self, ticker: str, n_future: int, name: str, start: str='10y') -> None:
        self.n_past: int = 45
        self.ticker: str = ticker
        self.original: pd.DataFrame = self.get_original(ticker, start)
        self.now_index: int = self.original.index[-1]
        self.last_updated: int = db_ops.get_last_updated(name)
        self.df: pd.DataFrame = self.get_data(ticker, start)[(self.last_updated - n_future - self.n_past):]
        self.df = self.df.reset_index(drop=True)
        self.scaler: StandardScaler
        self.scaled_data: np.ndarray
        self.scaler, self.scaled_data = self.scale_data(self.df)
        self.split: int = len(self.scaled_data) - n_future
        self.train_dataX: np.ndarray
        self.train_dataY: np.ndarray
        self.train_dataX, self.train_dataY = self.get_training_data(self.scaled_data, n_future, self.n_past)
        self.test_data: np.ndarray = self.get_testing_data(self.scaled_data, self.n_past, self.split)


    # Retrieve model from database
    def get_model(self, name: str) -> Sequential:
        model: Sequential = db_ops.get_model_from_db(name)
        return model
    
    # Recompile model with new learning rate and fit to new training data
    def fit_model(self, model: Sequential) -> Sequential:
        if self.now_index - self.last_updated > 0:
            new_lr = PiecewiseConstantDecay(
                boundaries=[math.ceil((self.now_index - self.last_updated) * (1/2))],
                values=[0.00066, 0.00033]
            )
            new_optimizer = Adam(learning_rate=new_lr)
            model.compile(optimizer=new_optimizer, loss='mse')
            model.fit(self.train_dataX, self.train_dataY, epochs=5, batch_size=12, verbose=0)

        return model
