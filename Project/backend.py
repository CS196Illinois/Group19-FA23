import random
import pandas as pd
import tensorflow as tf
from keras.models import load_model

# Randomly generate list of ten tickers from all possible tickers on yfinance
stock_list = pd.read_csv("Project/nasdaq_screener_1698644339918.csv")["Symbol"].tolist()
random_stocks = random.sample(stock_list, 10)

# The player selects three stocks, and then picks how much to invest in each 
# They also select one of three models

selected_model = None
# Load the model the player selected 
# if model_choice == BNN:
    # model = load_model('model_file')
# elif model_choice == GRU:
    # model = load_model('model_file')
# elif model_choice == LSTM:
    # you have to check whether it's an old stock (in the db) or a new stock and then load accordingly

# Similarly, the model would have to pick amounts to invest for each stock
# For simplicity right now I'm just going to use moving average price +/- %tolerance
def calculate_gain():
    # formula: 100 * (final_price - initial_investment) / initial_investment
    # for a stock, get a db from yfinance
    # then take the average of several opening prices (since this is when most traders start)
    # account for tolerance (2%), then you get initial_investment
    # final_price is the predicted final price after however long the game runs using model.predict()
    # return the result of formula
    pass

three_best_stocks = []
# for stock in random_stocks:
    # gain = calculate_gain()
    # three_best_stocks.append(gain)
# three_best_stocks = sorted(three_best_stocks)[0:3]

# then the model gains are compared to the player gains