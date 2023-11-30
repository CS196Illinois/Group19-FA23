import random
import pandas as pd
import LSTM_model as LSTM
import GRU_model as GRU
import BidirectionalNN_model as BNN
import database_operations as db_ops
from keras.models import load_model
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import functions as func
import model as model
import database_operations as db

# Random stocks is a list of tickers
def game(random_stocks, number_of_days, model_type, player_stocks):
    stock_gains = func.calculate_gain(random_stocks, number_of_days, model_type) # list of money earned from each stock
    stock_dict = {random_stocks[i]: stock_gains[i] for i in range(len(random_stocks))} # dictionary of stocks and their gains
    # This if for model
    sorted_stocks = sorted(stock_dict.items(), key=lambda x: x[1], reverse=True)
    top_3_stocks = sorted_stocks[:3] # list of floats
    model_gains = sum(top_3_stocks)
    # Player
    player_gains = 0
    for ticker in player_stocks:
        if ticker in stock_dict.items():
            player_gains += stock_dict[ticker]
    if model_gains > player_gains:
        return "The model won!"
    elif model_gains < player_gains:
        return "You win!"
    else:
        return "Draw!"
