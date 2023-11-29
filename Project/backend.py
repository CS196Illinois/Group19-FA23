import pandas as pd
import random
import yfinance as yf
from datetime import datetime, timedelta
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

# Function to fetch historical stock data for a given company
def fetch_stock_data(company_symbol, start_date, end_date):
    stock_data = yf.download(company_symbol, start=start_date, end=end_date)
    return stock_data

def get_three_stocks(company_symbols):
    # Number of random stocks to fetch
    num_stocks_to_fetch = 3  # You can change this to any number you want

    # Define the parameters for stock data fetching
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2021, 12, 31)

    # Randomly choose stock symbols to fetch data
    random_stocks = random.sample(company_symbols, num_stocks_to_fetch)

    return random_stocks

# Define a list of stock symbols
company_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'FB', 'GOOG', 'NVDA', 'AMD',
                   'JPM', 'GS', 'WFC', 'C', 'BAC', 'XOM', 'CVX', 'BP', 'TOT',
                   'INTC', 'IBM', 'ORCL', 'CSCO', 'QCOM', 'TXN', 'ADBE', 'CRM', 'NOW',
                   'PG', 'KO', 'PEP', 'WMT', 'AMT', 'VZ', 'T', 'TMUS', 'SBUX',
                   'JNJ', 'PFE', 'MRK', 'GSK', 'NVS', 'ABBV', 'LLY', 'BIIB', 'REGN']

# Call the function
selected_stocks = get_three_stocks(company_symbols)
print(f"Selected Stocks: {selected_stocks}")

# Load the pretrained LSTM model
model = load_model("stock_prediction_model.py")
# In reality, this would depend on the model the user chooses but I just did one for test purposes

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

# Run the game with the randomly chosen stocks
game_result = game(selected_stocks, 10, 'LSTM', selected_stocks)
print(game_result)
