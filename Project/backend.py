import random
import pandas as pd
import yfinance as yf
import LSTM_model as LSTM
import database_operations as db_ops
from keras.models import load_model

# Load the pretrained LSTM model
model = load_model("stock_prediction_model.py")
# In reality, this would depend on the model the user chooses but I just did one for test purposes

# Randomly generate list of stocks
stock_list = pd.read_csv("nasdaq_screener_1698644339918.csv")["Symbol"].tolist()
random_stocks = random.sample(stock_list, 10)

# Helper function to calculate gain
def calculate_gain(ticker, model):
    stock = None
    name_in_db = f"{ticker}.h5"
    in_db = db_ops.check_if_exists(name_in_db)
    if in_db:
        stock = LSTM.OldStock()
    else:
        stock = LSTM.NewStock()
    model = stock.get_model()
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, name_in_db, in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data)
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.scaled_data, stock.original)
    final_price = merged_df['Predictions'].iloc[-1]

    df = yf.Ticker(ticker)
    df = df.history(period="200d")

    # Calculate initial investment
    avg_opening_price = df['Open'].mean()
    tolerance = 0.02
    initial_investment = avg_opening_price * (1 + tolerance)

    # Calculate final gain
    final_gain = 100 * (final_price - initial_investment) / initial_investment
    return final_gain

three_best_stocks = []
for stock in random_stocks:
    gain = calculate_gain(stock)
    three_best_stocks.append((stock, gain))

# Sort stocks by gain and then return predicted top 3
three_best_stocks = sorted(three_best_stocks, key=lambda x: x[1], reverse=True)[:3]
print(three_best_stocks)

# then the model gains are compared to the player gains