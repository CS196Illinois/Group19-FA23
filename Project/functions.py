# This file is what the streamlit app is supposed to interact with. It combines the stock objects
# with the database operations to make function calls in the streamlit app more intuitive.

import yfinance
import model as m
import pandas as pd
import plotly.graph_objects as go
import database_operations as db_ops

# Cache to allow for more adaptable order of operations
cached: dict[tuple[str, int, str | None], tuple[pd.DataFrame, pd.DataFrame]] = {}

# Internal function for getting data needed for predictions
def _load_data(ticker: str, n_future: int, type: str | None, merged: bool=True, orig: bool=True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    if (ticker, n_future, type) in cached:
        return cached[(ticker, n_future, type)]
    stock: m.NewStock | m.OldStock | None = None
    model: m.Sequential | None = None
    name: str = f'{ticker}_{type}_{n_future}.h5'

    in_db = db_ops.check_if_exists(name)
    if in_db:
        stock = m.OldStock(ticker, n_future, name)
        model = stock.get_model(name)
    else:
        stock = m.NewStock(ticker, n_future)
        model = stock.get_model(type)
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, name, in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data).numpy()
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.scaled_data, stock.original, n_future)
    cached[(ticker, n_future, type)] = merged_df, stock.original

    if merged and orig:
        return merged_df, stock.original
    elif merged and not orig:
        return merged_df
    else:
        return stock.original
    
# Returns the chart for the predictions
def get_chart(ticker: str, n_future: int, type: str | None) -> go.Figure:
    if (ticker, n_future, type) in cached:
        data = cached[(ticker, n_future, type)]
        return m.StockUtilities.display_predictions(data[1], n_future, data[0])
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type)
    return m.StockUtilities.display_predictions(original, n_future, merged_df)

# Returns the top 3 picks from the model
def get_model_picks(tickers: list[str], n_future: int, type: str | None) -> list[tuple[str, float]]:
    close_prices: dict[str, float] = {}
    args: list[tuple[str, int, str | None]] = [(ticker, n_future, type) for ticker in tickers]
    for arg in args:
        if arg in cached:
            close_prices[arg[0]] = cached[arg][0]['Predicted'].iloc[-1]
        else:
            merged_df = _load_data(arg[0], arg[1], arg[2], True, False)
            close_prices[arg[0]] = merged_df['Predicted'].iloc[-1]
    sorted_close_prices = [k for k in sorted(close_prices.items(), key=lambda item: item[1], reverse=True)]
    return sorted_close_prices[:3]

def calculate_gain(tickers: list[str], n_future: int, type: str | None):
    final_prices = []
    current_prices = []
    final_gains = []
    args = [(ticker, n_future, type) for ticker in tickers]
    for arg in args:
        if arg in cached:
            final_prices.append(cached[arg][0]['Predicted'].iloc[-1])
            current_prices.append(cached[arg][1]['Open'].iloc[-1])
        else:
            merged_df, original = _load_data(arg[0], arg[1], arg[2])
            final_prices.append(merged_df['Predicted'].iloc[-1])
            current_prices.append(original['Open'].iloc[-1])

    # Second for loop to calculate gain for each stock

    # Calculate final gain
    for i in range(len(final_prices)):
        final_gain = final_prices[i] - (current_prices[0] * 0.98)
        final_gains.append(final_gain)
    return final_gains


def clear_cache() -> None:
    cached = {}

# process in the main backend file
# load data for each stock
# and then do the calculation
# then somehow keep track of top 3 stocks since tuples are immutable and we cannot add a new gain input 
