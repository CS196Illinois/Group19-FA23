# This file is what the streamlit app is supposed to interact with. It combines the stock objects
# with the database operations to make function calls in the streamlit app more intuitive.

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
        stock = m.NewStock(ticker, n_future, name)
        model = stock.get_model(type)
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, name, in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data).numpy()
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.original, n_future)
    cached[(ticker, n_future, type)] = merged_df, stock.original

    return merged_df, stock.original if merged and orig else merged_df if merged else stock.original
    
# Returns the chart for the predictions
def get_final_chart(ticker: str, n_future: int, type: str | None) -> go.Figure:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type)
    return m.StockUtilities.display_predictions(original, n_future, merged_df, ticker), merged_df[len(merged_df)-20:]

# Returns the chart for the user to chart their own predictions
def get_user_chart(ticker: str, n_future: int, type: str | None) -> go.Figure:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type, False, True)
    return m.StockUtilities.get_plot(original, n_future, ticker)

# Returns the top 3 picks from the model
def get_model_picks(tickers: list[str], n_future: int, type: str | None) -> list[tuple[str, float]]:
    close_prices: dict[str, float] = {}
    args: list[tuple[str, int, str | None]] = [(ticker, n_future, type) for ticker in tickers]
    for arg in args:
        merged_df = _load_data(arg[0], arg[1], arg[2])[0]
        close_prices[arg[0]] = merged_df['Predicted'].iloc[-1]
    sorted_close_prices = [k for k in sorted(close_prices.items(), key=lambda item: item[1], reverse=True)]
    return sorted_close_prices[:3]



# Clears the cache
def clear_cache() -> None:
    cached.clear()
