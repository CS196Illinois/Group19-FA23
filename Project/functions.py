# This file is what the streamlit app is supposed to interact with. It combines the stock objects
# with the database operations to make function calls in the streamlit app more intuitive.

import math
import model as m
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
import database_operations as db_ops

# Cache to allow for more adaptable order of operations
cached: dict[tuple[str, int, str | None], tuple[pd.DataFrame, pd.DataFrame]] = {}

# Internal function for getting data needed for predictions
def _load_data(ticker: str, n_future: int, type: str | None, merged: bool=True, orig: bool=True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    if (ticker, n_future, type) in cached:
        return cached[(ticker, n_future, type)]
    stock: m.SetbackNewStock | m.SetbackOldStock | None = None
    model: m.Sequential | None = None
    name: str = f'{ticker}_{type}_{n_future}.h5'

    in_db = db_ops.check_if_exists(name)
    if in_db:
        stock = m.SetbackOldStock(ticker, n_future, name)
        model = stock.get_model(name)
    else:
        stock = m.SetbackNewStock(ticker, n_future, name)
        model = stock.get_model(type)
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, name, in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data).numpy()
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.original, n_future)
    cached[(ticker, n_future, type)] = merged_df, stock.original

    return merged_df, stock.original if merged and orig else merged_df if merged else stock.original

def _load_data_normal(ticker: str, n_future: int, type: str | None, merged: bool=True, orig: bool=True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    stock: m.NewStock | m.OldStock | None = None
    model: m.Sequential | None = None
    name: str = f'{ticker}_{type}_{n_future}_original.h5'

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

    return merged_df, stock.original if merged and orig else merged_df if merged else stock.original
    
# Returns the chart for the predictions
def get_setback_final_chart(ticker: str, n_future: int, type: str | None) -> tuple[go.Figure, pd.DataFrame]:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type)
    return m.SetbackStockUtilities.display_predictions(original, n_future, merged_df, ticker), merged_df[len(merged_df)-20:]

# Returns the chart for the user to chart their own predictions
def get_setback_user_chart(ticker: str, n_future: int, type: str | None) -> go.Figure:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type, False, True)
    return m.SetbackStockUtilities.get_plot(original, n_future, ticker)

# Returns the top 3 picks from the model
def get_model_picks(tickers: list[str], n_future: int, type: str | None) -> list[str]:
    close_prices: dict[str, float] = {}
    args: list[tuple[str, int, str | None]] = [(ticker, n_future, type) for ticker in tickers]
    for arg in args:
        merged_df, _ = _load_data(arg[0], arg[1], arg[2])
        close_prices[arg[0]] = merged_df['Predicted'].iloc[-1]
    sorted_close_prices: list[str] = [k for k in sorted(close_prices.keys(), key=lambda x: close_prices[x], reverse=True)]
    return sorted_close_prices[:3]

# Get the model's prediction
def get_prices(ticker: str, n_future: int, type: str | None) -> tuple[float, float]:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data(ticker, n_future, type)
    return merged_df['Predicted'].iloc[-1], original['Close'].iloc[-1]


# Get the combined graph of the model's top 3 picks and the winner
def get_gain_result_chart(model_tickers: list[str], user_tickers: list[str], n_future: int, type: str | None) -> tuple[go.Figure, str]:
    model_original: pd.DataFrame | pd.Series[any] = cached[(model_tickers[0], n_future, type)][1]
    user_original: pd.DataFrame | pd.Series[any] = cached[(user_tickers[0], n_future, type)][1]
    model_original['Return'] = model_original['Close'].shift(-1) - model_original['Close']
    user_original['Return'] = user_original['Close'].shift(-1) - user_original['Close']
    model_original.dropna()
    user_original.dropna()
    model_tickers.sort()
    user_tickers.sort()

    print(model_tickers)
    print(model_tickers[0])
    for ticker in model_tickers[1:]:
        print(ticker)
        cur_model_original = cached[(ticker, n_future, type)][1]
        cur_model_original['Return'] = cur_model_original['Close'].shift(-1) - cur_model_original['Close']
        cur_model_original.dropna()
        model_original['Return'] += cur_model_original['Return']
    
    for ticker in user_tickers[1:]:
        cur_user_original = cached[(ticker, n_future, type)][1]
        cur_user_original['Return'] = cur_user_original['Close'].shift(-1) - cur_user_original['Close']
        cur_user_original.dropna()
        user_original['Return'] += cur_user_original['Return']

    start = math.ceil((len(model_original) - n_future))

    model_original = model_original.iloc[start:]
    user_original = user_original.iloc[start:]
    model_original["Net Gain"] = model_original["Return"].cumsum()
    user_original["Net Gain"] = user_original["Return"].cumsum()
    model_original.dropna(inplace=True)
    user_original.dropna(inplace=True)
    model_original["Date"] = pd.to_datetime(model_original["Date"]).dt.strftime('%Y-%m-%d')
    user_original["Date"] = pd.to_datetime(user_original["Date"]).dt.strftime('%Y-%m-%d')

    trace1 = go.Scatter(x=model_original["Date"], y=model_original['Net Gain'], mode='lines+markers', name='Model Picks', line=dict(color='blue'))
    trace2 = go.Scatter(x=user_original["Date"], y=user_original['Net Gain'], mode='lines+markers', name='Your Picks', line=dict(color='red'))

    layout = go.Layout(title=f'Net Gain/Loss of Model vs. Your Picks', 
                        xaxis=dict(title='Date'), 
                        yaxis=dict(title='Close Price'),
                        )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    winner: str
    model_gain: float = model_original['Net Gain'].iloc[len(model_original)-1]
    user_gain: float = user_original['Net Gain'].iloc[-1]
    if model_gain > user_gain:
        winner = "Model"
    elif user_gain > model_gain:
        winner = "User"
    else:
        winner = "Tie"

    return fig, winner

# Reutrns the normal standard prediction chart
def get_standard_chart(ticker: str, n_future: int, type: str | None) -> go.Figure:
    merged_df: pd.DataFrame
    original: pd.DataFrame
    merged_df, original = _load_data_normal(ticker, n_future, type)
    return m.StockUtilities.display_predictions(original, n_future, merged_df)

def get_normal_chart(ticker: str, time_frame: str) -> go.Figure:
    original = yf.Ticker(ticker).history(period=time_frame)
    original["Date"] = pd.to_datetime(original.index).strftime('%Y-%m-%d')
    trace = go.Scatter(x=original["Date"], y=original['Close'], mode='lines+markers', name='Close Price', line=dict(color='blue'))
    layout = go.Layout(title=f'Close Price of {ticker} Over the Past {time_frame}', 
                        xaxis=dict(title='Date'), 
                        yaxis=dict(title='Close Price'),
                        )
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# Clears the cache
def clear_cache() -> None:
    cached.clear()
