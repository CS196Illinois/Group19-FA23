from io import StringIO
import functions as f
import pandas as pd
import streamlit as st
import model as m
import random
import requests

# def fetch_data(ticker: str, n_future: int, model_type: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
#     response = requests.post("http://127.0.0.1:8000/predict/", json={"ticker": ticker, "n_future": n_future, "type": model_type})
#     if response.status_code != 200:
#         st.error("Error fetching data from API")
#     response_data = response.json()
#     merged_df = pd.read_json(StringIO(response_data['json_merged_df']), orient='split')
#     original = pd.read_json(StringIO(response_data['json_original_df']), orient='split')
#     return merged_df, original


st.title('Stock Price Prediction')

# Prevents the app from reloading random 10 stocks on every interaction
@st.cache_resource
def load_data() -> list[str]:
    stock_list: list[str] = pd.read_csv("/Users/jameskendrick/Group19-FA23/Docs/ticker_list.csv")["Symbol"].tolist()
    random_ten: list[str] = random.sample(stock_list, 10)
    return random_ten

random_ten: list[str] = load_data()
tickers: list[str] = st.multiselect('Select 3 stocks', sorted(random_ten), max_selections=3)

n_future: int | float = st.number_input('Enter Number of Days to Predict', 1)
model_type: str | None = st.selectbox("Choose Model To Compete Against", ["GRU", "LSTM", "Bidirectional LSTM"], placeholder="LSTM")
if model_type == "Bidirectional LSTM":
    model_type = "BiLSTM"

if st.button("Submit"):
    for ticker in random_ten:
        st.plotly_chart(f.get_chart(ticker, int(n_future), model_type))
    st.write(f.get_model_picks(tickers, int(n_future), model_type))
    f.clear_cache()
    st.cache_resource.clear()
