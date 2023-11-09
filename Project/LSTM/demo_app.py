import pandas as pd
import streamlit as st
import model as m
import requests

def fetch_data(ticker: str, n_future: int, model_type: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = requests.post("http://127.0.0.1:8000/predict/", json={"ticker": ticker, "n_future": n_future, "type": model_type})
    if response.status_code != 200:
        st.error("Error fetching data from API")
    response_data = response.json()
    merged_df = pd.read_json(response_data['json_merged_df'], orient='split')
    original = pd.read_json(response_data['json_original_df'], orient='split')
    return merged_df, original


st.title('Stock Price Prediction')
ticker: str = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
n_future: int | float = st.number_input('Enter Number of Days to Predict', 1)
model_type: str | None = st.selectbox("Choose Model To Compete Against", ["GRU", "LSTM", "Bidirectional LSTM"])
if model_type == "Bidirectional LSTM":
    model_type = "BiLSTM"

if st.button("Submit"):
    merged_df, original = fetch_data(ticker, int(n_future), model_type)
    st.pyplot(m.StockUtilities.display_predictions(original, int(n_future), merged_df))
