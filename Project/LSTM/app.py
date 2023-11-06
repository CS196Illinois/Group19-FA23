import pandas as pd
import streamlit as st
import model as m
import requests

st.title('Stock Price Prediction')
ticker = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
n_future = st.number_input('Enter Number of Days to Predict', 10)

if st.button("Submit"):
    response = requests.post("http://127.0.0.1:8000/predict/", json={"ticker": ticker, "n_future": n_future})
    if response.status_code != 200:
        st.error("Error fetching data from API")
    response_data = response.json()
    merged_df = pd.read_json(response_data['json_merged_df'], orient='split')
    original = pd.read_json(response_data['json_original_df'], orient='split')
    m.StockUtilities.display_predictions(original, n_future, merged_df)