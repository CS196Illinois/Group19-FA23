# This file is an example of how you could use the functions in functions.py to create a streamlit app.


import functions as f
import pandas as pd
import streamlit as st
import random


st.title('Stock Price Prediction')

# Prevents the app from reloading random 10 stocks on every interaction
@st.cache_resource
def get_tickers() -> list[str]:
    stock_list: list[str] = pd.read_csv("/Users/jameskendrick/Group19-FA23/Docs/ticker_list.csv")["Symbol"].tolist()
    random_ten: list[str] = random.sample(stock_list, 10)
    return random_ten

# random_ten: list[str] = get_tickers()
# Manually setting just for testing purposes
random_ten = ["AAPL", "MSFT", "AMZN"]
tickers: list[str] = st.multiselect('Select 3 stocks', sorted(random_ten), max_selections=3)

n_future: int | None = st.selectbox('Enter Number of Days to Predict', [5 ,10, 30])
model_type: str | None = st.selectbox("Choose Model To Compete Against", ["GRU", "LSTM", "Bidirectional LSTM"], placeholder="LSTM")
if model_type == "Bidirectional LSTM":
    model_type = "BiLSTM"

user_prediction: float
model_prediction: float
actual_price: float
if st.button("Submit"):
    (f.get_user_chart(tickers[0], int(n_future), model_type)).show(config = {'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]})
    for ticker in random_ten:
        chart, merged_df = f.get_final_chart(ticker, int(n_future), model_type)
        st.plotly_chart(chart)
        st.write(merged_df)
    st.write(f.get_model_picks(tickers, int(n_future), model_type))

    f.clear_cache()
    st.cache_resource.clear()

user_prediction = st.number_input('Enter Your Prediction', min_value=0.0, max_value=None, value=0.0, step=0.01)
if st.button("Submit Prediction"):
    model_prediction, actual_price = f.get_prices(tickers[0], int(n_future), model_type)
    if (abs(user_prediction - actual_price) > abs(model_prediction - actual_price)):
        st.write("You lost to the model :(")
    elif (abs(user_prediction - actual_price) < abs(model_prediction - actual_price)):
        st.write("You beat the model :)")
    else:
        st.write("You tied the model :|")
