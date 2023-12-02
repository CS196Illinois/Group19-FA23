# This file is an example of how you could use the functions in functions.py to create a streamlit app.


import functions as f
import pandas as pd
import streamlit as st
import random

# Title
st.markdown("<h1 style='text-align: center;'>Stock Price Prediction Game</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Choose your stocks:</h3>", unsafe_allow_html=True)

# Prevents the app from reloading random 10 stocks on every interaction
@st.cache_resource
def get_tickers() -> list[str]:
    stock_list: list[str] = pd.read_csv("/Users/jameskendrick/Group19-FA23/Docs/ticker_list.csv")["Symbol"].tolist()
    random_ten: list[str] = random.sample(stock_list, 10)
    return random_ten

# random_ten: list[str] = get_tickers()
# Manually setting just for testing purposes
random_ten = ["AAPL", "MSFT", "AMZN", "GOOG", "AMD", "TSLA", "NVDA", "META", "ADBE"]
tickers: list[str] = st.multiselect('Select 3 stocks', sorted(random_ten), max_selections=3)

n_future: int | None = st.selectbox('TImeframe (days)', [5 ,10, 30])
model_type: str | None = st.selectbox("Choose your model to compete against", ["GRU", "LSTM", "Bidirectional LSTM"], placeholder="LSTM")
if model_type == "Bidirectional LSTM":
    model_type = "BiLSTM"


if st.button("Submit"):
    model_picks_list = f.get_model_picks(random_ten, int(n_future), model_type)
    st.write(f"The model's top 3 picks are: {model_picks_list}")
    st.write(f"Your top 3 picks are: {tickers}")
    chart, winner = f.get_gain_result_chart(model_picks_list, tickers, int(n_future), model_type)
    st.plotly_chart(chart)
    if winner == "Model":
        st.markdown("<h2 style='text-align: center;'>The model won :(</h3>", unsafe_allow_html=True)
    elif winner == "User":
        st.markdown("<h2 style='text-align: center;'>You won :)</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center;'>You tied :|</h3>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Model's forecasts for it's picks:</h3>", unsafe_allow_html=True)
    for ticker in model_picks_list:
        chart, merged_df = f.get_final_chart(ticker, int(n_future), model_type)
        st.plotly_chart(chart)
    
    st.markdown("<h3 style='text-align: center;'>Model's forecasts for your picks:</h3>", unsafe_allow_html=True)
    for ticker in tickers:
        chart, merged_df = f.get_final_chart(ticker, int(n_future), model_type)
        st.plotly_chart(chart)

    f.clear_cache()
    st.cache_resource.clear()


# user_prediction = st.number_input('Enter Your Prediction', min_value=0.0, max_value=None, value=0.0, step=0.01)
# if st.button("Submit Prediction"):
#     model_prediction, actual_price = f.get_prices(tickers[0], int(n_future), model_type)
#     if (abs(user_prediction - actual_price) > abs(model_prediction - actual_price)):
#         st.write("You lost to the model :(")
#     elif (abs(user_prediction - actual_price) < abs(model_prediction - actual_price)):
#         st.write("You beat the model :)")
#     else:
#         st.write("You tied the model :|")


    # (f.get_user_chart(tickers[0], int(n_future), model_type)).show(config = {'modeBarButtonsToAdd':['drawline',
    #                                     'drawopenpath',
    #                                     'drawclosedpath',
    #                                     'drawcircle',
    #                                     'drawrect',
    #                                     'eraseshape'
    #                                    ]})