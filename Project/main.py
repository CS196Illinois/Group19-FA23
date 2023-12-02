import streamlit as st
import backend as model

def game():    
    # Title
    st.markdown("<h1 style='text-align: center;'>Stock Price Prediction Game</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Choose your stocks:</h3>", unsafe_allow_html=True)

        # Add stop options dropdowns
    stock1 = st.selectbox("Stock 1", ["AAPL", "TSLA", "NKE"])
    stock2 = st.selectbox("Stock 2", ["AAPL", "TSLA", "NKE"])
    stock3 = st.selectbox("Stock 3", ["AAPL", "TSLA", "NKE"])

    st.markdown("<h3 style='text-align: center;'>Choose your timeframe:</h3>", unsafe_allow_html=True)

    timeFrame = st.selectbox("Timeframe", [1, 7])

    st.markdown("<h3 style='text-align: center;'>Choose your model to compete against:</h3>", unsafe_allow_html=True)

    modelSelection = st.selectbox("Choose Model To Compete Against", ["GRU", "LSTM", "RNN"])

    # submit button
    if st.button("Submit"):
        result = model.game(["AAPL", "TSLA", "NKE"], timeFrame, modelSelection, [stock1, stock2, stock3])
        st.markdown("<h2 style='text-align: center;'>Results:</h2>", unsafe_allow_html=True)  # Display "You Win" message
        if result == 1:
            st.markdown("<h2 style='text-align: center;'>You Win!</h2>", unsafe_allow_html=True)  # Display "You Win" message
        elif result == -1:
            st.markdown("<h2 style='text-align: center;'>You Lose!</h2>", unsafe_allow_html=True)  # Display "You Lose" message
        elif result == 0:
            st.markdown("<h2 style='text-align: center;'>Draw!</h2>", unsafe_allow_html=True)  # Display "You Lose" message
