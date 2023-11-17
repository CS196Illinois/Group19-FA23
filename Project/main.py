import streamlit as st

# Title
st.markdown("<h1 style='text-align: center;'>Stock Price Prediction Game</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>Choose your stocks:</h3>", unsafe_allow_html=True)

# Add stop options dropdowns
stock1 = st.selectbox("Stock 1", ["Apple", "Tesla", "Nike"])
stock2 = st.selectbox("Stock 2", ["Apple", "Tesla", "Nike"])
stock3 = st.selectbox("Stock 3", ["Apple", "Tesla", "Nike"])

st.markdown("<h3 style='text-align: center;'>Choose your timeframe:</h3>", unsafe_allow_html=True)

timeFrame = st.selectbox("Timeframe", ["Half Day", "Day", "Week"])

st.markdown("<h3 style='text-align: center;'>Choose your model to compete against:</h3>", unsafe_allow_html=True)

modelSelection = st.selectbox("Choose Model To Compete Against", ["GRU", "LSTM", "RNN"])

# submit button
if st.button("Submit"):
    st.markdown("<h2 style='text-align: center;'>You Win!</h2>", unsafe_allow_html=True)  # Display "You Win" message
    st.markdown("<h2 style='text-align: center;'>Results:</h2>", unsafe_allow_html=True)  # Display "You Win" message
