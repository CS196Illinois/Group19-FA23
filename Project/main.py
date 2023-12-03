# This is the main file for the project. It is the file that is run to start the streamlit app.

import streamlit as st
import functions as f
import game as game
import stock_info as stock_info
import plotly.graph_objects as go

def home() -> None:
    # Website style
    st.markdown(
        """
        <style>
        .main{
        background-color: #4d4dff
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    header = st.container()

    with header: 
        title = 'Welcome to Group 19 Project'
        
        styled_title = f"""
        <h1 style='color: white; text-align: center; font-size: 64px;'>
        {title}</h1>
        <br>
        """
        st.markdown(styled_title, unsafe_allow_html=True)
        st.write("This website has three different pages:")
        st.write("Home - allows you to predict stock prices using three different neural network architectures")
        st.write("Game - allows you to compete against a model to see who can predict the best")
        st.write("Stock Info - allows you to see information about a stock")
        st.write("")

        ticker: str = st.text_input('Enter Stock Ticker Symbol', 'AAPL')
        n_future: int | None = st.selectbox('Timeframe (days)', [5 ,10, 30])
        model_type: str | None = st.selectbox("Choose Model", ["GRU", "LSTM", "Bidirectional LSTM"])
        if model_type == "Bidirectional LSTM":
            model_type = "BiLSTM"

        if st.button("Submit"):
            chart: go.Figure = f.get_standard_chart(ticker, int(n_future), model_type)
            st.plotly_chart(chart)

selected_page = st.sidebar.selectbox(
    'Select a page',
    ['Home', 'Game', 'Stock Info'])

if selected_page == 'Home':
    home()
elif selected_page == 'Game':
    game.game()
elif selected_page == 'Stock Info':
    stock_info.show_stock_info()