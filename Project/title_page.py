import streamlit as st 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Bidirectional, LSTM
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from tensorflow.keras.regularizers import l2
import numpy as np
import main 

def home():
    # website style 
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



    @st.cache_data
    def get_data(filename):
        data = pd.read_csv(filename)
        return data

    # instantiate blocks
    header = st.container()
    dataset = st.container()
    features = st.container()
    model_training = st.container()

    #description block
    with header: 
        summary = """This website will allow you to predict stock prices for 30 days in the future.
        Eventually we will create a portfolio game where players can try different models and create
        stock portfolios and check who is more successful
        """
        title = 'Welcome to Group 19 Project'

        styled_title = f"""
        <h1 style='color: black; text-align: center; font-size: 64px;'>
        {title}</h1>
        <br>
        """
        st.markdown(styled_title, unsafe_allow_html=True)
        st.write(summary)

        st.write('Users can select through different Stocks and see how the model will predict growth or decline.')

        sel_col, disp_col =st.columns(2)

        stock_name = sel_col.selectbox('Choose your Stock', options=['Select','AAPL', 'NKE', 'TSLA'], index=0)


    #Set stock
    data = get_data('/Users/yugankmac/Downloads/AAPL.csv')
    historical_data = data
    stocks = ['/Users/yugankmac/Downloads/AAPL.csv', '/Users/yugankmac/Downloads/TSLA.csv', '/Users/yugankmac/Downloads/TSLA.csv']
    if stock_name == 'AAPL' :
        data = get_data(stocks[0])
        historical_data = get_data(stocks[0])
    elif stock_name == 'NKE':
        data = get_data(stocks[1])
        historical_data = get_data(stocks[1])
    elif stock_name == 'TSLA':
        data = get_data(stocks[2])
        historical_data = get_data(stocks[2])

    # Preprocess the data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data['Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Split the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Create sequences and labels
    def create_sequences(data, seq_length):
        sequences, labels = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            label = data[i+seq_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    seq_length = 30  # Adjust as needed
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Build the Bi-RNN model
    model = Sequential()
    model.add(Bidirectional(LSTM(50, input_shape=(seq_length, 1))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform the predictions to get actual stock prices
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label='Predicted')
    plt.plot(y_test, label='Actual')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction with Bi-RNN')
    plt.show()


    with dataset:

        

        if stock_name != 'Select' :
            st.header('Stock Price Model:')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(predictions, label='Predicted')
            ax.plot(y_test, label='Actual')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Stock Price')
            ax.set_title('Stock Price Prediction with Bi-RNN')
            st.pyplot(fig)
        

    # Load historical data

    next_30_days_data = historical_data.tail(30)
    # Preprocess
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(next_30_days_data['Close'].values.reshape(-1, 1))

    # array to store predictions
    predictions = []

    # Iterate through each day
    for day in range(1, 31):
        # Make a prediction for the next day
        prediction = model.predict(scaled_data.reshape(1, 30, 1))

        # Append the prediction to the list
        predictions.append(prediction[0][0])

        # Update the input data for the next iteration
        scaled_data = np.roll(scaled_data, shift=-1)
        scaled_data[-1] = prediction

    # Inverse transform the predictions to get actual stock prices
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    #Optimize
    model = Sequential()
    model.add(Bidirectional(LSTM(50, input_shape=(seq_length, 1), kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))))
    model.add(Dense(1))

    with model_training:
        if stock_name != 'Select' :
            fig_optimized, ax_optimized = plt.subplots(figsize=(12, 6))
            ax_optimized.plot(predictions, label='Predicted')
            ax_optimized.legend()
            ax_optimized.set_xlabel('Time (Days)')
            ax_optimized.set_ylabel('Stock Price (Dollars)')
            ax_optimized.set_title('Stock Price Prediction with Optimized Bi-RNN')
            st.pyplot(fig_optimized)
        # Plot the results
        

selected_page = st.sidebar.selectbox(
    'Select a page',
    ['Home', 'Game'])

if selected_page == 'Home':
    home()
elif selected_page == 'Game':
    main.game()
