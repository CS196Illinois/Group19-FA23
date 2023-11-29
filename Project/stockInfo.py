import streamlit as st
import yfinance as yf

def fetch_stock_data(stock_ticker):
    stock = yf.Ticker(stock_ticker)

    # Fetching company fundamentals
    fundamentals = stock.info
    revenue = fundamentals.get('revenue', 'N/A')
    profit_margin = fundamentals.get('profitMargins', 'N/A')

    # Fetching historical stock performance
    historical_data = stock.history(period="1y")  # Adjust the period as needed

    # Formatting the display information
    display_info = f"""
    **Stock**: {stock_ticker}
    **Revenue**: {revenue}
    **Profit Margin**: {profit_margin}
    \n**Historical Data (Last 5)**:\n{historical_data.tail()}
    """
    return display_info

# Streamlit UI layout
st.title('Stock Information')

# Buttons for each stock
if st.button('Apple'):
    st.text(fetch_stock_data("AAPL"))

if st.button('Tesla'):
    st.text(fetch_stock_data("TSLA"))

if st.button('Nike'):
    st.text(fetch_stock_data("NKE"))
