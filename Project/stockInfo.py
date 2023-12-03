import streamlit as st
import yfinance as yf

def show_stock_info():
    def fetch_stock_data(stock_ticker):
        stock = yf.Ticker(stock_ticker)
        data = {}

        # Fetch company fundamentals and other financials
        fundamentals = stock.info
        data['profit_margin'] = fundamentals.get('profitMargins', 'N/A')

        try:
            data['cash_flow_statement'] = stock.cashflow.iloc[:, 0]
        except Exception:   
            data['cash_flow_statement'] = "Data not available"

        try:
            income_statement = stock.financials
            data['revenue'] = income_statement.loc['Total Revenue'][0]
            data['full_income_statement'] = income_statement.iloc[:, 0]
        except Exception:
            data['revenue'] = "Data not available"
            data['full_income_statement'] = "Data not available"

        # Fetching historical stock performance
        historical_data = stock.history(period="1y")
        # Resampling the data to get the last trading day of each month
        monthly_data = historical_data['Close'].resample('M').last()
        # Selecting the last 12 months
        data['historical_data'] = monthly_data.tail(12)

        return data

    # Streamlit UI layout
    st.title('Stock Information')

    # Handling button clicks and data display
    def handle_stock_button(stock_ticker):
        data = fetch_stock_data(stock_ticker)
        st.write(f"**Revenue (Most Recent Year)**: {data['revenue']}")
        st.write(data['full_income_statement'])
        st.write(f"**Cash Flow (Most Recent Year):**")
        st.write(data['cash_flow_statement'])
        st.write("**Historical Data (Last 5):**")
        st.write(data['historical_data'].tail())

    if st.button('Apple'):
        handle_stock_button("AAPL")

    if st.button('Tesla'):
        handle_stock_button("TSLA")

    if st.button('Nike'):
        handle_stock_button("NKE")
