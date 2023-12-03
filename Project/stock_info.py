import streamlit as st
import yfinance as yf
import functions as f

def plot(stock_ticker: str):
    time_frame =  st.select_slider('Select number of days', options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], value="1y")
    if time_frame:
        st.plotly_chart(f.get_normal_chart(stock_ticker, time_frame))

def show_stock_info() -> None:
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
    def fetch_stock_data(stock_ticker: str):
        stock = yf.Ticker(stock_ticker)
        data = {}

        # Fetch company fundamentals and other financials

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
        st.write("**Income Statement (Most Recent Year):**")
        st.write(data['full_income_statement'])
        st.write("**Cash Flow (Most Recent Year):**")
        st.write(data['cash_flow_statement'])
        st.write("**Historical Data (Last 5):**")
        st.write(data['historical_data'].tail())

    ticker: str = st.text_input('Enter Stock Ticker Symbol')
    if st.button('Search'):
        handle_stock_button(ticker)

    plot(ticker)