import tkinter as tk
import yfinance as yf

def fetch_stock_data(stock_ticker):
    stock = yf.Ticker(stock_ticker)

    # Fetching company fundamentals
    fundamentals = stock.info
    revenue = fundamentals.get('revenue', 'N/A')
    profit_margin = fundamentals.get('profitMargins', 'N/A')

    # Fetching historical stock performance
    historical_data = stock.history(period="1y")  # You can adjust the period

    # Fetching earnings reports
    earnings = stock.earnings

    # Displaying the information
    display_info = f"Stock: {stock_ticker}\nRevenue: {revenue}\nProfit Margin: {profit_margin}\n\nHistorical Data (Last 5):\n{historical_data.tail()}\n\nEarnings:\n{earnings}"
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, display_info)


# Set up the main window
root = tk.Tk()
root.title("Stock Information")

# Text box for displaying stock info
text_output = tk.Text(root, height=20, width=80)
text_output.pack()

# Buttons for each stock
apple_button = tk.Button(root, text="Apple", command=lambda: fetch_stock_data("AAPL"))
apple_button.pack()

tesla_button = tk.Button(root, text="Tesla", command=lambda: fetch_stock_data("TSLA"))
tesla_button.pack()

nike_button = tk.Button(root, text="Nike", command=lambda: fetch_stock_data("NKE"))
nike_button.pack()

root.mainloop()
