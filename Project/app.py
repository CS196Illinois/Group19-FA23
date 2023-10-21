from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows cross-origin requests, which is important for connecting with the React frontend

# Dummy model for stock price prediction
def get_stock_price_prediction(ticker):
    # For demonstration purposes, we'll just return a dummy value.
    # Replace this with your actual model.
    return f"The predicted price for {ticker} is $100."

@app.route('/')
def home():
    return "Welcome to the Stock Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data['ticker']
    prediction = get_stock_price_prediction(ticker)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
