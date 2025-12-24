import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Prepare data for modeling
def prepare_data(data):
    data = data[['Close']].copy()  # Use .copy() to avoid SettingWithCopyWarning
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    
    X = data[['Close']].values
    y = data['Target'].values
    
    return X, y, data.index


def main():
    ticker = 'NVDA'  # Chosen Stock Ticker
    #Takes Stock data from start_date to end_date
    start_date = '2020-01-01'
    end_date = '2024-07-29'
    
    data = fetch_stock_data(ticker, start_date, end_date)
    
    # Prepare data for model training
    X, y, index = prepare_data(data)
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict prices for the entire dataset
    predictions = model.predict(X)
    
    # Create a DataFrame to show actual vs predicted prices
    results = pd.DataFrame({'Close': X.flatten(), 'Predicted': predictions}, index=index)
    
    # Align the length of predictions with the DataFrame index
    predictions_with_nan = np.concatenate([predictions, [np.nan]])
   

    # Predict the next day’s price based on the latest available data
    latest_price = data[['Close']].iloc[-1].values.reshape(1, -1)
    next_day_prediction = model.predict(latest_price)
    print(f"Predicted next day's price: {next_day_prediction[0]:.2f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(results.index, results['Close'], label='Actual Prices')
    plt.plot(results.index, results['Predicted'], label='Predicted Prices', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()