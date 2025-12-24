import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Attention, Dense, LSTM, Input, Multiply, Layer, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
# Function to download and prepare data
def prepare_data(stock_symbol, start_date, end_date, sequence_length):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    
    dataset = df[['Open']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, 0])
    
    X, y = np.array(X), np.array(y).reshape(-1, 1)
    
    print("Dataset Shape:", dataset.shape)
    print("Scaled Data Sample:", scaled_data[:5])
    print("X Shape:", X.shape)
    print("y Shape:", y.shape)
    
    return X, y, scaler, df







# Function to build and train the model
def build_and_train_model(X_train, y_train, epochs=500, batch_size=32, learning_rate=0.001):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(16, return_sequences=True, kernel_regularizer=l2(0.03)),
        Dropout(0.2),
        LSTM(16, kernel_regularizer=l2(0.03)),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(0.03)),
        Dropout(0.3),
        Dense(1)  # This should output a single value for each sample
    ])

    optimizer = Adam(learning_rate=learning_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Remove sample_weight as it's causing issues
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.3, verbose=1, callbacks=[early_stopping])
    
    return model, history


def create_sample_weights(sequence_length, num_samples):
    weights = np.linspace(0.5, 1, sequence_length)
    return np.tile(weights, (num_samples, 1))



# Function to make forecasts
def forecast(model, data, scaler, sequence_length, forecast_steps):
    last_sequence = data[-sequence_length:].reshape(1, sequence_length, 1)  # Reshape to (1, sequence_length, 1)
    forecasted_values = []
    
    for _ in range(forecast_steps):
        # Predict the next value
        prediction = model.predict(last_sequence)
        forecasted_values.append(prediction[0, 0])
        
        # Update the last_sequence with the new prediction
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = prediction[0, 0]
    
    # Convert forecasted values to array and inverse transform to original scale
    forecasted_values = np.array(forecasted_values).reshape(-1, 1)
    forecasted_values = scaler.inverse_transform(forecasted_values)
    
    print("Forecasted Values:", forecasted_values.flatten())
    
    return forecasted_values.flatten()


def create_sample_weights(sequence_length, num_samples):
    weights = np.linspace(0.5, 1, sequence_length)
    return np.tile(weights, (num_samples, 1)).flatten()

if __name__ == "__main__":
    stock_symbol = 'AAPL'
    start_date = '2025-08-01'
    end_date = datetime.today().strftime("%Y-%m-%d")
    sequence_length = 30
    forecast_steps = 30
    
    X, y, scaler, df = prepare_data(stock_symbol, start_date, end_date, sequence_length)
    
    training_data_len = int(len(X) * 0.8)
    X_train, y_train = X[:training_data_len], y[:training_data_len]
    X_test, y_test = X[training_data_len:], y[training_data_len:]
    
    model, history = build_and_train_model(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show() 
    
    forecasted_values = forecast(model, df[['Open']].values, scaler, sequence_length, forecast_steps)
    
    # Create a date range for the forecasted values
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]
    
    # Ensure that the historical data being plotted is correctly scaled
    historical_data_scaled = scaler.transform(df[['Open']])
    historical_data_unscaled = scaler.inverse_transform(historical_data_scaled)
    
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(df.index, historical_data_unscaled, label="Historical Data", color="blue")
    
    # Plot forecasted data
    plt.plot(forecast_dates, forecasted_values, label="Forecasted Values", color="red")
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.title(f'{stock_symbol} Stock Price Forecast')
    plt.grid(True)
    plt.show()
