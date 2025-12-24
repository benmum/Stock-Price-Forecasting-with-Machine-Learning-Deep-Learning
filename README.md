# 📈 Stock Price Forecasting with Machine Learning & Deep Learning

This repository demonstrates two different approaches to stock price prediction using historical market data:

1. **Linear Regression** – A baseline machine learning model  
2. **LSTM Neural Network** – A deep learning time-series forecasting model

The goal is to compare a simple statistical model with a more advanced sequence-based neural network.

---

## 🚀 Models Included

### 1️⃣ Linear Regression Stock Predictor
- Uses previous day's closing price to predict the next day's price
- Built with:
  - scikit-learn
  - yfinance
  - pandas / numpy
- Visualizes actual vs predicted prices

📂 **Location:** `linear_regression_model/stock.py`

---

### 2️⃣ LSTM Time-Series Forecasting Model
- Uses historical price sequences to predict future stock prices
- Built with:
  - TensorFlow / Keras
  - LSTM layers with dropout & regularization
  - MinMax scaling
- Forecasts multiple future trading days

📂 **Location:** `lstm_forecasting_model/tsf.py`

---

## 📊 Data Source
- Market data fetched from **Yahoo Finance** using the `yfinance` API

---

## 🛠 Installation

Clone the repository:
```bash
git clone https://github.com/benmum/Stock-Price-Forecasting-with-Machine-Learning-Deep-Learning.git
cd Stock-Price-Forecasting-with-Machine-Learning-Deep-Learning
