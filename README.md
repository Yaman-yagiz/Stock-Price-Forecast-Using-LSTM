# Stock Price Prediction Project

## Overview

This project aims to predict stock prices using historical data and machine learning techniques, specifically Long Short-Term Memory (LSTM) networks. The primary goal is to provide a predictive model that can forecast the next day's closing price of a given stock based on its historical performance.

## Project Structure

The project consists of three main Python files:

1. **`run.py`**: The entry point of the application that initializes the configuration, downloads data, and runs the prediction pipeline.
2. **`stock_price_predicton.py`**: Contains the `StockPricePredictor` class, which implements the core functionality for downloading data, preprocessing, feature engineering, model building, training, and evaluation.
3. **`stock_price_pydantic.py`**: Defines the configuration model using Pydantic, ensuring that the input parameters are validated and correctly formatted.

## Techniques Used

### 1. Data Downloading
- **Yahoo Finance API**: The project uses the `yfinance` library to download historical stock data. This allows for easy access to stock prices over a specified date range.

### 2. Data Preprocessing
- **Handling Missing Values**: The data is forward-filled to handle any missing values, ensuring that the model has a complete dataset for training.
- **Normalization**: The features and target values are normalized using `MinMaxScaler` to scale the data between 0 and 1. This is crucial for LSTM models, as they are sensitive to the scale of input data.

### 3. Feature Engineering
- **Technical Indicators**: The project calculates various technical indicators such as:
  - **Relative Strength Index (RSI)**: Measures the speed and change of price movements to identify overbought or oversold conditions.
  - **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages.
  - **Bollinger Bands**: Measures volatility and identifies overbought or oversold conditions.
- **Temporal Features**: Additional features such as year, month, day, and day of the week are extracted to provide the model with more context about the data.

### 4. Model Building
- **LSTM Networks**: The project employs LSTM networks, which are well-suited for time series forecasting due to their ability to learn from sequences of data. The model architecture includes:
  - Input layer
  - LSTM layers with dropout for regularization
  - Dense layers for output

### 5. Model Training and Evaluation
- The model is trained using the training dataset, and its performance is evaluated on a separate test dataset. Metrics such as Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared (R²) are calculated to assess the model's accuracy.

### 6. Visualization
- The training history and predictions are visualized using Matplotlib, allowing for a better understanding of the model's performance over time.

## Configuration

The configuration for the stock price predictor is defined in the `StockPricePredictorConfig` class using Pydantic. The following parameters can be set:

- `ticker`: The stock ticker symbol (default: "AAPL").
- `start_date`: The start date for historical data (default: "2015-01-01").
- `end_date`: The end date for historical data (optional).
- `seq_length`: The sequence length for the LSTM model (default: 30).
- `epochs`: The number of epochs for training (default: 20).
- `batch_size`: The batch size for training (default: 32).

## Usage

To run the project, execute the `run.py` file. This will initiate the entire stock price prediction pipeline, including data downloading, preprocessing, model training, and evaluation.


## Conclusion

This project demonstrates the application of machine learning techniques to predict stock prices using historical data. By leveraging LSTM networks and various feature engineering techniques, the model aims to provide accurate predictions that can assist investors and analysts in making informed decisions.
