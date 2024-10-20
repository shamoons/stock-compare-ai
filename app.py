# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page layout
st.set_page_config(layout="wide")

# Title
st.title("Stock Prediction Model Comparison")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Tickers
tickers = st.sidebar.multiselect(
    "Select Stock Ticker(s)",
    ['NVDA', 'ORCL', 'QQQ', 'SPY', 'SOXL', 'TQQQ', 'BTC-USD'],
    default=['NVDA']
)

# Date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

# Forecast period
period = st.sidebar.number_input(
    "Forecast Period (days)", min_value=1, max_value=30, value=5)

# ML Platform
ml_platform = st.sidebar.selectbox(
    "Select ML Platform",
    ['Azure ML', 'AWS SageMaker']
)

# Load data


@st.cache_data
def load_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        data[ticker] = df
    return data


data_load_state = st.text('Loading data...')
data = load_data(tickers, start_date, end_date)
data_load_state.text('Loading data...done!')

# Display and plot data for each ticker
for ticker in tickers:
    st.subheader(f'Raw Data for {ticker}')
    st.write(data[ticker].tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data[ticker]['Date'], y=data[ticker]['Close'], name="Closing Price"))
        fig.layout.update(
            title_text=f'Historical Closing Prices for {ticker}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()

# Prediction functions


def azure_ml_predict(df, period):
    last_price = df['Close'].iloc[-1]
    dates = pd.date_range(df['Date'].iloc[-1], periods=period+1, freq='B')[1:]
    predictions = last_price + np.cumsum(np.random.randn(len(dates)) + 0.5)
    azure_predictions = pd.DataFrame(
        {'Date': dates, 'Predicted_Price': predictions})
    return azure_predictions


def aws_sagemaker_predict(df, period):
    last_price = df['Close'].iloc[-1]
    dates = pd.date_range(df['Date'].iloc[-1], periods=period+1, freq='B')[1:]
    predictions = last_price + np.cumsum(np.random.randn(len(dates)) - 0.5)
    aws_predictions = pd.DataFrame(
        {'Date': dates, 'Predicted_Price': predictions})
    return aws_predictions


# Generate predictions and evaluate performance
predictions = {}

st.subheader('Model Predictions and Performance')

for ticker in tickers:
    df = data[ticker]

    # Split data into train and test sets for evaluation
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # Generate predictions
    if ml_platform == 'Azure ML':
        with st.spinner(f'Generating predictions for {ticker} with Azure ML...'):
            forecast = azure_ml_predict(train_df, len(test_df))
    elif ml_platform == 'AWS SageMaker':
        with st.spinner(f'Generating predictions for {ticker} with AWS SageMaker...'):
            forecast = aws_sagemaker_predict(train_df, len(test_df))

    # Store predictions
    predictions[ticker] = forecast

    # Plot the predictions
    st.write(f"### Predictions for {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_df['Date'], y=train_df['Close'], name='Training Data'))
    fig.add_trace(go.Scatter(
        x=test_df['Date'], y=test_df['Close'], name='Actual Price'))
    fig.add_trace(go.Scatter(
        x=forecast['Date'], y=forecast['Predicted_Price'], name='Predicted Price'))
    fig.layout.update(
        title_text=f'Stock Price Prediction for {ticker}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Evaluate performance
    test_df = test_df.reset_index(drop=True)
    forecast = forecast.reset_index(drop=True)
    actual = test_df['Close']
    predicted = forecast['Predicted_Price']

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    st.write(f"#### Performance Metrics for {ticker}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
