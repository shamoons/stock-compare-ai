# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title("Stock Prediction Model Comparison")

# Sidebar inputs
st.sidebar.header("User Input Features")
ml_platform = st.sidebar.selectbox(
    "Select ML Platform",
    ['Azure ML', 'AWS SageMaker']
)
period = st.sidebar.number_input(
    "Forecast Period (days)", min_value=1, max_value=30, value=5)

# Load data from CSV


def load_data_from_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data_from_csv('fake_stock_data.csv')
data_load_state.text('Loading data...done!')

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name="Closing Price"))
    fig.layout.update(title_text='Historical Closing Prices',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Simulate Azure ML predictions


def azure_ml_predict(data, period):
    last_price = data['Close'].iloc[-1]
    dates = pd.date_range(data['Date'].iloc[-1],
                          periods=period+1, freq='B')[1:]
    predictions = last_price + np.cumsum(np.random.randn(len(dates)) + 0.5)
    azure_predictions = pd.DataFrame(
        {'Date': dates, 'Predicted_Price': predictions})
    return azure_predictions

# Simulate AWS SageMaker predictions


def aws_sagemaker_predict(data, period):
    last_price = data['Close'].iloc[-1]
    dates = pd.date_range(data['Date'].iloc[-1],
                          periods=period+1, freq='B')[1:]
    predictions = last_price + np.cumsum(np.random.randn(len(dates)) - 0.5)
    aws_predictions = pd.DataFrame(
        {'Date': dates, 'Predicted_Price': predictions})
    return aws_predictions


# Generate predictions based on selected platform
if ml_platform == 'Azure ML':
    with st.spinner('Generating predictions with Azure ML...'):
        forecast = azure_ml_predict(data, period)
elif ml_platform == 'AWS SageMaker':
    with st.spinner('Generating predictions with AWS SageMaker...'):
        forecast = aws_sagemaker_predict(data, period)

# Plot the predictions
st.subheader(f'Predictions using {ml_platform}')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Price'))
fig.add_trace(go.Scatter(
    x=forecast['Date'], y=forecast['Predicted_Price'], name='Predicted Price'))
fig.layout.update(title_text='Stock Price Prediction',
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Performance metrics (demonstration purposes)
st.subheader('Performance Metrics (Demo)')
st.write("Since we're predicting future prices, we can't calculate metrics yet")
st.write("Here's how we'd calculate them if we had actual future prices.")
