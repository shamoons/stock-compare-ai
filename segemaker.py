import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Download Stock Data

# List of tickers
tickers = ['NVDA', 'ORCL', 'QQQ', 'SPY', 'SOXL', 'TQQQ', 'BTC-USD']

# Define the time period (8 years back from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 8)).strftime('%Y-%m-%d')

# Function to download historical data
def download_data(ticker, start_date, end_date):
    try:
        # Download data for each ticker
        df = yf.download(ticker, start=start_date, end=end_date)[['Adj Close']].copy()
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        df.rename(columns={'Adj Close': 'price', 'Date': 'business_date'}, inplace=True)
        return df[['ticker', 'business_date', 'price']]
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

# Download data for all tickers and save to a CSV file
all_data = []
for ticker in tickers:
    print(f"Downloading data for {ticker}")
    data = download_data(ticker, start_date, end_date)
    if data is not None:
        all_data.append(data)

# Concatenate all data into a single DataFrame
combined_data = pd.concat(all_data)

# Save to a CSV file in the current directory
csv_file_path = os.path.join(os.getcwd(), 'eod_data_ticker_businessdate_price.csv')
combined_data.to_csv(csv_file_path, index=False)

print(f"EOD data for the last 8 years has been saved to '{csv_file_path}'")

# Step 2: Feature Engineering

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Convert 'business_date' to datetime
data['business_date'] = pd.to_datetime(data['business_date'])

# Calculate daily returns for each stock
data['price_return'] = data.groupby('ticker')['price'].pct_change()

# Calculate moving averages for each stock (7-day and 21-day)
data['ma7'] = data.groupby('ticker')['price'].transform(lambda x: x.rolling(window=7).mean())
data['ma21'] = data.groupby('ticker')['price'].transform(lambda x: x.rolling(window=21).mean())

# Lag the price returns (shift by 1 to avoid lookahead bias)
data['lagged_return'] = data.groupby('ticker')['price_return'].shift(1)

# Drop NaN values created by pct_change, rolling averages, and lagging
data.dropna(inplace=True)

# Split the data into training and testing datasets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training and testing datasets to CSV files in the current directory
train_csv_path = os.path.join(os.getcwd(), 'train_data.csv')
test_csv_path = os.path.join(os.getcwd(), 'test_data.csv')

train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Training data saved to '{train_csv_path}'")
print(f"Testing data saved to '{test_csv_path}'")

# Step 3: Prepare the Features and Labels

# Features to be used in the models
features = ['price_return', 'ma7', 'ma21', 'lagged_return']

# Prepare the training and testing features and labels
X_train = train_data[features].values
y_train = train_data['price'].values
X_test = test_data[features].values
y_test = test_data['price'].values

# Step 4: Train Three Different Models and Evaluate Them

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Step 5: Compare Actual vs Predicted Stock Prices for Each Model

print("\nModel Performance Comparison:")

# Evaluate Random Forest Model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
print(f"\nRandom Forest RMSE: {rf_rmse}")

# Evaluate XGBoost Model
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
print(f"XGBoost RMSE: {xgb_rmse}")

# Evaluate Linear Regression Model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
print(f"Linear Regression RMSE: {lr_rmse}")

# Step 6: Add the Predictions to the Test Dataset for Comparison

test_data['rf_predicted_price'] = rf_predictions
test_data['xgb_predicted_price'] = xgb_predictions
test_data['lr_predicted_price'] = lr_predictions

# Compare actual vs predicted stock prices for each ticker
for ticker in test_data['ticker'].unique():
    ticker_data = test_data[test_data['ticker'] == ticker]
    print(f"\nComparing predictions for {ticker}:")
    print(ticker_data[['business_date', 'price', 'rf_predicted_price', 'xgb_predicted_price', 'lr_predicted_price']])

# Step 7: Save Results to CSV for Further Analysis
results_csv_path = os.path.join(os.getcwd(), 'predicted_stock_prices.csv')
test_data.to_csv(results_csv_path, index=False)
print(f"\nPredicted stock prices have been saved to '{results_csv_path}'")
