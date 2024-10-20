# stock_trading_simulation.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Download Stock Data


def download_stock_data(tickers, start_date, end_date):
    """
    Downloads historical stock data for the given tickers
    and date range. Returns a combined DataFrame of all tickers.
    """
    all_data = []
    for ticker in tickers:
        print(f"Downloading data for {ticker}")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df = df[['Adj Close']].copy()
            df.reset_index(inplace=True)
            df['ticker'] = ticker
            df.rename(columns={
                'Adj Close': 'price',
                'Date': 'business_date'
            }, inplace=True)
            all_data.append(df[['ticker', 'business_date', 'price']])
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Step 2: Feature Engineering


def feature_engineering(data):
    """
    Performs feature engineering on the data.
    Adds price returns, moving averages, lagged returns, and RSI.
    """
    # Convert 'business_date' to datetime
    data['business_date'] = pd.to_datetime(data['business_date'])

    # Sort data
    data.sort_values(['ticker', 'business_date'], inplace=True)

    # Calculate daily returns for each stock
    data['price_return'] = data.groupby('ticker')['price'].pct_change()

    # Calculate moving averages for each stock (7-day and 21-day)
    data['ma7'] = data.groupby('ticker')['price'] \
        .transform(lambda x: x.rolling(window=7).mean())
    data['ma21'] = data.groupby('ticker')['price'] \
        .transform(lambda x: x.rolling(window=21).mean())

    # Lag the price returns (shift by 1 to avoid lookahead bias)
    data['lagged_return'] = data.groupby('ticker')['price_return'].shift(1)

    # Add Relative Strength Index (RSI) as a feature
    def compute_RSI(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data['rsi'] = data.groupby('ticker')['price'].transform(compute_RSI)

    # Drop NaN values
    data.dropna(inplace=True)

    return data

# Step 3: Prepare the Features and Labels


def prepare_datasets(data, features, test_size=0.2):
    """
    Splits the data into training and testing datasets.
    Returns X_train, X_test, y_train, y_test, train_data, test_data.
    """
    # Ensure data is sorted
    data.sort_values(['ticker', 'business_date'], inplace=True)

    # Split data by ticker
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        split_point = int(len(ticker_data) * (1 - test_size))
        train_data = pd.concat([train_data, ticker_data.iloc[:split_point]])
        test_data = pd.concat([test_data, ticker_data.iloc[split_point:]])

    # Prepare the training and testing features and labels
    X_train = train_data[features]
    y_train = train_data['price']
    X_test = test_data[features]
    y_test = test_data['price']

    return X_train, X_test, y_train, y_test, train_data, test_data

# Step 4: Train Models


def train_models(X_train, y_train):
    """
    Trains Random Forest, XGBoost, and Linear Regression models.
    Returns the trained models.
    """
    # Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # XGBoost Model
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    return rf_model, xgb_model, lr_model

# Step 5: Evaluate Models


def evaluate_models(models, X_test, y_test, test_data):
    """
    Evaluates the models and prints their RMSE.
    Returns a DataFrame with actual and predicted prices.
    """
    rf_model, xgb_model, lr_model = models

    # Predictions
    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    # Evaluate Models
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

    print(f"\nModel Performance:")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    print(f"XGBoost RMSE: {xgb_rmse:.2f}")
    print(f"Linear Regression RMSE: {lr_rmse:.2f}")

    # Determine the best model
    rmse_values = {'rf': rf_rmse, 'xgb': xgb_rmse, 'lr': lr_rmse}
    best_model_name = min(rmse_values, key=rmse_values.get)
    print(
        f"\nBest Model: {best_model_name.upper()} with RMSE: {rmse_values[best_model_name]:.2f}")

    # Create a DataFrame with actual and predicted prices and dates
    results = X_test.copy()
    results.reset_index(drop=True, inplace=True)
    test_data = test_data.reset_index(drop=True)
    results['ticker'] = test_data['ticker']
    results['business_date'] = test_data['business_date']
    results['actual_price'] = y_test.values
    results['rf_predicted_price'] = rf_predictions
    results['xgb_predicted_price'] = xgb_predictions
    results['lr_predicted_price'] = lr_predictions
    results['best_model'] = best_model_name

    return results, best_model_name

# Step 6: Simulate Trades


def simulate_trades(model_predictions, best_model_name, initial_capital=10000):
    """
    Simulates trades based on model predictions.
    Returns the trading history and final portfolio value.
    """
    trading_history = []
    portfolio = {'cash': initial_capital,
                 'holdings': {}, 'total_value': initial_capital}

    # Ensure 'business_date' is in datetime format
    model_predictions['business_date'] = pd.to_datetime(
        model_predictions['business_date'])

    # Select the predictions from the best model
    predicted_price_column = f"{best_model_name}_predicted_price"

    # Sort predictions by date
    model_predictions.sort_values('business_date', inplace=True)

    for index, row in model_predictions.iterrows():
        date = row['business_date']
        actual_price = row['actual_price']
        predicted_price = row[predicted_price_column]
        ticker = row['ticker']

        # Expected return
        expected_return = (predicted_price - actual_price) / actual_price

        # Buy/Sell thresholds
        buy_threshold = 0.01  # Buy if expected return > 1%
        sell_threshold = -0.005  # Sell if expected return < -0.5%

        # Calculate max amount to spend (20% of available cash)
        max_purchase_amount = portfolio['cash'] * 0.20

        # Determine the number of shares to buy
        shares_to_buy = max_purchase_amount // actual_price

        # Buy signal
        if expected_return > buy_threshold and shares_to_buy > 0:
            cost = shares_to_buy * actual_price
            portfolio['cash'] -= cost
            portfolio['holdings'][ticker] = portfolio['holdings'].get(
                ticker, 0) + shares_to_buy
            trading_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'buy',
                'price': actual_price,
                'shares': shares_to_buy
            })
        # Sell signal
        elif expected_return < sell_threshold and portfolio['holdings'].get(ticker, 0) > 0:
            shares_to_sell = portfolio['holdings'][ticker]
            revenue = shares_to_sell * actual_price
            portfolio['cash'] += revenue
            trading_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'sell',
                'price': actual_price,
                'shares': shares_to_sell
            })
            portfolio['holdings'][ticker] = 0

    # Final portfolio value
    total_holdings_value = 0
    for ticker, shares in portfolio['holdings'].items():
        if shares > 0:
            # Get the latest price
            latest_price = model_predictions[model_predictions['ticker']
                                             == ticker]['actual_price'].iloc[-1]
            total_holdings_value += shares * latest_price
    final_value = portfolio['cash'] + total_holdings_value

    return trading_history, final_value

# Main Function


def main():
    # Define parameters
    tickers = ['NVDA', 'BTC-USD']  # You can add more tickers
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365*8)
                  ).strftime('%Y-%m-%d')
    features = ['price_return', 'ma7', 'ma21', 'lagged_return', 'rsi']

    # Step 1: Download Stock Data
    data = download_stock_data(tickers, start_date, end_date)

    # Step 2: Feature Engineering
    data = feature_engineering(data)

    # Step 3: Prepare the Features and Labels
    X_train, X_test, y_train, y_test, train_data, test_data = \
        prepare_datasets(data, features)

    # Step 4: Train Models
    models = train_models(X_train, y_train)

    # Step 5: Evaluate Models
    results, best_model_name = evaluate_models(
        models, X_test, y_test, test_data)

    # Step 6: Simulate Trades
    model_predictions = results.copy()
    trading_history, final_portfolio_value = simulate_trades(
        model_predictions, best_model_name, initial_capital=10000)

    # Display Trading History
    print("\nTrading History:")
    for trade in trading_history:
        trade_date = trade['date'].strftime('%Y-%m-%d')
        print(f"Date: {trade_date}, Ticker: {trade['ticker']}, Action: {trade['action']}, "
              f"Price: {trade['price']:.2f}, Shares: {trade['shares']}")

    print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")

    # Save Results to CSV for Further Analysis
    results_csv_path = os.path.join(os.getcwd(), 'predicted_stock_prices.csv')
    results.to_csv(results_csv_path, index=False)
    print(f"\nPredicted stock prices saved to '{results_csv_path}'")


if __name__ == "__main__":
    main()
