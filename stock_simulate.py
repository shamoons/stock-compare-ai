# stock_trading_simulation.py

import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import os

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
    Adds price returns, moving averages, and lagged returns.
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

    # Drop NaN values
    data.dropna(inplace=True)

    return data

# Step 3: Prepare the Features and Labels


def prepare_datasets(data, features, split_date):
    """
    Splits the data into training and testing datasets for each ticker based on a split date.
    Returns a dictionary of datasets per ticker.
    """
    datasets = {}
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        # Ensure data is sorted
        ticker_data.sort_values('business_date', inplace=True)
        # Split data into training and testing sets based on date
        split_date = pd.to_datetime(split_date)
        train_data = ticker_data[ticker_data['business_date'] < split_date]
        test_data = ticker_data[(ticker_data['business_date'] >= split_date) &
                                (ticker_data['business_date'] <= pd.to_datetime('2020-12-31'))]
        # Prepare features and labels
        X_train = train_data[features]
        y_train = train_data['price']
        X_test = test_data[features]
        y_test = test_data['price']
        datasets[ticker] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_data': train_data,
            'test_data': test_data
        }
    return datasets

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


def evaluate_models(models, X_test, y_test, test_data, ticker):
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

    print(f"\nModel Performance for {ticker}:")
    print(f"Random Forest RMSE: {rf_rmse:.2f}")
    print(f"XGBoost RMSE: {xgb_rmse:.2f}")
    print(f"Linear Regression RMSE: {lr_rmse:.2f}")

    # Determine the best model
    rmse_values = {'rf': rf_rmse, 'xgb': xgb_rmse, 'lr': lr_rmse}
    best_model_name = min(rmse_values, key=rmse_values.get)
    print(
        f"Best Model for {ticker}: {best_model_name.upper()} with RMSE: {rmse_values[best_model_name]:.2f}")

    # Create a DataFrame with actual and predicted prices and dates
    results = X_test.copy()
    results.reset_index(drop=True, inplace=True)
    test_data = test_data.reset_index(drop=True)
    results['ticker'] = ticker
    results['business_date'] = test_data['business_date']
    results['actual_price'] = y_test.values
    results['rf_predicted_price'] = rf_predictions
    results['xgb_predicted_price'] = xgb_predictions
    results['lr_predicted_price'] = lr_predictions
    results['best_model'] = best_model_name

    return results, best_model_name

# Step 6: Simulate Trades Across All Tickers


def simulate_trades_all_tickers(all_predictions, initial_capital=10000):
    """
    Simulates trades across all tickers using combined model predictions.
    Returns the trading history, transaction history, final portfolio value, and portfolio values over time.
    """
    trading_history = []
    transaction_history = []
    portfolio = {'cash': initial_capital, 'holdings': {}, 'cost_basis': {}}

    portfolio_values = []
    last_portfolio_value = initial_capital

    # Iterate over all predictions in chronological order
    for index, row in all_predictions.iterrows():
        date = row['business_date']
        actual_price = row['actual_price']
        predicted_price = row[f"{row['best_model']}_predicted_price"]
        ticker = row['ticker']
        expected_return = row['expected_return']

        # Buy/Sell thresholds
        buy_threshold = 0.01  # Buy if expected return > 1%
        sell_threshold = -0.005  # Sell if expected return < -0.5%

        # Check for buy signal
        if expected_return > buy_threshold:
            # Determine the amount to invest (use all available cash)
            shares_to_buy = portfolio['cash'] // actual_price
            if shares_to_buy > 0:
                cost = shares_to_buy * actual_price
                portfolio['cash'] -= cost
                portfolio['holdings'][ticker] = portfolio['holdings'].get(
                    ticker, 0) + shares_to_buy
                # Update cost basis
                previous_cost = portfolio['cost_basis'].get(ticker, 0)
                previous_shares = portfolio['holdings'].get(
                    ticker, 0) - shares_to_buy
                total_cost = previous_cost + cost
                total_shares = previous_shares + shares_to_buy
                portfolio['cost_basis'][ticker] = total_cost

                # Record transaction
                transaction_history.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'buy',
                    'Price': actual_price,
                    'Shares': shares_to_buy,
                    'Transaction Value': cost,
                    'Portfolio Value': None,  # To be updated later
                    'Profit/Loss': 0.0  # No profit/loss on buy
                })
                trading_history.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'buy',
                    'price': actual_price,
                    'shares': shares_to_buy
                })
        # Check for sell signal
        elif expected_return < sell_threshold and portfolio['holdings'].get(ticker, 0) > 0:
            shares_to_sell = portfolio['holdings'][ticker]
            revenue = shares_to_sell * actual_price
            portfolio['cash'] += revenue
            portfolio['holdings'][ticker] = 0

            # Calculate profit/loss
            total_cost = portfolio['cost_basis'].get(ticker, 0)
            average_cost_basis = total_cost / shares_to_sell
            profit_loss = (actual_price - average_cost_basis) * shares_to_sell

            # Reset cost basis
            portfolio['cost_basis'][ticker] = 0

            # Record transaction
            transaction_history.append({
                'Date': date,
                'Ticker': ticker,
                'Action': 'sell',
                'Price': actual_price,
                'Shares': shares_to_sell,
                'Transaction Value': revenue,
                'Portfolio Value': None,  # To be updated later
                'Profit/Loss': profit_loss
            })
            trading_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'sell',
                'price': actual_price,
                'shares': shares_to_sell
            })

        # Calculate total portfolio value
        total_holdings_value = 0
        for tkr in portfolio['holdings']:
            shares = portfolio['holdings'][tkr]
            if shares > 0:
                # Get the latest available price for the ticker up to the current date
                try:
                    latest_price = all_predictions[(all_predictions['ticker'] == tkr) & (
                        all_predictions['business_date'] <= date)]['actual_price'].iloc[-1]
                except IndexError:
                    # If no price is available, skip
                    latest_price = 0
                total_holdings_value += shares * latest_price

        portfolio_value = portfolio['cash'] + total_holdings_value

        # Record portfolio value
        if not portfolio_values or date != portfolio_values[-1]['date']:
            portfolio_values.append(
                {'date': date, 'portfolio_value': portfolio_value})
            last_portfolio_value = portfolio_value

    # Final portfolio value
    final_value = portfolio_value

    # Update Portfolio Value in transaction history
    for i, transaction in enumerate(transaction_history):
        date = transaction['Date']
        # Find the corresponding portfolio value
        matching_values = [pv['portfolio_value']
                           for pv in portfolio_values if pv['date'] == date]
        if matching_values:
            transaction['Portfolio Value'] = matching_values[0]
        else:
            transaction['Portfolio Value'] = last_portfolio_value

    # Create DataFrame for portfolio values
    portfolio_values_df = pd.DataFrame(portfolio_values)
    portfolio_values_df.set_index('date', inplace=True)
    portfolio_values_df = portfolio_values_df[~portfolio_values_df.index.duplicated(
        keep='last')]

    return trading_history, transaction_history, final_value, portfolio_values_df

# Main Function


def main():
    # Define parameters
    tickers = ['NVDA', 'BTC-USD', 'QQQ', 'ORCL', 'SOXL',
               'TSLA', 'GME', 'WMT', 'BRK.A']  # Add more tickers as needed
    start_date = '2015-01-01'  # Start date for data download
    split_date = '2020-01-01'  # Training data before this date
    end_date = '2020-12-31'    # End date for simulation
    features = ['price_return', 'ma7', 'ma21',
                'lagged_return']  # Removed 'rsi'
    initial_capital = 10000  # Starting capital

    # Step 1: Download Stock Data
    data = download_stock_data(tickers, start_date, end_date)

    # Step 2: Feature Engineering
    data = feature_engineering(data)

    # Step 3: Prepare the Features and Labels
    datasets = prepare_datasets(data, features, split_date)

    # Initialize variables
    all_predictions = pd.DataFrame()

    # Step 4-5: Process each ticker separately to get predictions
    for ticker in tickers:
        dataset = datasets.get(ticker)
        if dataset is None:
            print(f"No data available for {ticker}. Skipping...")
            continue

        X_train = dataset['X_train']
        X_test = dataset['X_test']
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        train_data = dataset['train_data']
        test_data = dataset['test_data']

        # Check if there is enough training data
        if len(X_train) < 30:
            print(f"Not enough training data for {ticker}. Skipping...")
            continue

        # Train Models
        models = train_models(X_train, y_train)

        # Evaluate Models
        results, best_model_name = evaluate_models(
            models, X_test, y_test, test_data, ticker)

        # Calculate expected returns
        predicted_price_column = f"{best_model_name}_predicted_price"
        results['expected_return'] = (
            results[predicted_price_column] - results['actual_price']) / results['actual_price']
        results['best_model'] = best_model_name
        all_predictions = pd.concat(
            [all_predictions, results], ignore_index=True)

        # Save Results to CSV for Further Analysis
        results_csv_path = os.path.join(
            os.getcwd(), f'predicted_stock_prices_{ticker}.csv')
        results.to_csv(results_csv_path, index=False)
        print(
            f"\nPredicted stock prices for {ticker} saved to '{results_csv_path}'\n")

    # Sort all predictions chronologically
    all_predictions.sort_values('business_date', inplace=True)

    # Simulate Trades across all tickers
    trading_history, transaction_history, final_portfolio_value, portfolio_values_df = simulate_trades_all_tickers(
        all_predictions, initial_capital)

    # Display Transaction History as a DataFrame
    transaction_df = pd.DataFrame(transaction_history)
    transaction_df.sort_values('Date', inplace=True)
    transaction_df.reset_index(drop=True, inplace=True)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width', None)  # Auto-detect display width

    print("\nTransaction History:")
    print(transaction_df[['Date', 'Ticker', 'Action', 'Price', 'Shares',
                          'Transaction Value', 'Profit/Loss', 'Portfolio Value']])

    # Save Transaction History to CSV
    transaction_history_csv_path = os.path.join(
        os.getcwd(), 'transaction_history.csv')
    transaction_df.to_csv(transaction_history_csv_path, index=False)
    print(
        f"\nTransaction history saved to '{transaction_history_csv_path}'\n")

    print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")

    # Calculate Sharpe Ratio
    if not portfolio_values_df.empty:
        # Resample portfolio values to daily frequency, forward-fill missing values
        portfolio_values_df = portfolio_values_df.asfreq('D', method='ffill')
        # Calculate daily returns
        portfolio_values_df['daily_return'] = portfolio_values_df['portfolio_value'].pct_change(
        )
        # Drop NaN values
        portfolio_values_df.dropna(inplace=True)
        # Calculate Sharpe Ratio
        average_return = portfolio_values_df['daily_return'].mean()
        std_return = portfolio_values_df['daily_return'].std()
        sharpe_ratio = (average_return / std_return) * np.sqrt(252)
        print(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("\nNo trades were made. Sharpe Ratio cannot be calculated.")


if __name__ == "__main__":
    main()
