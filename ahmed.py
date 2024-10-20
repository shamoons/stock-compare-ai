import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import talib

# List of tickers
tickers = ['NVDA', 'ORCL', 'QQQ', 'SPY', 'SOXL', 'TQQQ']

# Define the time period (8 years back from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 8)).strftime('%Y-%m-%d')

# Function to download historical data
def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)[['Open', 'High', 'Close', 'Adj Close', 'Volume']].copy()
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        df.rename(columns={'Adj Close': 'adj_close', 'Date': 'business_date'}, inplace=True)
        df[['Open', 'High', 'Close', 'adj_close']] = df[['Open', 'High', 'Close', 'adj_close']].round(2)
        return df[['ticker', 'business_date', 'Open', 'High', 'Close', 'adj_close', 'Volume']]
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

# Download data for all tickers
all_data = []
for ticker in tickers:
    print(f"Downloading data for {ticker}")
    data = download_data(ticker, start_date, end_date)
    if data is not None:
        all_data.append(data)

# Concatenate all data into a single DataFrame
combined_data = pd.concat(all_data)

# Calculate Technical Indicators
def calculate_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['Close'].rolling(window=20).std() * 2)

    # Volume Average
    df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()

    return df

combined_data = calculate_indicators(combined_data)

# Generate Combined Buy/Sell Signals
def generate_combined_signals(df):
    df['Buy_Signal'] = (
        (df['Close'] > df['SMA_200']) &
        (df['SMA_10'] > df['SMA_50']) &
        (df['RSI'] < 30) &  # Adjusted RSI threshold
        (df['MACD'] > df['Signal_Line']) &
        (df['Close'] < df['BB_Lower']) &
        (df['Volume'] > df['Volume_Avg'])  # Confirm with volume
    ).astype(int)
    
    df['Sell_Signal'] = (
        (df['Close'] < df['SMA_200']) &
        (df['SMA_10'] < df['SMA_50']) &
        (df['RSI'] > 70) &  # Adjusted RSI threshold
        (df['MACD'] < df['Signal_Line']) &
        (df['Close'] > df['BB_Upper']) &
        (df['Volume'] > df['Volume_Avg'])  # Confirm with volume
    ).astype(int)

    return df

combined_data = generate_combined_signals(combined_data)

# Backtesting Function
def backtest_strategy(df):
    initial_capital = 10000
    shares = 0
    capital = initial_capital

    for index, row in df.iterrows():
        if row['Buy_Signal'] == 1 and capital > 0:  # Buy condition
            shares = capital // row['Close']
            capital -= shares * row['Close']
            print(f"Buying {shares} shares at {row['Close']} on {row['business_date']}")
        
        elif row['Sell_Signal'] == 1 and shares > 0:  # Sell condition
            capital += shares * row['Close']
            print(f"Selling {shares} shares at {row['Close']} on {row['business_date']}")
            shares = 0

    final_portfolio_value = capital + shares * df.iloc[-1]['Close']
    return final_portfolio_value

final_value = backtest_strategy(combined_data)

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.01):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess_returns = returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio

# Example Sharpe Ratio Calculation
portfolio_values = [initial_capital, final_value]
sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
print(f"Final Portfolio Value: ${final_value:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Save to a CSV file
combined_data.to_csv('eod_data_full_with_signals.csv', index=False)
print("EOD data with buy/sell signals has been saved to 'eod_data_full_with_signals.csv'")
