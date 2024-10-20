import yfinance as yf
import pandas as pd
from datetime import datetime

# List of tickers
tickers = ['NVDA', 'ORCL', 'QQQ', 'SPY', 'SOXL', 'TQQQ', 'BTC-USD']

# Define the time period (8 years back from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 8)).strftime('%Y-%m-%d')

# Function to download historical data
def download_data(ticker, start_date, end_date):
    try:
        # Download data for each ticker with the required columns
        df = yf.download(ticker, start=start_date, end=end_date)[['Open', 'High', 'Close', 'Adj Close', 'Volume']].copy()
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        df.rename(columns={'Adj Close': 'adj_close', 'Date': 'business_date'}, inplace=True)
        
        # Round price-related columns to two decimal places
        df[['Open', 'High', 'Close', 'adj_close']] = df[['Open', 'High', 'Close', 'adj_close']].round(2)
        
        return df[['ticker', 'business_date', 'Open', 'High', 'Close', 'adj_close', 'Volume']]
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

# Save to a CSV file
combined_data.to_csv('eod_data_full_2_decimal.csv', index=False)

print("EOD data for the last 8 years with prices rounded to 2 decimal places has been saved to 'eod_data_full_2_decimal.csv'")
