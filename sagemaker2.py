import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sklearn.metrics import mean_squared_error

# Step 1: Download and Preprocess Stock Data

tickers = ['NVDA', 'ORCL', 'QQQ', 'SPY', 'SOXL', 'TQQQ', 'BTC-USD']
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today().replace(year=datetime.today().year - 8)).strftime('%Y-%m-%d')

def download_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)[['Adj Close']].copy()
        df.reset_index(inplace=True)
        df['ticker'] = ticker
        df.rename(columns={'Adj Close': 'price', 'Date': 'business_date'}, inplace=True)
        return df[['ticker', 'business_date', 'price']]
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

all_data = []
for ticker in tickers:
    print(f"Downloading data for {ticker}")
    data = download_data(ticker, start_date, end_date)
    if data is not None:
        all_data.append(data)

combined_data = pd.concat(all_data)

# Save locally
csv_file_path = os.path.join(os.getcwd(), 'eod_data_ticker_businessdate_price.csv')
combined_data.to_csv(csv_file_path, index=False)

print(f"EOD data for the last 8 years has been saved to '{csv_file_path}'")

# Step 2: Preprocess Data for DeepAR (Time Series JSON Format)

def convert_to_deepar_format(data):
    # Create a JSON file for each ticker in DeepAR format
    series_list = []
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        ticker_data = ticker_data.sort_values(by='business_date')
        series = {
            'start': str(ticker_data['business_date'].iloc[0]),
            'target': list(ticker_data['price'])
        }
        series_list.append(series)
    return series_list

deepar_train_data = convert_to_deepar_format(combined_data)

# Save train data in JSON Lines format
train_json_path = os.path.join(os.getcwd(), 'deepar_train.json')
with open(train_json_path, 'w') as f:
    for series in deepar_train_data:
        f.write(json.dumps(series) + '\n')

# Step 3: Upload Data to S3 for SageMaker

sagemaker_session = sagemaker.Session()
role = get_execution_role()

bucket = 'outputdata-4'  # Replace with your actual S3 bucket
train_s3_uri = sagemaker_session.upload_data(train_json_path, bucket=bucket, key_prefix='deepar-data')

# Step 4: Train DeepAR Model in SageMaker

deepar_container = sagemaker.image_uris.retrieve("forecasting-deepar", boto3.Session().region_name)

deepar_estimator = sagemaker.estimator.Estimator(
    image_uri=deepar_container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{sagemaker_session.default_bucket()}/output',
    sagemaker_session=sagemaker_session
)

deepar_estimator.set_hyperparameters(
    time_freq='D',  # Daily frequency
    context_length=30,  # Context length for the model (how much history to consider)
    prediction_length=10,  # Predict next 10 days
    epochs=20,  # Number of epochs for training
    early_stopping_patience=10,  # Early stopping
    mini_batch_size=64
)

train_input = TrainingInput(train_s3_uri, content_type="json")
deepar_estimator.fit({'train': train_input})

# Step 5: Deploy the DeepAR Model

deepar_predictor = deepar_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Step 6: Make Predictions with the Deployed DeepAR Model

def predict_future(predictor, start_date, target_series, num_days=10):
    # Ensure the target_series (last 30 days of data) is a Python list
    target_series = target_series.tolist()
    
    payload = {
        "instances": [
            {
                "start": start_date,
                "target": target_series[-30:],  # Use the last 30 days of data (context length)
            }
        ]
    }

    # Set Content-Type to application/json to avoid 415 error
    predictions = predictor.predict(json.dumps(payload), initial_args={"ContentType": "application/json"})
    
    return predictions

# Extract last 30 days of NVDA prices for prediction
nvda_data = combined_data[combined_data['ticker'] == 'NVDA']
nvda_prices = nvda_data['price'].values[-30:]

# Make predictions starting from today
start_date = datetime.today().strftime('%Y-%m-%d')
predicted_results = predict_future(deepar_predictor, start_date, nvda_prices, num_days=10)

# Extract predictions from response
predicted_values = json.loads(predicted_results)['predictions'][0]['mean']
dates = [datetime.today() + timedelta(days=i) for i in range(10)]

# Step 7: Get Real Stock Prices for the Next 10 Days from Yahoo Finance

real_stock_data = yf.download('NVDA', start=dates[0].strftime('%Y-%m-%d'), end=(dates[-1] + timedelta(days=1)).strftime('%Y-%m-%d'))
real_prices = real_stock_data['Adj Close'].values

# Ensure real prices match the prediction period
if len(real_prices) != 10:
    raise ValueError("Mismatch in the number of real prices and prediction length")

# Step 8: Calculate Mean Squared Error (MSE)

mse = mean_squared_error(real_prices, predicted_values)
print(f"Mean Squared Error (MSE): {mse}")

# Step 9: Save Real and Predicted Prices to a CSV File

# Create a DataFrame with real and predicted prices
predictions_df = pd.DataFrame({
    'date': dates,
    'real_price': real_prices,
    'predicted_price': predicted_values
})

# Save to CSV
output_csv_path = os.path.join(os.getcwd(), 'deepar_real_vs_predicted_prices.csv')
predictions_df.to_csv(output_csv_path, index=False)

print(f"Real and predicted stock prices have been saved to '{output_csv_path}'")

# Step 10: Clean Up Resources

deepar_predictor.delete_endpoint()

print("DeepAR model training, deployment, and prediction completed.")