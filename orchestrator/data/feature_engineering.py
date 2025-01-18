""" Feature Engineering module using Redis"""

import os
import json
from dotenv import load_dotenv
import pandas as pd
import redis
from prefect import task

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True  # To handle strings instead of bytes
    )
    # Test connection
    redis_client.ping()
    print("Connected to Redis successfully.")
except redis.exceptions.ConnectionError as e:
    print(f"Redis connection error: {e}")
    raise e
@task
def engineer_features(data : dict, symbol : str):
    """
    Engineer features from raw stock data and store them in a Redis feature store.
    """

    time_series = data.get("Time Series (1min)", {})
    if not time_series:
        print("No time series data available")
        return

    # Convert to DataFrame for easier manipulation
    #TODO: Use polars to investigate efficiency

    try:
        df = pd.DataFrame.from_dict(time_series, orient='index')
        print(f"DataFrame created with shape: {df.shape}")
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return

    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'timestamp'})
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": int
    })

    # TODO: Get real features for time-series

    df = df.sort_values('timestamp')

    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_15'] = df['close'].rolling(window=15).mean()

    # Example Feature: Relative Strength Index (RSI)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=14).mean()
    ma_down = down.rolling(window=14).mean()
    rs = ma_up / ma_down
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values resulting from rolling calculations
    # TODO: Data Imputation
    df = df.dropna()

    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"DataFrame after feature engineering has shape: {df.shape}")


    # Convert DataFrame to JSON for Redis storage
    features = df.to_dict(orient='records')

    redis_key = f"{symbol}_features"
    try:
        redis_client.delete(redis_key)
        for feature in features:
            redis_client.rpush(redis_key, json.dumps(feature))
        print(f"Engineered and stored {len(features)} feature records in Redis under key '{redis_key}'.")
    except Exception as e:
        print(f"Error storing features in Redis: {e}")
