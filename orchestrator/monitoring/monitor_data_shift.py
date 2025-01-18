"""Monitors data-distribution shift"""

import os
import json
from dotenv import load_dotenv
import redis
import pandas as pd
from scipy.stats import ks_2samp
from prefect import task, flow
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import io

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

KS_THRESHOLD = 0.05 # Common Value
SD_THRESHOLD = 0.8

@task
def retrieve_lastest_features(symbol : str, window : int = 100) -> pd.DataFrame:
    """
    Retrieves the latest 'window' number of feature records from Redis.
    """
    redis_key = f"{symbol}_features"
    feature_jsons = redis_client.lrange(redis_key, -window, -1)
    if not feature_jsons:
        print(f"No features found in Redis under key '{redis_key}'.")
        return pd.DataFrame() # Return empty df

    features = [json.loads(f) for f in feature_jsons]
    df = pd.DataFrame(features)
    return df


@task
def load_reference_data(symbol: str, window: int = 100) -> pd.DataFrame:
    """
    Loads reference data for comparison. This could be from a historical dataset or predefined snapshot.
    """
    redis_key = f"{symbol}_features"
    feature_jsons = redis_client.lrange(redis_key, 0, window - 1)
    if not feature_jsons:
        print(f"No reference features found in Redis under key '{redis_key}'.")
        return pd.DataFrame()

    features = [json.loads(f) for f in feature_jsons]
    df = pd.DataFrame(features)
    return df

@task
def detect_seasonality_shift(current_df: pd.DataFrame, reference_df: pd.DataFrame, symbol: str):
    """
    Detects shifts in seasonality patterns using seasonal decomposition.
    """
    feature = 'close'  # Choose the feature to analyze
    if feature not in current_df.columns or feature not in reference_df.columns:
        print(f"Feature '{feature}' not found for seasonality detection.")
        return False

    # Decompose seasonality in reference data
    reference_decompose = seasonal_decompose(reference_df.set_index('timestamp')[feature], model='additive', period=60)  # period=60 for hourly data if 1min intervals
    current_decompose = seasonal_decompose(current_df.set_index('timestamp')[feature], model='additive', period=60)

    # Compare seasonal components
    reference_seasonal = reference_decompose.seasonal.dropna()
    current_seasonal = current_decompose.seasonal.dropna()

    # Perform statistical comparison (e.g., correlation)
    correlation = reference_seasonal.corr(current_seasonal)
    print(f"Seasonality correlation for '{feature}': {correlation}")

    if correlation < SD_THRESHOLD:  # Threshold can be adjusted
        print(f"Seasonality shift detected in feature '{feature}' (correlation = {correlation}).")
        return True
    else:
        print(f"No significant seasonality shift detected in feature '{feature}' (correlation = {correlation}).")
        return False
@task
def perform_ks_test(ref_data : pd.Series, new_data : pd.Series) -> float:
    """
    Performs the KS-Test between reference_data and new_data.
    Returns the p-value.
    """

    stat, p_value = ks_2samp(ref_data, new_data)
    return p_value

@task
def detect_distribution_shift(current_df: pd.DataFrame, ref_df: pd.DataFrame) -> bool:
    """
    Detects distribution shift using KS-Test for key features.
    Returns True if shift is detected, else False.
    """

    shift_detected = False
    features_to_test = ['close', 'volume', 'ma_5', 'ma_15', 'rsi']

    for feature in features_to_test:
        if feature not in current_df.columns or feature not in ref_df.columns:
            print(f"Feature '{feature}' not found in data.")
            continue

        current_series = current_df[feature].dropna()
        ref_series = ref_df[feature].dropna()

        p_value = perform_ks_test(ref_series, current_series)
        if p_value < KS_THRESHOLD:
            print(f"Distribution shift detected in feature '{feature}' (p-value = {p_value}).")
            shift_detected = True


    return shift_detected

@task
def plot_feature_distribution(ref_df : pd.DataFrame, current_df : pd.DataFrame, features : list, symbol : str):
    """
    Plots the distribution of selected features for reference and current data.
    """
    for feature in features:
        if feature not in ref_df.columns or feature not in current_df.columns:
            continue

        plt.figure(figsize=(10, 6))
        plt.hist(ref_df[feature], bins=50, alpha=0.5, label='Reference')
        plt.hist(current_df[feature], bins=50, alpha=0.5, label='Current')
        plt.title(f'Distribution Comparison for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

@flow
def data_shift_monitoring_flow(symbol : str = "AAPL", window: int = 100):
    """
    Prefect flow to monitor data distribution shift.
    """

    current_df = retrieve_lastest_features(symbol, window)
    ref_df = load_reference_data(symbol, window)

    if current_df.empty or ref_df.empty:
        print("Insufficient data for distribution shift monitoring.")
        return

    shift_detected = detect_distribution_shift(current_df, ref_df)

    seasonality_shift = detect_seasonality_shift(current_df, ref_df, symbol)

    # Combine shift detections
    total_shift = shift_detected or seasonality_shift

    if total_shift:
        print("Data distribution shift detected! Triggering alert or retraining process.")
        # TODO: Trigger retraining with new data
    else:
        print("No significant data distribution shift detected.")

        # Optional: Plot feature distributions for visual inspection
    features_to_plot = ['close', 'volume', 'ma_5', 'ma_15', 'rsi']
    plot_feature_distribution(ref_df, current_df, features_to_plot, symbol)

if __name__ == "__main__":
    data_shift_monitoring_flow()