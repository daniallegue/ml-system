""" Data Pipeline module to train the model. """

import os
import json
from dotenv import load_dotenv
import redis
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch

load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Sequence Parameters
SEQUENCE_LENGTH = 50
PREDICTION_LENGTH = 1

# Initialize Redis Client
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

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=SEQUENCE_LENGTH, prediction_length=PREDICTION_LENGTH):
        """
        Initializes the TimeSeriesDataset.

        Parameters:
        - data (numpy.ndarray): The input feature data.
        - sequence_length (int): The length of the input sequences.
        - prediction_length (int): The length of the prediction targets.
        """
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.scaled_data) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, idx):
        """
        Retrieves the input sequence and target for a given index.

        Parameters:
        - idx (int): The index of the sample.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: The input sequence and target.
        """
        x = self.scaled_data[idx : idx + self.sequence_length]
        y = self.scaled_data[
            idx + self.sequence_length : idx + self.sequence_length + self.prediction_length, 0
        ]  # 'close' is at index 0 after excluding 'timestamp'
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_features(symbol: str, window: int = 1000) -> pd.DataFrame:
    """
    Retrieves the latest 'window' number of feature records from Redis.

    Parameters:
    - symbol (str): The stock symbol.
    - window (int): The number of latest records to retrieve.

    Returns:
    - pd.DataFrame: The retrieved feature data.
    """
    redis_key = f"{symbol}_features"
    feature_jsons = redis_client.lrange(redis_key, -window, -1)
    if not feature_jsons:
        raise ValueError(f"No features found in Redis under key '{redis_key}'.")

    features = [json.loads(f) for f in feature_jsons]
    df = pd.DataFrame(features)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

def prepare_dataloaders(symbol: str, window: int = 1000, batch_size: int = 64, sequence_length: int = SEQUENCE_LENGTH):
    """
    Prepares PyTorch DataLoaders for training and validation.

    Parameters:
    - symbol (str): The stock symbol.
    - window (int): The number of latest records to retrieve from Redis.
    - batch_size (int): The batch size for the DataLoaders.
    - sequence_length (int): The length of input sequences.

    Returns:
    - Tuple[DataLoader, DataLoader]: The training and validation DataLoaders.
    """
    df = get_features(symbol, window)

    # Exclude 'timestamp' from features
    feature_cols = ['close', 'volume', 'ma_5', 'ma_15', 'rsi']
    df = df[feature_cols]
    df = df.dropna()

    # Convert to numpy array
    data = df.values  # Fixed typo: df.value -> df.values

    # 80-20 split
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size - sequence_length - PREDICTION_LENGTH + 1 :]

    # Initialize Datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length, PREDICTION_LENGTH)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, PREDICTION_LENGTH)

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader
