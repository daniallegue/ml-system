""" Simple app for inference"""

import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

load_dotenv()

app = FastAPI(title="ML Model Inference API", version="1.0")

# Config
SYMBOL = "AAPL"
MODEL_PATH = f"../models/files/{SYMBOL}_best_model.pth"
BEST_MODEL_NAME_PATH = f"../models/files/{SYMBOL}_best_model.txt"
SCALER_PATH = f"../models/files/{SYMBOL}_scaler.pkl"
SEQ_LENGTH = 50  # Must match training sequence length
INPUT_FEATURES = ['close', 'volume', 'ma_5', 'ma_15', 'rsi']


# Define the models (must match those used during training)
class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_length, hidden_size)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out


class GRUTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUTimeSeries, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)  # out: (batch, seq_length, hidden_size)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out


class FeedforwardTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FeedforwardTimeSeries, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_length * input_size)
        out = self.network(x)
        return out


# Define Pydantic model for request body
class PredictionRequest(BaseModel):
    features: List[List[float]]  # List of sequences, each sequence is a list of feature values


# Load scaler
try:
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

# Load best model name
try:
    with open(BEST_MODEL_NAME_PATH, "r") as f:
        best_model_name = f.read().strip()
    logging.info(f"Best model name retrieved: '{best_model_name}'")
except Exception as e:
    logging.error(f"Error reading best model name: {e}")
    best_model_name = None

# Initialize the model
model = None
if best_model_name:
    try:
        if best_model_name == 'LSTM':
            model = LSTMTimeSeries(input_size=len(INPUT_FEATURES), hidden_size=64, num_layers=2, output_size=1)
        elif best_model_name == 'GRU':
            model = GRUTimeSeries(input_size=len(INPUT_FEATURES), hidden_size=64, num_layers=2, output_size=1)
        elif best_model_name == 'Feedforward':
            model = FeedforwardTimeSeries(input_size=len(INPUT_FEATURES) * SEQ_LENGTH, hidden_size=64, num_layers=2,
                                          output_size=1)
        else:
            raise ValueError(f"Unknown model name: {best_model_name}")

        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        logging.info(f"Model '{best_model_name}' loaded successfully from '{MODEL_PATH}'")
    except Exception as e:
        logging.error(f"Error loading model: {e}")


# API Endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame(request.features, columns=INPUT_FEATURES)
        logging.info(f"Received input data with shape: {input_data.shape}")

        # Scale features
        input_scaled = scaler.transform(input_data)
        logging.info(f"Scaled input data shape: {input_scaled.shape}")

        # Prepare data for model
        if best_model_name == 'Feedforward':
            # Flatten the input sequence
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            # Ensure input size matches
            if input_tensor.shape[1] != SEQ_LENGTH * len(INPUT_FEATURES):
                raise ValueError(f"Expected input size {SEQ_LENGTH * len(INPUT_FEATURES)}, got {input_tensor.shape[1]}")
        else:
            # Reshape for LSTM/GRU: (batch_size, seq_length, input_size)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).reshape(-1, SEQ_LENGTH, len(INPUT_FEATURES))

        # Make predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.cpu().numpy().flatten()

        # Return predictions
        return {"predictions": predictions.tolist()}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
