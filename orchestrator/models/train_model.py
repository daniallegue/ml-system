""" Script to train model """

# my_continual_learning_project/orchestrator/train_model.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from model import LSTMTimeSeries
from data_pipeline import prepare_dataloaders
from prefect import task, flow

load_dotenv()

# Configuration
SYMBOL = "AAPL"
WINDOW = 1000
BATCH_SIZE = 10
SEQ_LENGTH = 50
PRED_LENGTH = 1
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = PRED_LENGTH
MODEL_PATH = f"models/{SYMBOL}_lstm.pth"
SCALER_PATH = f"models/{SYMBOL}_scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@task
def train_model():
    """
    Trains the LSTM model on the prepared data.
    """
    print("Preparing DataLoaders...")
    # Prepare DataLoaders
    try:
        train_loader, val_loader = prepare_dataloaders(SYMBOL, WINDOW, BATCH_SIZE, SEQ_LENGTH)
        print("DataLoaders prepared successfully.")
    except Exception as e:
        print(f"Error preparing DataLoaders: {e}")
        return

    # Initialize the model
    try:
        input_size = train_loader.dataset.scaled_data.shape[1]
        print(f"Input size: {input_size}")
        model = LSTMTimeSeries(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model = model.to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # Define Loss and Optimizer
    try:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Loss function and optimizer defined.")
    except Exception as e:
        print(f"Error defining loss/optimizer: {e}")
        return

    # Training Loop
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch_idx, (X, y) in enumerate(train_loader):
            try:
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            except Exception as e:
                print(f"Error during training at batch {batch_idx}: {e}")
                continue

        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_idx, (X_val, y_val) in enumerate(val_loader):
                try:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_losses.append(loss.item())
                except Exception as e:
                    print(f"Error during validation at batch {val_idx}: {e}")
                    continue

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')

        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {e}")

@task
def evaluate_model():
    """
    Evaluates the trained model on validation data.
    """
    print("Evaluating the model...")
    # Prepare DataLoaders
    try:
        _, val_loader = prepare_dataloaders(SYMBOL, WINDOW, BATCH_SIZE, SEQ_LENGTH)
        print("Validation DataLoader prepared.")
    except Exception as e:
        print(f"Error preparing Validation DataLoader: {e}")
        return float('inf')

    # Initialize the model
    try:
        input_size = val_loader.dataset.scaled_data.shape[1]
        print(f"Input size for evaluation: {input_size}")
        model = LSTMTimeSeries(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model_path = f"models/{SYMBOL}_lstm.pth"
        # Set weights_only=True to address FutureWarning
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading model for evaluation: {e}")
        return float('inf')

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for val_idx, (X_val, y_val) in enumerate(val_loader):
            try:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                outputs = model(X_val)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_val.cpu().numpy())
            except Exception as e:
                print(f"Error during evaluation at batch {val_idx}: {e}")
                continue

    if not all_preds or not all_targets:
        print("No predictions or targets available for evaluation.")
        return float('inf')

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        print(f"Validation RMSE: {rmse:.4f}")
        return rmse
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return float('inf')

@flow
def model_training_flow():
    """
    Prefect flow that handles model training and evaluation.
    """
    train_model()
    rmse = evaluate_model()

    # TODO: Monitor this value and store it in BigQuery -> Trigger retraining
    print(f"Training completed with Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    model_training_flow()
