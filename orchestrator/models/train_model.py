# my_continual_learning_project/orchestrator/train_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from model import LSTMTimeSeries, GRUTimeSeries, FeedforwardTimeSeries
from data_pipeline import prepare_dataloaders
from multi_armed_bandit import MultiArmedBandit
from prefect import task, flow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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
MODEL_PATH = f"models/files/{SYMBOL}_best_model.pth"
BEST_MODEL_NAME_PATH = f"models/files/{SYMBOL}_best_model.txt"
SCALER_PATH = f"models/files/{SYMBOL}_scaler.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

@task
def train_model():
    """
    Trains multiple models using the Multi-Armed Bandit algorithm.
    """
    logging.info("Preparing DataLoaders...")
    # Prepare DataLoaders
    try:
        train_loader, val_loader = prepare_dataloaders(SYMBOL, WINDOW, BATCH_SIZE, SEQ_LENGTH)
        logging.info("DataLoaders prepared successfully.")
    except Exception as e:
        logging.error(f"Error preparing DataLoaders: {e}")
        return

    # Define the models (arms)
    input_size = train_loader.dataset.scaled_data.shape[1]  # 5
    flattened_input_size = input_size * SEQ_LENGTH       # 5 * 50 = 250
    MODELS = {
        'LSTM': LSTMTimeSeries(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE),
        'GRU': GRUTimeSeries(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE),
        'Feedforward': FeedforwardTimeSeries(input_size=flattened_input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
    }

    # Initialize the Multi-Armed Bandit
    bandit = MultiArmedBandit(n_arms=len(MODELS))
    arm_names = list(MODELS.keys())

    # Move all models to device
    for name, model in MODELS.items():
        model.to(device)
        logging.info(f"Model '{name}' initialized and moved to device.")

    # Define Loss and Optimizer for each model
    criteria = {name: nn.MSELoss() for name in arm_names}
    optimizers = {name: optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in MODELS.items()}

    # Initialize metrics tracking
    metrics = {name: [] for name in arm_names}

    # Training Loop with MAB
    logging.info("Starting training with Multi-Armed Bandit...")
    for epoch in range(1, EPOCHS + 1):
        bandit_losses = {name: [] for name in arm_names}

        for batch_idx, (X, y) in enumerate(train_loader):
            # Select a model based on MAB
            chosen_arm = bandit.select_arm()
            chosen_model_name = arm_names[chosen_arm]
            chosen_model = MODELS[chosen_model_name]
            chosen_optimizer = optimizers[chosen_model_name]
            chosen_criterion = criteria[chosen_model_name]

            X = X.to(device)
            y = y.to(device)

            try:
                chosen_optimizer.zero_grad()
                outputs = chosen_model(X)
                loss = chosen_criterion(outputs, y)
                loss.backward()
                chosen_optimizer.step()

                bandit_losses[chosen_model_name].append(loss.item())
            except Exception as e:
                logging.error(f"Error during training at batch {batch_idx} with model '{chosen_model_name}': {e}")
                continue

        # Calculate average losses per model for this epoch
        avg_losses = {}
        for name, losses in bandit_losses.items():
            avg_losses[name] = np.mean(losses) if losses else float('inf')

        # Select the best model based on average loss
        best_model_name = min(avg_losses, key=avg_losses.get)
        best_loss = avg_losses[best_model_name]

        # Reward: inverse of loss (higher is better)
        reward = 1 / best_loss if best_loss != 0 else 0

        # Update the bandit with the reward for the best arm
        best_arm = arm_names.index(best_model_name)
        bandit.update(best_arm, reward)

        # Log metrics
        metrics[best_model_name].append(best_loss)

        logging.info(f"Epoch [{epoch}/{EPOCHS}] - Selected Model: '{best_model_name}' - Avg Train Loss: {best_loss:.4f}")

    # After training, select the best model based on MAB estimates
    final_estimates = bandit.get_estimates()
    best_arm_final = np.argmax(final_estimates)
    best_model_final_name = arm_names[best_arm_final]
    best_model_final = MODELS[best_model_final_name]
    logging.info(f"Best model selected: '{best_model_final_name}' with estimated success probability: {final_estimates[best_arm_final]:.4f}")

    # Save the best model's state_dict
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(best_model_final.state_dict(), MODEL_PATH)
        logging.info(f"Model saved to '{MODEL_PATH}'")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

    # Save the best model's name for evaluation
    try:
        with open(BEST_MODEL_NAME_PATH, "w") as f:
            f.write(best_model_final_name)
        logging.info(f"Best model name saved to '{BEST_MODEL_NAME_PATH}'")
    except Exception as e:
        logging.error(f"Error saving best model name: {e}")

@task
def evaluate_model():
    """
    Evaluates the trained model on validation data.
    """
    logging.info("Evaluating the model...")
    # Prepare DataLoaders
    try:
        _, val_loader = prepare_dataloaders(SYMBOL, WINDOW, BATCH_SIZE, SEQ_LENGTH)
        logging.info("Validation DataLoader prepared.")
    except Exception as e:
        logging.error(f"Error preparing Validation DataLoader: {e}")
        return float('inf')

    # Read the best model's name
    BEST_MODEL_NAME_PATH = f"models/files/{SYMBOL}_best_model.txt"
    try:
        with open(BEST_MODEL_NAME_PATH, "r") as f:
            best_model_name = f.read().strip()
        logging.info(f"Best model name retrieved: '{best_model_name}'")
    except Exception as e:
        logging.error(f"Error reading best model name: {e}")
        return float('inf')

    # Initialize the appropriate model
    try:
        input_size = val_loader.dataset.scaled_data.shape[1]
        flattened_input_size = input_size * SEQ_LENGTH
        if best_model_name == 'LSTM':
            model = LSTMTimeSeries(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
        elif best_model_name == 'GRU':
            model = GRUTimeSeries(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
        elif best_model_name == 'Feedforward':
            model = FeedforwardTimeSeries(input_size=flattened_input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
        else:
            raise ValueError(f"Unknown model name: {best_model_name}")

        model_path = f"models/files/{SYMBOL}_best_model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        logging.info(f"Model '{best_model_name}' loaded successfully for evaluation.")
    except Exception as e:
        logging.error(f"Error loading model for evaluation: {e}")
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
                logging.error(f"Error during evaluation at batch {val_idx}: {e}")
                continue

    if not all_preds or not all_targets:
        logging.warning("No predictions or targets available for evaluation.")
        return float('inf')

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        logging.info(f"Validation RMSE: {rmse:.4f}")
        return rmse
    except Exception as e:
        logging.error(f"Error calculating RMSE: {e}")
        return float('inf')

@flow
def model_training_flow():
    """
    Prefect flow that handles model training and evaluation using Multi-Armed Bandit.
    """
    train_model()
    rmse = evaluate_model()

    # TODO: Monitor this value and store it in BigQuery -> Trigger retraining
    logging.info(f"Training completed with Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    model_training_flow()