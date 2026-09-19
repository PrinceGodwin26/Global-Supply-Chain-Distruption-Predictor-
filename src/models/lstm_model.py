import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.models.data_loader import (
    load_feature_vectors,
    prepare_lstm_data,
    FEATURE_COLUMNS,
)

MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

MIN_SEQUENCES = 50  # minimum sequences needed for meaningful LSTM training


class LSTMNetwork(nn.Module):
    """
    The actual PyTorch neural network architecture.

    Architecture:
        Input  → LSTM layer(s) → Dropout → Linear → Output

    Why this architecture?
    - LSTM handles sequential dependencies (what happened 3 windows ago
      affects the current prediction)
    - Dropout prevents overfitting on small datasets by randomly
      disabling neurons during training, forcing the network to learn
      more robust representations
    - Single linear output layer produces a continuous risk score (0-1)
      suitable for regression rather than classification

    Args:
        input_size: number of features per timestep (14 in our case)
        hidden_size: number of LSTM memory units (larger = more capacity)
        num_layers: how many LSTM layers to stack (deeper = more complex patterns)
        dropout: fraction of neurons to randomly disable during training
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The LSTM layer — processes sequences and maintains hidden state
        # batch_first=True means input shape is (batch, sequence, features)
        # which is more intuitive than PyTorch's default (sequence, batch, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Dropout after LSTM — applied to the final hidden state
        self.dropout = nn.Dropout(dropout)

        # Linear layer maps from LSTM hidden state to our single output
        self.linear = nn.Linear(hidden_size, 1)

        # Sigmoid squashes output to [0, 1] — matches our risk score range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predicted risk scores of shape (batch_size, 1)
        """
        # lstm_out shape: (batch, sequence, hidden_size)
        # We only need the last timestep's output — that's the prediction
        # for the next window based on the whole sequence seen so far
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep: (batch, hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout and linear projection
        out = self.dropout(last_hidden)
        out = self.linear(out)
        out = self.sigmoid(out)

        return out


class LSTMRiskPredictor:
    """
    Predicts future supply chain disruption risk trends using an LSTM
    that learns sequential patterns from historical feature vectors.

    Unlike XGBoost which classifies current risk, LSTM learns temporal
    patterns — "when risk has been climbing for 6 consecutive windows,
    it tends to keep climbing" — enabling early warning before
    disruption becomes obvious from current data alone.
    """

    def __init__(self, sequence_length: int = 12):
        # sequence_length = 12 windows × 30 min = 6 hours of history
        # per prediction — enough to catch escalation patterns
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.device = torch.device("cpu")  # explicitly use CPU

    def _normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalizes features to [0, 1] range using min-max scaling.

        We use min-max (not StandardScaler) for LSTM because:
        - Our features are all naturally bounded (risk scores 0-1, counts 0-N)
        - Min-max preserves the relative relationships between values
        - Works well with sigmoid activation which also outputs [0, 1]

        Args:
            X: array of shape (n_sequences, sequence_length, n_features)
            fit: if True, compute min/max from data; if False, use stored values

        Returns:
            Normalized array of same shape
        """
        original_shape = X.shape
        # Reshape to 2D for scaling: (n_sequences * sequence_length, n_features)
        X_flat = X.reshape(-1, X.shape[-1])

        if fit:
            self.scaler = {
                "min": X_flat.min(axis=0),
                "max": X_flat.max(axis=0),
            }

        # Avoid division by zero when min == max (constant feature)
        range_vals = self.scaler["max"] - self.scaler["min"]
        range_vals[range_vals == 0] = 1

        X_scaled = (X_flat - self.scaler["min"]) / range_vals
        return X_scaled.reshape(original_shape)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 16,
              learning_rate: float = 0.001) -> dict:
        """
        Trains the LSTM on sequential feature data.

        Args:
            X: sequences of shape (n_sequences, sequence_length, n_features)
            y: target risk indices of shape (n_sequences,)
            epochs: number of full passes through training data
            batch_size: samples per gradient update
            learning_rate: step size for weight updates

        Returns:
            Dict with training history (loss per epoch)
        """
        if len(X) < MIN_SEQUENCES:
            logger.warning(
                f"Only {len(X)} sequences available — need {MIN_SEQUENCES} minimum. "
                f"Keep the scheduler running to collect more data."
            )
            return {}

        # Normalize inputs
        X_scaled = self._normalize(X, fit=True)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        # DataLoader handles batching and shuffling automatically
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the network
        self.model = LSTMNetwork(
            input_size=X.shape[2],  # number of features (14)
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        ).to(self.device)

        # Adam optimizer — adaptive learning rate, works well for LSTMs
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # MSE loss — suitable for regression (predicting continuous risk score)
        criterion = nn.MSELoss()

        history = {"loss": []}

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                # Zero gradients from previous batch
                optimizer.zero_grad()

                # Forward pass
                predictions = self.model(X_batch)

                # Compute loss
                loss = criterion(predictions, y_batch)

                # Backward pass — compute gradients
                loss.backward()

                # Clip gradients to prevent exploding gradient problem
                # (common in LSTMs without this)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update weights
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")

        self.is_trained = True
        logger.info("LSTM training complete")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts disruption risk for sequences of feature vectors.

        Args:
            X: sequences of shape (n_sequences, sequence_length, n_features)

        Returns:
            Predicted risk scores of shape (n_sequences,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self._normalize(X, fit=False)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().squeeze()

    def save(self):
        """Saves the trained model and scaler to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"LSTM model saved to {MODEL_PATH}")

    def load(self, input_size: int = 14):
        """Loads a previously trained model from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No saved model at {MODEL_PATH}")
        self.model = LSTMNetwork(input_size=input_size)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        logger.info("LSTM model loaded from disk")


if __name__ == "__main__":
    df = load_feature_vectors()

    if df.empty:
        print("No feature vectors found. Run the scheduler first.")
    else:
        X, y = prepare_lstm_data(df, sequence_length=12)

        if X is None:
            print(f"\nInsufficient data for LSTM training.")
            print(f"Need at least 13 feature vectors, currently have {len(df)}.")
            print("The scheduler collects one every 30 minutes.")
            print(f"Estimated time until ready: "
                  f"{max(0, (13 - len(df)) * 30)} minutes")
        else:
            predictor = LSTMRiskPredictor(sequence_length=12)
            history = predictor.train(X, y)

            if predictor.is_trained:
                predictor.save()
                predictions = predictor.predict(X)
                print("\nLSTM Predictions vs Actual:\n")
                for i, (pred, actual) in enumerate(zip(predictions, y)):
                    print(f"  Sequence {i+1}: Predicted={pred:.4f}, Actual={actual:.4f}")
            else:
                print("\nLSTM could not be trained — need more data.")
                print("Keep the scheduler running.")
