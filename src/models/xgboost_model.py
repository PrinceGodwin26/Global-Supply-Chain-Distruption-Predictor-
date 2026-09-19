import os
import pickle
from datetime import datetime, UTC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from loguru import logger

from src.models.data_loader import (
    load_feature_vectors,
    prepare_xgboost_data,
    FEATURE_COLUMNS,
    risk_index_to_class,
)

# Where trained models get saved — matches our data/models/ folder
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "xgboost_scaler.pkl")

# Minimum samples needed for meaningful training
# With fewer than this, results are statistically meaningless
MIN_SAMPLES = 20


class XGBoostRiskClassifier:
    """
    Classifies current supply chain disruption risk as Low/Medium/High
    using XGBoost gradient boosting on engineered feature vectors.

    Why XGBoost for this task?
    - Works well on tabular data (our feature vectors are exactly that)
    - Handles small-to-medium datasets well (unlike deep learning)
    - Produces feature importance scores — tells us which signals
      matter most, which is valuable for explainability in interviews
    - Fast to train and predict — suitable for real-time scoring

    Classes:
        0 = Low risk    (disruption_risk_index < 0.3)
        1 = Medium risk (0.3 <= disruption_risk_index < 0.6)
        2 = High risk   (disruption_risk_index >= 0.6)
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Trains the XGBoost classifier on prepared feature data.

        Args:
            X: feature matrix of shape (n_samples, n_features)
            y: class labels of shape (n_samples,)

        Returns:
            Dict of evaluation metrics, or empty dict if insufficient data
        """
        if len(X) < MIN_SAMPLES:
            logger.warning(
                f"Only {len(X)} samples available — need {MIN_SAMPLES} minimum. "
                f"Model will be trained but results may not be reliable. "
                f"Keep the scheduler running to collect more data."
            )

        # XGBoost requires all expected classes to be present in training data
        # With very little data, we may only have one class (e.g. all Medium)
        # In that case, we cannot train a meaningful multi-class model yet
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(
                f"Only {len(unique_classes)} risk class(es) present in data "
                f"({unique_classes}). Need at least 2 classes to train. "
                f"Keep the scheduler running — more data will introduce class variety."
            )
            return {}

        # StandardScaler normalizes features to mean=0, std=1
        # This ensures no single feature dominates due to scale differences
        # (e.g. high_risk_article_count can be 0-50, avg_news_risk is 0-1)
        X_scaled = self.scaler.fit_transform(X)

        # Only split train/test if we have enough data
        # With very few samples, we train on everything and skip evaluation
        if len(X) >= MIN_SAMPLES:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            logger.warning("Too few samples for train/test split — training on full dataset")
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y

        # XGBoost hyperparameters — tuned for small tabular datasets
        self.model = xgb.XGBClassifier(
            n_estimators=100,       # number of trees in the ensemble
            max_depth=4,            # depth of each tree — shallow prevents overfitting
            learning_rate=0.1,      # how much each tree contributes
            subsample=0.8,          # fraction of data used per tree (prevents overfitting)
            colsample_bytree=0.8,   # fraction of features used per tree
            use_label_encoder=False,
            eval_metric="mlogloss", # multi-class log loss
            random_state=42,
        )

        # Only pass eval_set when we have a proper train/test split
        # (when training on full dataset, eval_set causes class mismatch errors)
        fit_params = {"verbose": False}
        if len(X) >= MIN_SAMPLES:
            fit_params["eval_set"] = [(X_test, y_test)]

        self.model.fit(X_train, y_train, **fit_params)

        self.is_trained = True
        logger.info("XGBoost model trained successfully")

        # Evaluate on test set
        y_pred = self.model.predict(X_test)

        # Use only the classes actually present in the data
        # (with limited data, "High" class may not exist yet)
        present_classes = sorted(list(set(y_test) | set(y_pred)))
        class_name_map = {0: "Low", 1: "Medium", 2: "High"}
        present_names = [class_name_map[c] for c in present_classes]

        metrics = {
            "classification_report": classification_report(
                y_test, y_pred,
                labels=present_classes,
                target_names=present_names,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
        return metrics

    def predict(self, X: np.ndarray) -> dict:
        """
        Predicts risk class for new feature vectors.

        Args:
            X: feature matrix of shape (n_samples, n_features)

        Returns:
            Dict with predicted class labels and probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first or load a saved model.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        class_names = {0: "Low", 1: "Medium", 2: "High"}

        # probabilities shape depends on how many classes exist in training data
        # With limited data, "High" class may not exist yet — handle gracefully
        n_classes = probabilities.shape[1]
        return {
            "predictions": [class_names.get(p, "Unknown") for p in predictions],
            "probabilities": {
                "low": probabilities[:, 0].tolist() if n_classes > 0 else [0.0] * len(predictions),
                "medium": probabilities[:, 1].tolist() if n_classes > 1 else [0.0] * len(predictions),
                "high": probabilities[:, 2].tolist() if n_classes > 2 else [0.0] * len(predictions),
            },
        }

    def get_feature_importance(self) -> dict:
        """
        Returns feature importance scores — which features most influenced
        the model's decisions.

        This is one of XGBoost's key advantages over neural networks:
        full explainability. In a real system, you'd use this to tell
        stakeholders "the model is primarily driven by news risk score
        and oil price changes."

        Returns:
            Dict mapping feature names to importance scores, sorted descending
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(FEATURE_COLUMNS, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def save(self):
        """Saves the trained model and scaler to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Model saved to {MODEL_PATH}")

    def load(self):
        """Loads a previously trained model and scaler from disk."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No saved model found at {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        logger.info("Model loaded from disk")


if __name__ == "__main__":
    df = load_feature_vectors()

    if df.empty:
        print("No feature vectors found. Run the scheduler first.")
    else:
        X, y = prepare_xgboost_data(df)

        classifier = XGBoostRiskClassifier()
        metrics = classifier.train(X, y)

        if not classifier.is_trained:
            print("\nModel could not be trained yet — insufficient data.")
            print("The scheduler is running every 30 minutes.")
            print("Come back when more feature vectors have accumulated.")
        else:
            classifier.save()

            print("\nFeature Importance (what drives the model's decisions):\n")
            for feature, importance in classifier.get_feature_importance().items():
                bar = "█" * int(importance * 50)
                print(f"  {feature:<35} {importance:.4f} {bar}")

            print("\nPredicting on current data:")
            results = classifier.predict(X)
            for i, (pred, low, med, high) in enumerate(zip(
                results["predictions"],
                results["probabilities"]["low"],
                results["probabilities"]["medium"],
                results["probabilities"]["high"],
            )):
                print(f"  Window {i+1}: {pred} risk  "
                      f"(Low: {low:.2f}, Med: {med:.2f}, High: {high:.2f})")
