import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from src.utils.database import get_connection

load_dotenv()


# These are the exact columns we feed into our models —
# defined once here so both XGBoost and LSTM use identical input shapes
FEATURE_COLUMNS = [
    "avg_news_risk",
    "max_news_risk",
    "high_risk_article_count",
    "geopolitical_count",
    "natural_disaster_count",
    "labor_dispute_count",
    "trade_policy_count",
    "logistics_count",
    "avg_wind_speed",
    "max_wind_speed",
    "severe_weather_port_count",
    "avg_oil_change",
    "avg_shipping_stock_change",
    "market_stress_score",
]

# Our target column — the disruption risk index we engineered in Phase 4
TARGET_COLUMN = "disruption_risk_index"

# Thresholds for converting the continuous risk index into
# discrete risk classes for XGBoost classification
# Low: 0.0-0.3, Medium: 0.3-0.6, High: 0.6-1.0
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
}


def load_feature_vectors() -> pd.DataFrame:
    """
    Loads all feature vectors from the database into a pandas DataFrame.

    Returns:
        DataFrame with all feature vectors ordered by window_start,
        or an empty DataFrame if none exist
    """
    # SQLAlchemy engine — pd.read_sql() works best with SQLAlchemy
    # rather than raw psycopg2 connections
    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )

    query = f"""
        SELECT
            window_start,
            {', '.join(FEATURE_COLUMNS)},
            {TARGET_COLUMN}
        FROM feature_vectors
        ORDER BY window_start ASC
    """

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    logger.info(f"Loaded {len(df)} feature vectors from database")
    return df


def risk_index_to_class(risk_index: float) -> int:
    """
    Converts a continuous risk index to a discrete risk class.

    Classes:
        0 = Low risk    (0.0 - 0.3)
        1 = Medium risk (0.3 - 0.6)
        2 = High risk   (0.6 - 1.0)

    Args:
        risk_index: float between 0 and 1

    Returns:
        Integer class label (0, 1, or 2)
    """
    if risk_index < RISK_THRESHOLDS["low"]:
        return 0
    elif risk_index < RISK_THRESHOLDS["medium"]:
        return 1
    else:
        return 2


def prepare_xgboost_data(df: pd.DataFrame):
    """
    Prepares data for XGBoost classification.

    Converts continuous risk index to discrete class labels.
    No sequence structure needed — XGBoost treats each row independently.

    Args:
        df: DataFrame from load_feature_vectors()

    Returns:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of class labels (0, 1, 2)
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].apply(risk_index_to_class).values

    logger.info(f"XGBoost data prepared — X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Class distribution — Low: {(y==0).sum()}, Medium: {(y==1).sum()}, High: {(y==2).sum()}")

    return X, y


def prepare_lstm_data(df: pd.DataFrame, sequence_length: int = 12):
    """
    Prepares sequential data for LSTM training.

    LSTM learns from sequences — each training sample is a window of
    `sequence_length` consecutive feature vectors, and the label is
    the risk index of the NEXT window (what we're predicting).

    Basic example with sequence_length=3:
        Input:  [window_1, window_2, window_3]  → Predict: window_4's risk
        Input:  [window_2, window_3, window_4]  → Predict: window_5's risk
        ...and so on, sliding one window at a time

    Args:
        df: DataFrame from load_feature_vectors()
        sequence_length: how many past windows to look at (default 12 = 6 hours)

    Returns:
        X: numpy array of shape (n_sequences, sequence_length, n_features)
        y: numpy array of target risk indices
        or (None, None) if insufficient data
    """
    if len(df) < sequence_length + 1:
        logger.warning(
            f"Insufficient data for LSTM: need {sequence_length + 1} rows, "
            f"have {len(df)}. Collect more data before training LSTM."
        )
        return None, None

    feature_data = df[FEATURE_COLUMNS].values
    target_data = df[TARGET_COLUMN].values

    X, y = [], []
    for i in range(len(feature_data) - sequence_length):
        # Each X sample is a sequence of `sequence_length` windows
        X.append(feature_data[i: i + sequence_length])
        # Each y is the risk index of the window AFTER the sequence
        y.append(target_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"LSTM data prepared — X shape: {X.shape}, y shape: {y.shape}")
    return X, y


if __name__ == "__main__":
    df = load_feature_vectors()

    if df.empty:
        print("No feature vectors found. Run the scheduler first.")
    else:
        print(f"\nLoaded {len(df)} feature vectors")
        print(f"\nColumn names:\n{list(df.columns)}")
        print(f"\nFirst row:\n{df.iloc[0].to_dict()}")

        X_xgb, y_xgb = prepare_xgboost_data(df)
        print(f"\nXGBoost — X: {X_xgb.shape}, y: {y_xgb.shape}")

        X_lstm, y_lstm = prepare_lstm_data(df)
        if X_lstm is not None:
            print(f"LSTM — X: {X_lstm.shape}, y: {y_lstm.shape}")
        else:
            print("LSTM — insufficient data (need 13+ feature vectors)")
