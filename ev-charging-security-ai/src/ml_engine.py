"""
Machine Learning Anomaly Detection Engine.

Uses Isolation Forest trained only on normal telemetry to produce an anomaly
score per timestamp. Isolation Forest is chosen because: (1) it is designed
for anomaly detection with no distributional assumptions; (2) it trains on
normal data and isolates outliers via random splits (anomalies need fewer
splits → higher score); (3) it scales well to time-series features and
handles mixed numerical inputs; (4) it provides a continuous anomaly score
suitable for correlation with rule-based outputs.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

# Contamination must be in (0, 0.5] for sklearn
CONTAMINATION = 0.01
N_ESTIMATORS = 100
RANDOM_STATE = 42

FEATURE_COLS = [
    "voltage",
    "current",
    "power_kw",
    "energy_kwh",
    "meter_value_increment",
    "interruption_flag",
    "error_code",
]


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better separation."""
    df = df.copy()
    df["power_energy_ratio"] = np.where(
        df["meter_value_increment"] > 1e-6,
        df["power_kw"] / (df["meter_value_increment"] * 60),
        1.0,
    )
    df["power_energy_ratio"] = df["power_energy_ratio"].clip(0, 10)
    return df


def prepare_ml_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for ML (with derived features)."""
    df = _add_derived_features(telemetry_df)
    cols = FEATURE_COLS + ["power_energy_ratio"]
    return df[cols].fillna(0)


def train_ml_engine(
    normal_telemetry: pd.DataFrame,
    contamination: float = CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Train Isolation Forest on normal data only.
    Returns fitted model and scaler for use at inference.
    """
    X = prepare_ml_features(normal_telemetry)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)
    return model, scaler


def score_telemetry(
    model: IsolationForest,
    scaler: StandardScaler,
    telemetry_df: pd.DataFrame,
) -> np.ndarray:
    """
    Output anomaly score per timestamp. Higher = more anomalous.
    Uses -decision_function and shifts to non-negative.
    """
    X = prepare_ml_features(telemetry_df)
    X_scaled = scaler.transform(X)
    raw = model.decision_function(X_scaled)
    # Lower decision_function => more anomalous
    scores = -raw
    if scores.min() < 0:
        scores = scores - scores.min()
    return scores


def session_ml_scores(
    telemetry_df: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """
    Attach anomaly score to each row and compute per-session aggregate.
    Returns session-level DataFrame: session_id, ml_score_mean, ml_score_max.
    """
    df = telemetry_df.copy()
    df["ml_anomaly_score"] = scores
    session_agg = df.groupby("session_id").agg(
        ml_score_mean=("ml_anomaly_score", "mean"),
        ml_score_max=("ml_anomaly_score", "max"),
    ).reset_index()
    return session_agg


def run_ml_engine(
    telemetry_df: pd.DataFrame,
    normal_mask: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, IsolationForest, StandardScaler]:
    """
    Train on normal rows only, score full telemetry, return per-row scores,
    per-session summary, model, and scaler.
    """
    normal_df = telemetry_df[normal_mask]
    model, scaler = train_ml_engine(normal_df)
    scores = score_telemetry(model, scaler, telemetry_df)
    session_summary = session_ml_scores(telemetry_df, scores)
    return scores, session_summary, model, scaler
