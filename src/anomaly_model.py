"""
Anomaly Detection Model for EV Charging Sessions.

Uses Isolation Forest trained on normal data only to detect anomalous sessions.
This is an independent, standalone implementation for cybersecurity-oriented
behavioral anomaly detection.

Model choice: Isolation Forest
- Justification: Isolation Forest is well-suited for anomaly detection when
  we train on normal (or predominantly normal) data. It isolates observations
  by randomly selecting features and split values; anomalies require fewer
  splits to be isolated, yielding lower path lengths and thus higher anomaly
  scores. It is efficient, scalable, and does not assume a specific distribution.
  One-Class SVM is an alternative but is more sensitive to hyperparameters and
  scales less well with sample size. For mixed numerical features and the
  target of >= 95% accuracy, Isolation Forest is a robust choice.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, List


def train_isolation_forest(
    X_normal: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 100,
    max_samples: str | int = "auto",
    random_state: int = 42,
) -> IsolationForest:
    """
    Train Isolation Forest on normal data only.

    Parameters
    ----------
    X_normal : np.ndarray
        Feature matrix of normal samples (label=0). Shape (n_samples, n_features).
    contamination : float
        Expected fraction of outliers. Use a small value (e.g. 0.01) when
        training on normal-only data so the model still produces anomaly scores;
        sklearn requires contamination in (0, 0.5] or 'auto'.
    n_estimators : int
        Number of trees in the forest. More trees improve stability and separation.
    max_samples : str or int
        Number of samples to draw for each tree. 'auto' uses min(256, n_samples).
    random_state : int
        Reproducibility.

    Returns
    -------
    model : IsolationForest
        Fitted Isolation Forest. Predictions: -1 = anomaly, 1 = normal.
        decision_scores = -model.decision_function(X) so that higher = more anomalous.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_normal)
    return model


def predict_and_score(
    model: IsolationForest,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get binary predictions (-1/1) and anomaly scores for X.

    Isolation Forest's decision_function returns negative values for anomalies
    and positive for normal. We convert to a positive anomaly score (higher = more anomalous)
    so that it can be passed to a risk scoring engine.

    Returns
    -------
    predictions : np.ndarray
        -1 = anomaly, 1 = normal (sklearn convention).
    anomaly_scores : np.ndarray
        Non-negative; higher value means more anomalous. Derived from -decision_function
        and shifted so that scores are in a usable range.
    """
    raw_scores = model.decision_function(X)
    # In sklearn Isolation Forest, lower decision_function = more anomalous.
    # Use -raw_scores so that higher = more anomalous; then shift to non-negative.
    anomaly_scores = -raw_scores
    min_s = anomaly_scores.min()
    if min_s < 0:
        anomaly_scores = anomaly_scores - min_s
    predictions = model.predict(X)
    return predictions, anomaly_scores


def anomaly_predictions_to_binary(predictions: np.ndarray) -> np.ndarray:
    """
    Convert Isolation Forest output (-1 = anomaly, 1 = normal) to binary labels
    for evaluation: 0 = normal, 1 = anomaly.
    """
    return (predictions == -1).astype(int)


def threshold_from_normal_scores(
    anomaly_scores_normal: np.ndarray,
    percentile: float = 92.0,
) -> float:
    """
    Compute decision threshold from anomaly scores of normal training samples.
    Samples with score above this threshold are classified as anomaly.
    Using a percentile (e.g. 92) allows tuning toward target accuracy/recall.
    """
    return float(np.percentile(anomaly_scores_normal, percentile))


def predict_with_threshold(
    anomaly_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Binary prediction from anomaly scores: 1 if score > threshold else 0.
    Returns 0 = normal, 1 = anomaly (same as anomaly_predictions_to_binary).
    """
    return (anomaly_scores > threshold).astype(int)
