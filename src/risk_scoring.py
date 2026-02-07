"""
Risk Scoring Engine for EV Charging Anomaly Detection.

Converts raw anomaly scores from the ML model into a 0–100 risk score and
categorizes sessions into Low, Medium, High, and Critical risk levels.
Designed for cybersecurity-oriented interpretation.
"""

import numpy as np
from typing import Tuple, List


# Risk level labels
RISK_LEVELS = ("Low", "Medium", "High", "Critical")


def score_to_risk_0_100(anomaly_scores: np.ndarray) -> np.ndarray:
    """
    Map anomaly scores to a 0–100 risk score.

    Uses min-max scaling over the provided scores so that the minimum score
    maps to 0 and the maximum to 100. For production, fixed percentiles or
    a fixed scaler fitted on historical data could be used instead.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        Raw anomaly scores (higher = more anomalous).

    Returns
    -------
    np.ndarray
        Risk scores in [0, 100]. Same shape as anomaly_scores.
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    if scores.size == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s <= min_s:
        return np.zeros_like(scores)
    risk = (scores - min_s) / (max_s - min_s) * 100.0
    return np.clip(risk, 0.0, 100.0)


def risk_score_to_level(risk_scores: np.ndarray) -> List[str]:
    """
    Categorize risk scores into Low, Medium, High, Critical.

    Thresholds (inclusive boundaries):
    - Low:      0 <= score < 25
    - Medium:  25 <= score < 50
    - High:    50 <= score < 75
    - Critical: 75 <= score <= 100

    Parameters
    ----------
    risk_scores : np.ndarray
        Risk scores in [0, 100].

    Returns
    -------
    List[str]
        Risk level for each score.
    """
    risk_scores = np.asarray(risk_scores)
    levels = []
    for s in risk_scores.flat:
        if s < 25:
            levels.append("Low")
        elif s < 50:
            levels.append("Medium")
        elif s < 75:
            levels.append("High")
        else:
            levels.append("Critical")
    return levels


def compute_risk(
    anomaly_scores: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert anomaly scores to 0–100 risk and risk levels.

    Parameters
    ----------
    anomaly_scores : np.ndarray
        Raw anomaly scores from the model.

    Returns
    -------
    risk_scores : np.ndarray
        Risk in [0, 100].
    risk_levels : List[str]
        One of Low, Medium, High, Critical per sample.
    """
    risk_scores = score_to_risk_0_100(anomaly_scores)
    risk_levels = risk_score_to_level(risk_scores)
    return risk_scores, risk_levels
