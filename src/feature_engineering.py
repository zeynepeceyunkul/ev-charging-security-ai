"""
Feature Engineering Module for EV Charging Anomaly Detection.

Normalizes numerical features, selects relevant features for anomaly detection,
and prepares data for machine learning models. Part of a standalone project;
no external or previous team code is used.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


# Features used for anomaly detection (numerical only; exclude ids and label for fitting)
FEATURE_COLUMNS = [
    "charging_duration",
    "average_power_kw",
    "total_energy_kwh",
    "number_of_interruptions",
    "error_flag",
]

# Optional: derived feature for energy consistency (energy / (duration/60 * power))
DERIVED_FEATURE_NAME = "energy_consistency_ratio"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that help distinguish anomalies.

    - energy_consistency_ratio: total_energy_kwh / (charging_duration/60 * average_power_kw).
      Normal sessions cluster near 1.0; tampering or meter faults can deviate.
    """
    df = df.copy()
    # Avoid division by zero
    expected_energy = (df["charging_duration"] / 60) * df["average_power_kw"].replace(0, np.nan)
    df[DERIVED_FEATURE_NAME] = df["total_energy_kwh"] / expected_energy
    df[DERIVED_FEATURE_NAME] = df[DERIVED_FEATURE_NAME].fillna(1.0).clip(0, 10)
    return df


def get_feature_columns(include_derived: bool = True) -> List[str]:
    """Return the list of feature column names used for ML."""
    cols = list(FEATURE_COLUMNS)
    if include_derived:
        cols.append(DERIVED_FEATURE_NAME)
    return cols


def prepare_features_and_labels(
    df: pd.DataFrame,
    include_derived: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare feature matrix and labels from session DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FEATURE_COLUMNS and optionally allow derived feature.
    include_derived : bool
        Whether to add and use the energy_consistency_ratio feature.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (with derived feature if requested).
    y : np.ndarray
        Label array (0 = normal, 1 = anomalous). Only present if 'label' in df.
    feature_names : List[str]
        List of feature column names in X.
    """
    df = add_derived_features(df)
    feature_names = get_feature_columns(include_derived=include_derived)
    X = df[feature_names].copy()

    if "label" in df.columns:
        y = df["label"].values
        return X, y, feature_names
    return X, np.array([]), feature_names


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    feature_names: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """
    Normalize numerical features using StandardScaler (zero mean, unit variance).
    Fit on training data only to avoid data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame or None
        Optional test feature matrix to transform.
    feature_names : List[str] or None
        If provided, use these columns only; otherwise use all columns of X_train.

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training features.
    X_test_scaled : np.ndarray or None
        Scaled test features if X_test was provided.
    scaler : StandardScaler
        Fitted scaler for use at inference time.
    """
    if feature_names is not None:
        X_train = X_train[feature_names]
        if X_test is not None:
            X_test = X_test[feature_names].copy()
    else:
        X_test = X_test.copy() if X_test is not None else None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def train_test_split_by_label(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test, stratifying by label so that
    both normal and anomalous samples appear in test set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'label' column.
    test_ratio : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df, test_df
