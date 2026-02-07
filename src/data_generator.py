"""
Data Simulation Module for EV Charging Sessions.

Generates synthetic EV charging session data for anomaly detection research.
This module is part of an independent, standalone project. No real hardware
or charging stations are involved; all data is simulated.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import uuid


# ---------------------------------------------------------------------------
# Normal session parameter ranges (realistic EV charging behavior)
# ---------------------------------------------------------------------------
NORMAL_DURATION_MIN = 15
NORMAL_DURATION_MAX = 120
NORMAL_POWER_MEAN = 11.0   # kW (AC Level 2 typical)
NORMAL_POWER_STD = 3.0
NORMAL_ENERGY_MEAN = 25.0  # kWh per session
NORMAL_ENERGY_STD = 15.0
NORMAL_INTERRUPTIONS_MAX = 2
NORMAL_ERROR_RATE = 0.02
NUM_STATIONS = 20
NUM_NORMAL_SAMPLES = 1000
NUM_ANOMALOUS_SAMPLES = 100


def _generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid.uuid4())[:12]


def _random_station_id() -> str:
    """Return a random station ID from the pool."""
    return f"STATION_{np.random.randint(1, NUM_STATIONS + 1):03d}"


def _random_start_time(base_date: datetime, days_span: int = 90) -> datetime:
    """Random start time within the last days_span days."""
    delta = timedelta(
        days=np.random.randint(0, days_span),
        hours=np.random.randint(0, 24),
        minutes=np.random.randint(0, 60),
    )
    return base_date - delta


def generate_normal_sessions(n: int = NUM_NORMAL_SAMPLES, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic normal EV charging sessions.

    Parameters
    ----------
    n : int
        Number of normal sessions to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: session_id, station_id, start_time, charging_duration,
        average_power_kw, total_energy_kwh, number_of_interruptions, error_flag, label.
    """
    np.random.seed(seed)
    base = datetime(2024, 1, 1, 0, 0, 0)

    rows = []
    for _ in range(n):
        duration = np.random.uniform(NORMAL_DURATION_MIN, NORMAL_DURATION_MAX)
        power = np.clip(
            np.random.normal(NORMAL_POWER_MEAN, NORMAL_POWER_STD),
            3.0, 22.0
        )
        # Energy roughly consistent with duration and power (with noise)
        energy = np.clip(
            (duration / 60) * power * np.random.uniform(0.85, 1.15),
            5.0, 80.0
        )
        interruptions = np.random.poisson(0.3)
        interruptions = min(interruptions, NORMAL_INTERRUPTIONS_MAX)
        error_flag = 1 if np.random.random() < NORMAL_ERROR_RATE else 0

        rows.append({
            "session_id": _generate_session_id(),
            "station_id": _random_station_id(),
            "start_time": _random_start_time(base),
            "charging_duration": round(duration, 2),
            "average_power_kw": round(power, 2),
            "total_energy_kwh": round(energy, 2),
            "number_of_interruptions": int(interruptions),
            "error_flag": int(error_flag),
            "label": 0,
        })

    return pd.DataFrame(rows)


def generate_anomalous_sessions(n: int = NUM_ANOMALOUS_SAMPLES, seed: int = 43) -> pd.DataFrame:
    """
    Generate synthetic anomalous sessions simulating malicious or faulty behavior.

    Anomaly types simulated:
    - Abnormally long or short charging duration
    - Unusual power consumption spikes
    - Frequent interruptions
    - Inconsistent energy usage (energy vs duration/power mismatch)
    - Repeated suspicious sessions at the same station

    Parameters
    ----------
    n : int
        Number of anomalous sessions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with same schema as normal sessions; label=1 for all.
    """
    np.random.seed(seed)
    base = datetime(2024, 1, 1, 0, 0, 0)

    # Pre-choose some stations for "repeated suspicious" pattern
    suspicious_stations = [f"STATION_{i:03d}" for i in np.random.choice(range(1, NUM_STATIONS + 1), size=5, replace=False)]
    suspicious_session_count = {s: 0 for s in suspicious_stations}

    rows = []
    anomaly_types = [
        "extreme_duration",
        "power_spike",
        "frequent_interruptions",
        "inconsistent_energy",
        "repeated_station",
    ]

    for i in range(n):
        atype = anomaly_types[i % len(anomaly_types)]

        if atype == "extreme_duration":
            # Abnormally long or very short
            if np.random.random() < 0.5:
                duration = np.random.uniform(180, 400)  # 3–6+ hours
            else:
                duration = np.random.uniform(1, 5)  # < 5 minutes
            power = np.clip(np.random.normal(NORMAL_POWER_MEAN, NORMAL_POWER_STD), 3.0, 22.0)
            energy = (duration / 60) * power * np.random.uniform(0.9, 1.1)
            interruptions = np.random.poisson(0.5)
            interruptions = min(interruptions, 3)

        elif atype == "power_spike":
            duration = np.random.uniform(NORMAL_DURATION_MIN, NORMAL_DURATION_MAX)
            power = np.random.uniform(50, 120)  # Unrealistic spike (e.g. DC fast or fault)
            energy = (duration / 60) * power * np.random.uniform(0.7, 1.3)
            interruptions = np.random.poisson(0.3)
            interruptions = min(interruptions, 2)

        elif atype == "frequent_interruptions":
            duration = np.random.uniform(NORMAL_DURATION_MIN, NORMAL_DURATION_MAX)
            power = np.clip(np.random.normal(NORMAL_POWER_MEAN, NORMAL_POWER_STD), 3.0, 22.0)
            energy = (duration / 60) * power * np.random.uniform(0.85, 1.15)
            interruptions = int(np.random.uniform(8, 18))
            interruptions = min(interruptions, 25)

        elif atype == "inconsistent_energy":
            duration = np.random.uniform(NORMAL_DURATION_MIN, NORMAL_DURATION_MAX)
            power = np.clip(np.random.normal(NORMAL_POWER_MEAN, NORMAL_POWER_STD), 3.0, 22.0)
            # Energy inconsistent with duration * power (e.g. tampering / meter fault)
            energy = (duration / 60) * power * np.random.uniform(0.15, 0.4)
            if np.random.random() < 0.5:
                energy = (duration / 60) * power * np.random.uniform(2.5, 4.0)
            energy = np.clip(energy, 0.5, 200.0)
            interruptions = np.random.poisson(0.5)
            interruptions = min(interruptions, 4)

        else:  # repeated_station
            station = np.random.choice(suspicious_stations)
            suspicious_session_count[station] += 1
            duration = np.random.uniform(NORMAL_DURATION_MIN, NORMAL_DURATION_MAX)
            power = np.clip(np.random.normal(NORMAL_POWER_MEAN, NORMAL_POWER_STD), 3.0, 22.0)
            energy = (duration / 60) * power * np.random.uniform(0.85, 1.15)
            interruptions = np.random.poisson(0.5)
            interruptions = min(interruptions, 4)

        if atype == "repeated_station":
            station_id = station
        else:
            station_id = _random_station_id()

        error_flag = 1 if np.random.random() < 0.15 else 0  # Anomalies often have errors

        rows.append({
            "session_id": _generate_session_id(),
            "station_id": station_id,
            "start_time": _random_start_time(base),
            "charging_duration": round(float(duration), 2),
            "average_power_kw": round(float(power), 2),
            "total_energy_kwh": round(float(energy), 2),
            "number_of_interruptions": int(interruptions),
            "error_flag": int(error_flag),
            "label": 1,
        })

    return pd.DataFrame(rows)


def generate_all_sessions(
    n_normal: int = NUM_NORMAL_SAMPLES,
    n_anomalous: int = NUM_ANOMALOUS_SAMPLES,
    seed_normal: int = 42,
    seed_anomalous: int = 43,
) -> pd.DataFrame:
    """
    Generate combined normal and anomalous sessions and shuffle.

    Returns
    -------
    pd.DataFrame
        Combined dataset with label 0 (normal) and 1 (anomalous).
    """
    df_normal = generate_normal_sessions(n=n_normal, seed=seed_normal)
    df_anomalous = generate_anomalous_sessions(n=n_anomalous, seed=seed_anomalous)
    combined = pd.concat([df_normal, df_anomalous], ignore_index=True)
    combined = combined.sample(frac=1, random_state=99).reset_index(drop=True)
    return combined


def save_sessions(df: pd.DataFrame, filepath: str | Path) -> None:
    """Save session DataFrame to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_sessions(filepath: str | Path) -> pd.DataFrame:
    """Load session DataFrame from CSV. Parse start_time as datetime."""
    df = pd.read_csv(filepath)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


if __name__ == "__main__":
    # Generate and save default dataset
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "generated_sessions.csv"
    df = generate_all_sessions()
    save_sessions(df, data_path)
    print(f"Generated {len(df)} sessions: { (df['label']==0).sum() } normal, { (df['label']==1).sum() } anomalous.")
    print(f"Saved to {data_path}")
