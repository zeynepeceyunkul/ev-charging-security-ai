"""
Data Ingestion Layer: Synthetic EV Charging Telemetry Simulator.

Generates time-series telemetry at 1-minute intervals for simulated charging
sessions. Supports normal and anomalous scenarios. This is a standalone
academic project; no real hardware or protocols are used.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_NORMAL_SESSIONS = 50
NUM_ANOMALOUS_SESSIONS = 20
NUM_STATIONS = 15
VOLTAGE_NOMINAL = 400  # V (AC)
VOLTAGE_TOLERANCE = 0.1
CURRENT_MAX_NORMAL = 32  # A
POWER_MAX_NORMAL = 22  # kW (AC Level 2)
ANOMALY_TYPES = [
    "power_spike",
    "meter_manipulation",
    "session_flooding",
    "interrupted_charging",
    "unrealistic_energy_growth",
    "flatline_meter",
]


def _session_id() -> str:
    return str(uuid.uuid4())[:12]


def _station_id() -> str:
    return f"STATION_{np.random.randint(1, NUM_STATIONS + 1):02d}"


def _normal_session_duration_minutes() -> int:
    """Typical session length in minutes."""
    return int(np.clip(np.random.lognormal(3.5, 0.8), 5, 90))


def generate_normal_session(
    session_idx: int,
    base_ts: datetime,
    seed: int,
) -> pd.DataFrame:
    """
    Generate one normal charging session as time series at 1-min intervals.
    Voltage, current, power, and energy follow realistic AC charging behavior.
    """
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = _normal_session_duration_minutes()

    rows = []
    energy_cum = 0.0
    meter_cum = 0.0

    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        # Slight voltage fluctuation around nominal
        voltage = VOLTAGE_NOMINAL * (1 + np.random.uniform(-0.02, 0.02))
        current = np.clip(np.random.normal(16, 4), 0, CURRENT_MAX_NORMAL)
        power_kw = (voltage / 1000) * current * (0.95 + np.random.uniform(0, 0.05))
        power_kw = np.clip(power_kw, 0, POWER_MAX_NORMAL)
        # Energy increment this minute
        energy_inc = power_kw * (1 / 60)
        energy_cum += energy_inc
        meter_inc = energy_inc * (1 + np.random.uniform(-0.01, 0.01))
        meter_cum += meter_inc
        interruption_flag = 1 if t > 0 and np.random.random() < 0.02 else 0
        error_code = int(np.random.choice([0, 0, 0, 0, 1], p=[0.8, 0.1, 0.05, 0.04, 0.01]))

        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": round(power_kw, 4),
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(meter_inc, 4),
            "interruption_flag": interruption_flag,
            "error_code": error_code,
            "anomaly_label": 0,
            "anomaly_type": "normal",
        })

    return pd.DataFrame(rows)


def generate_anomaly_power_spike(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Simulate sudden power spike (e.g. fault or attack)."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = min(20, _normal_session_duration_minutes())
    spike_minute = duration // 2

    rows = []
    energy_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL * (1 + np.random.uniform(-0.02, 0.02))
        if t == spike_minute:
            current = np.random.uniform(80, 120)
            power_kw = np.clip((voltage / 1000) * current, 50, 150)
        else:
            current = np.clip(np.random.normal(16, 4), 0, CURRENT_MAX_NORMAL)
            power_kw = (voltage / 1000) * current * 0.97
        power_kw = round(np.clip(power_kw, 0, 200), 4)
        energy_inc = power_kw * (1 / 60)
        energy_cum += energy_inc
        meter_inc = energy_inc * (1 + np.random.uniform(-0.01, 0.01))
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(meter_inc, 4),
            "interruption_flag": 0,
            "error_code": 1 if t == spike_minute else 0,
            "anomaly_label": 1,
            "anomaly_type": "power_spike",
        })
    return pd.DataFrame(rows)


def generate_anomaly_meter_manipulation(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Meter reports do not match physical power/energy."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = _normal_session_duration_minutes()

    rows = []
    energy_cum = 0.0
    meter_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL * (1 + np.random.uniform(-0.02, 0.02))
        current = np.clip(np.random.normal(16, 4), 0, CURRENT_MAX_NORMAL)
        power_kw = (voltage / 1000) * current * 0.97
        power_kw = round(np.clip(power_kw, 0, POWER_MAX_NORMAL), 4)
        energy_inc_real = power_kw * (1 / 60)
        energy_cum += energy_inc_real
        # Manipulated: meter reports much less
        meter_inc = energy_inc_real * np.random.uniform(0.3, 0.5)
        meter_cum += meter_inc
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(meter_inc, 4),
            "interruption_flag": 0,
            "error_code": 0,
            "anomaly_label": 1,
            "anomaly_type": "meter_manipulation",
        })
    return pd.DataFrame(rows)


def generate_anomaly_session_flooding(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Very short repeated sessions from same station (flooding)."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = min(5, 2 + session_idx % 4)
    for _ in range(3):
        duration = min(duration + 1, 8)
    rows = []
    energy_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL
        current = np.random.uniform(20, CURRENT_MAX_NORMAL)
        power_kw = round((voltage / 1000) * current * 0.97, 4)
        energy_inc = power_kw * (1 / 60)
        energy_cum += energy_inc
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(energy_inc, 4),
            "interruption_flag": 1 if t in (1, 3) else 0,
            "error_code": 0,
            "anomaly_label": 1,
            "anomaly_type": "session_flooding",
        })
    return pd.DataFrame(rows)


def generate_anomaly_interrupted_charging(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Excessive interruptions."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = 25

    rows = []
    energy_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL * (1 + np.random.uniform(-0.02, 0.02))
        current = np.clip(np.random.normal(16, 4), 0, CURRENT_MAX_NORMAL)
        power_kw = (voltage / 1000) * current * 0.97
        power_kw = round(np.clip(power_kw, 0, POWER_MAX_NORMAL), 4)
        energy_inc = power_kw * (1 / 60) if t % 3 != 1 else 0
        energy_cum += energy_inc
        interruption_flag = 1 if t % 2 == 1 or t % 3 == 0 else 0
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(energy_inc, 4),
            "interruption_flag": interruption_flag,
            "error_code": 1 if interruption_flag else 0,
            "anomaly_label": 1,
            "anomaly_type": "interrupted_charging",
        })
    return pd.DataFrame(rows)


def generate_anomaly_unrealistic_energy_growth(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Energy increases faster than power * time allows."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = 20

    rows = []
    energy_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL
        current = np.random.uniform(10, 20)
        power_kw = (voltage / 1000) * current
        power_kw = round(np.clip(power_kw, 0, POWER_MAX_NORMAL), 4)
        energy_inc = power_kw * (1 / 60) * np.random.uniform(1.8, 2.5)
        energy_cum += energy_inc
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(energy_inc, 4),
            "interruption_flag": 0,
            "error_code": 0,
            "anomaly_label": 1,
            "anomaly_type": "unrealistic_energy_growth",
        })
    return pd.DataFrame(rows)


def generate_anomaly_flatline_meter(session_idx: int, base_ts: datetime, seed: int) -> pd.DataFrame:
    """Meter value does not change despite power draw."""
    np.random.seed(seed)
    station_id = _station_id()
    session_id = _session_id()
    duration = 25

    rows = []
    energy_cum = 0.0
    for t in range(duration):
        ts = base_ts + timedelta(minutes=t)
        voltage = VOLTAGE_NOMINAL * (1 + np.random.uniform(-0.02, 0.02))
        current = np.clip(np.random.normal(16, 4), 0, CURRENT_MAX_NORMAL)
        power_kw = (voltage / 1000) * current * 0.97
        power_kw = round(np.clip(power_kw, 0, POWER_MAX_NORMAL), 4)
        energy_inc_real = power_kw * (1 / 60)
        energy_cum += energy_inc_real
        meter_inc = 0.0 if t > 5 else energy_inc_real
        rows.append({
            "station_id": station_id,
            "session_id": session_id,
            "timestamp": ts,
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power_kw": power_kw,
            "energy_kwh": round(energy_cum, 4),
            "meter_value_increment": round(meter_inc, 4),
            "interruption_flag": 0,
            "error_code": 0,
            "anomaly_label": 1,
            "anomaly_type": "flatline_meter",
        })
    return pd.DataFrame(rows)


ANOMALY_GENERATORS = {
    "power_spike": generate_anomaly_power_spike,
    "meter_manipulation": generate_anomaly_meter_manipulation,
    "session_flooding": generate_anomaly_session_flooding,
    "interrupted_charging": generate_anomaly_interrupted_charging,
    "unrealistic_energy_growth": generate_anomaly_unrealistic_energy_growth,
    "flatline_meter": generate_anomaly_flatline_meter,
}


def generate_all_telemetry(
    n_normal: int = NUM_NORMAL_SESSIONS,
    n_anomalous: int = NUM_ANOMALOUS_SESSIONS,
    base_ts: datetime | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate normal and anomalous sessions, concatenate and return one DataFrame."""
    if base_ts is None:
        base_ts = datetime(2024, 6, 1, 0, 0, 0)

    np.random.seed(seed)
    all_dfs = []

    for i in range(n_normal):
        ts = base_ts + timedelta(days=i % 30, hours=(i * 2) % 24, minutes=(i * 7) % 60)
        df = generate_normal_session(i, ts, seed + 1000 + i)
        all_dfs.append(df)

    anomaly_type_list = list(ANOMALY_GENERATORS.keys())
    for i in range(n_anomalous):
        atype = anomaly_type_list[i % len(anomaly_type_list)]
        ts = base_ts + timedelta(days=10 + i % 20, hours=(i * 3) % 24, minutes=(i * 11) % 60)
        df = ANOMALY_GENERATORS[atype](i, ts, seed + 5000 + i)
        all_dfs.append(df)

    out = pd.concat(all_dfs, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def save_telemetry(df: pd.DataFrame, path: str | Path) -> None:
    """Save telemetry to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_telemetry(path: str | Path) -> pd.DataFrame:
    """Load telemetry from CSV."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df = generate_all_telemetry()
    save_telemetry(df, root / "data" / "simulated_telemetry.csv")
    print(f"Sessions: {df['session_id'].nunique()}, Rows: {len(df)}")
    print(df.groupby(["anomaly_label", "anomaly_type"]).size())
