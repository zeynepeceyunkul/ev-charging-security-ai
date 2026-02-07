"""
Rule-Based Intrusion Detection Engine.

Deterministic rules evaluated on telemetry (per row and per session).
Each rule produces rule_id, trigger flag, and severity weight.
Runs before the ML engine in the pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Voltage bounds (V)
VOLTAGE_MIN = 360
VOLTAGE_MAX = 440
# Power physical limit (kW)
POWER_PHYSICAL_MAX = 350
# Session duration bounds (minutes)
SESSION_DURATION_MIN = 1
SESSION_DURATION_MAX = 600
# Interruption threshold per session
MAX_INTERRUPTIONS_NORMAL = 5
# Meter jump threshold (kWh per minute)
METER_JUMP_THRESHOLD = 2.0
# Flat meter: max consecutive same increments
FLAT_METER_CONSECUTIVE = 5
# Session restart: min gap between sessions at same station (minutes)
MIN_SESSION_GAP_MINUTES = 2


def _rule(rule_id: str, triggered: bool, severity_weight: float) -> Dict[str, Any]:
    return {"rule_id": rule_id, "triggered": triggered, "severity_weight": severity_weight}


def rule_voltage_outside_range(row: pd.Series) -> Dict[str, Any]:
    """Voltage outside allowed operational range."""
    v = row.get("voltage", np.nan)
    triggered = not (VOLTAGE_MIN <= v <= VOLTAGE_MAX) if pd.notna(v) else False
    return _rule("R01_voltage_out_of_range", triggered, 0.8)


def rule_power_exceeds_physical_limits(row: pd.Series) -> Dict[str, Any]:
    """Power exceeds physical charger limits."""
    p = row.get("power_kw", np.nan)
    triggered = p > POWER_PHYSICAL_MAX if pd.notna(p) else False
    return _rule("R02_power_exceeds_physical_limits", triggered, 1.0)


def rule_energy_decreases(prev_energy: float, curr_energy: float) -> Dict[str, Any]:
    """Energy decreases between timestamps (impossible without reset)."""
    triggered = prev_energy > curr_energy and prev_energy > 0
    return _rule("R03_energy_decreases", triggered, 0.9)


def rule_excessive_interruptions(interruption_count: int, session_len: int) -> Dict[str, Any]:
    """Too many interruptions in one session."""
    triggered = session_len > 10 and (interruption_count / max(session_len, 1)) > 0.2
    return _rule("R04_excessive_interruptions", triggered, 0.7)


def rule_flat_meter_values(meter_increments: np.ndarray) -> Dict[str, Any]:
    """Meter value does not change over several consecutive minutes."""
    if len(meter_increments) < FLAT_METER_CONSECUTIVE:
        return _rule("R05_flatline_meter", False, 0.75)
    zero_run = 0
    for x in meter_increments:
        if x is not None and float(x) == 0:
            zero_run += 1
            if zero_run >= FLAT_METER_CONSECUTIVE:
                return _rule("R05_flatline_meter", True, 0.75)
        else:
            zero_run = 0
    return _rule("R05_flatline_meter", False, 0.75)


def rule_sudden_meter_jump(prev_meter_cum: float, curr_meter_inc: float) -> Dict[str, Any]:
    """Sudden large jump in meter reading."""
    triggered = curr_meter_inc > METER_JUMP_THRESHOLD if pd.notna(curr_meter_inc) else False
    return _rule("R06_sudden_meter_jump", triggered, 0.85)


def rule_session_duration_violation(session_duration_minutes: int) -> Dict[str, Any]:
    """Session duration outside allowed range."""
    triggered = not (SESSION_DURATION_MIN <= session_duration_minutes <= SESSION_DURATION_MAX)
    return _rule("R07_session_duration_violation", triggered, 0.5)


def rule_too_frequent_session_restarts(
    session_starts: pd.Series,
    station_id: str,
    gap_minutes: int = MIN_SESSION_GAP_MINUTES,
) -> Dict[str, Any]:
    """Same station starts a new session too soon (flooding)."""
    if len(session_starts) < 2:
        return _rule("R08_frequent_session_restarts", False, 0.7)
    ts = pd.to_datetime(session_starts)
    ts = ts.sort_values()
    diffs = ts.diff().dt.total_seconds() / 60
    triggered = (diffs < gap_minutes).any()
    return _rule("R08_frequent_session_restarts", triggered, 0.7)


def rule_repeated_failures_same_station(error_count: int, session_length: int) -> Dict[str, Any]:
    """Repeated error_code non-zero within session (or same station across sessions)."""
    triggered = session_length >= 5 and error_count >= 3
    return _rule("R09_repeated_failures_same_station", triggered, 0.6)


def rule_inconsistent_power_energy(
    power_kw: float,
    energy_increment: float,
    interval_minutes: float = 1.0,
) -> Dict[str, Any]:
    """Energy increment inconsistent with power * time (e.g. meter tampering)."""
    if power_kw is None or power_kw <= 0 or interval_minutes <= 0:
        return _rule("R10_inconsistent_power_energy", False, 0.8)
    expected_inc = power_kw * (interval_minutes / 60)
    if expected_inc <= 0:
        return _rule("R10_inconsistent_power_energy", False, 0.8)
    ratio = energy_increment / expected_inc if energy_increment is not None else 0
    triggered = ratio < 0.3 or ratio > 2.5
    return _rule("R10_inconsistent_power_energy", triggered, 0.8)


def rule_negative_current(row: pd.Series) -> Dict[str, Any]:
    """Negative current reading (invalid)."""
    c = row.get("current", np.nan)
    triggered = c is not None and float(c) < 0
    return _rule("R11_negative_current", triggered, 0.9)


def rule_zero_power_nonzero_energy(row: pd.Series) -> Dict[str, Any]:
    """Zero power but positive energy increment."""
    p = row.get("power_kw", 0)
    inc = row.get("meter_value_increment", 0)
    triggered = (p is not None and float(p) == 0) and (inc is not None and float(inc) > 0.001)
    return _rule("R12_zero_power_positive_energy", triggered, 0.85)


def evaluate_rules_on_session(session_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Evaluate all rules on a single session DataFrame (one session_id).
    Returns list of rule results; multiple rows may contribute to one rule.
    """
    results = []
    session_df = session_df.sort_values("timestamp").reset_index(drop=True)
    n = len(session_df)

    # Per-row rules
    for i, row in session_df.iterrows():
        results.append(rule_voltage_outside_range(row))
        results.append(rule_power_exceeds_physical_limits(row))
        results.append(rule_negative_current(row))
        results.append(rule_zero_power_nonzero_energy(row))
        if i > 0:
            prev = session_df.iloc[i - 1]
            results.append(rule_energy_decreases(
                prev.get("energy_kwh", 0), row.get("energy_kwh", 0)
            ))
            results.append(rule_sudden_meter_jump(
                prev.get("energy_kwh", 0), row.get("meter_value_increment", 0)
            ))
        results.append(rule_inconsistent_power_energy(
            row.get("power_kw"), row.get("meter_value_increment", 0), 1.0
        ))

    # Session-level rules (append once per session)
    interruption_count = int(session_df["interruption_flag"].sum())
    results.append(rule_excessive_interruptions(interruption_count, n))
    results.append(rule_flat_meter_values(session_df["meter_value_increment"].values))
    duration_min = n
    results.append(rule_session_duration_violation(duration_min))
    error_count = int((session_df["error_code"] != 0).sum())
    results.append(rule_repeated_failures_same_station(error_count, n))

    return results


def evaluate_rules_on_telemetry(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate rules for all sessions. Returns telemetry_df with added columns:
    - rule_triggered_count: number of rules that fired in this row's session (so far)
    - rule_max_severity: max severity_weight among triggered rules in session
    - rule_ids_triggered: comma-separated rule_ids triggered in session
    And a session-level summary: one row per session with list of triggered rule_ids and weights.
    """
    telemetry_df = telemetry_df.copy()
    session_ids = telemetry_df["session_id"].unique()

    # Session-level rule results: session_id -> list of triggered rule dicts
    session_rule_triggered: Dict[str, List[Dict]] = {}

    for sid in session_ids:
        sdf = telemetry_df[telemetry_df["session_id"] == sid]
        rule_results = evaluate_rules_on_session(sdf)
        triggered = [r for r in rule_results if r["triggered"]]
        session_rule_triggered[sid] = triggered

    # Per-row: for each row, use the session's triggered rules (we don't recompute per row)
    def row_rule_summary(sid):
        trig = session_rule_triggered.get(sid, [])
        count = len(trig)
        max_sev = max((r["severity_weight"] for r in trig), default=0)
        ids = ",".join(r["rule_id"] for r in trig)
        return count, max_sev, ids

    out = []
    for _, row in telemetry_df.iterrows():
        c, s, ids = row_rule_summary(row["session_id"])
        out.append({"rule_triggered_count": c, "rule_max_severity": s, "rule_ids_triggered": ids})

    summary_df = pd.DataFrame(out)
    for col in summary_df.columns:
        telemetry_df[col] = summary_df[col].values

    return telemetry_df


def get_stations_with_frequent_restarts(
    telemetry_df: pd.DataFrame,
    gap_minutes: int = MIN_SESSION_GAP_MINUTES,
) -> set:
    """Session flooding: stations that have session starts too close in time."""
    session_starts = telemetry_df.groupby("session_id").agg(
        station_id=("station_id", "first"),
        start=("timestamp", "min"),
    ).reset_index()
    triggered_stations = set()
    for sid, grp in session_starts.groupby("station_id"):
        if len(grp) < 2:
            continue
        ts = pd.to_datetime(grp["start"]).sort_values()
        diffs = ts.diff().dt.total_seconds() / 60
        if (diffs < gap_minutes).any():
            triggered_stations.add(sid)
    return triggered_stations


def get_session_rule_summary(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per session: session_id, station_id, anomaly_label, anomaly_type,
    rule_triggered_count, rule_max_severity, rule_ids_triggered.
    """
    cols = ["session_id", "station_id", "anomaly_label", "anomaly_type",
            "rule_triggered_count", "rule_max_severity", "rule_ids_triggered"]
    session_df = telemetry_df.groupby("session_id", as_index=False).agg({
        "station_id": "first",
        "anomaly_label": "first",
        "anomaly_type": "first",
        "rule_triggered_count": "max",
        "rule_max_severity": "max",
        "rule_ids_triggered": "first",
    })
    return session_df[cols]
