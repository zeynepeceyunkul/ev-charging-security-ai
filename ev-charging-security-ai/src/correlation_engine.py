"""
Correlation & Decision Engine (Recall-Aware).

Combines rule-based results, ML anomaly score, and anomaly type (if known)
to produce final_severity, confidence_score, and anomaly_summary per session.

Design principles for cybersecurity monitoring:
- ML-only path: if ML score exceeds adaptive threshold, assign at least MEDIUM
  severity so that missed attacks are reduced (recall over precision).
- Adaptive percentile-based ML threshold replaces static cutoffs.
- Temporal correlation: sliding window per station escalates severity when
  multiple anomalies occur at the same station within a short window.
- Confidence combines normalized ML score, rule count, and temporal bonus.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

SEVERITY_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Percentile on (all) ML scores to define "above threshold" for ML-only path
ML_THRESHOLD_PERCENTILE = 85
# Sliding window: minutes
TEMPORAL_WINDOW_MINUTES = 10
# Minimum number of anomalies from same station in window to escalate
TEMPORAL_ESCALATION_COUNT = 3
# Confidence: temporal correlation bonus (max added when escalation applies)
TEMPORAL_CONFIDENCE_BONUS = 0.15


def compute_adaptive_ml_threshold(ml_scores: np.ndarray, percentile: float = ML_THRESHOLD_PERCENTILE) -> float:
    """
    Adaptive ML threshold from percentile of scores.
    Sessions with ml_score_max above this are considered ML-anomalous.
    """
    if ml_scores.size == 0:
        return 0.0
    return float(np.percentile(ml_scores, percentile))


def _normalize_ml_score(score: float, normal_scores: np.ndarray) -> float:
    """Map ML score to 0-1 scale using normal distribution percentiles."""
    if normal_scores.size == 0:
        return 0.0
    p90 = np.percentile(normal_scores, 90)
    p99 = np.percentile(normal_scores, 99)
    if score <= p90:
        return 0.0
    if score >= p99:
        return 1.0
    return float((score - p90) / max(p99 - p90, 1e-9))


def temporal_correlation_per_station(
    session_starts: pd.DataFrame,
    window_minutes: int = TEMPORAL_WINDOW_MINUTES,
    escalation_count: int = TEMPORAL_ESCALATION_COUNT,
) -> pd.Series:
    """
    For each session, count how many other sessions at the same station
    started within the last window_minutes. Returns Series index by session_id:
    count_in_window (including self). Sessions with count >= escalation_count
    get temporal_escalation = True (used later to escalate severity).
    """
    # session_starts: columns session_id, station_id, start_ts
    session_starts = session_starts.copy()
    session_starts["start_ts"] = pd.to_datetime(session_starts["start_ts"])
    session_starts = session_starts.sort_values("start_ts")
    out = {}
    for station_id, grp in session_starts.groupby("station_id"):
        grp = grp.sort_values("start_ts").reset_index(drop=True)
        for i, row in grp.iterrows():
            sid = row["session_id"]
            t0 = row["start_ts"]
            window_start = t0 - pd.Timedelta(minutes=window_minutes)
            # Count sessions at this station that started in [window_start, t0]
            count = ((grp["start_ts"] >= window_start) & (grp["start_ts"] <= t0)).sum()
            out[sid] = int(count)
    return pd.Series(out)


def decide_severity(
    rule_triggered_count: int,
    rule_max_severity: float,
    ml_score_normalized: float,
    ml_above_threshold: bool,
    anomaly_type: str,
    temporal_escalation: bool,
) -> Tuple[str, float, str]:
    """
    Recall-aware decision logic:
    - rule_count >= 2 AND ml_score high -> CRITICAL
    - rule_count == 1 OR (ml_score very high) -> HIGH
    - ml_score > threshold only (no rules) -> MEDIUM  (ML-only path: at least MEDIUM)
    - else -> LOW

    Reduces rule dominance so that ML-only anomalies are not missed.
    Returns (final_severity, confidence_score, anomaly_summary).
    """
    summary_parts = []
    rule_contrib = rule_triggered_count > 0
    ml_high = ml_score_normalized > 0.5
    ml_very_high = ml_score_normalized > 0.75

    if rule_contrib:
        summary_parts.append(f"{rule_triggered_count} rule(s) triggered (max weight={rule_max_severity:.2f})")
    if ml_above_threshold or ml_score_normalized > 0:
        summary_parts.append(f"ML anomaly score elevated ({ml_score_normalized:.2f})")
    if anomaly_type and anomaly_type != "normal":
        summary_parts.append(f"type={anomaly_type}")
    if temporal_escalation:
        summary_parts.append("temporal correlation: multiple anomalies at same station in window")

    # Base confidence from ML and rules
    confidence = 0.2
    confidence += 0.25 * min(ml_score_normalized, 1.0)
    confidence += 0.15 * min(rule_triggered_count, 3) * 0.4
    confidence += rule_max_severity * 0.2
    if temporal_escalation:
        confidence += TEMPORAL_CONFIDENCE_BONUS
    confidence = min(1.0, confidence)

    # Severity: recall-aware; ML-only path guarantees at least MEDIUM
    if rule_triggered_count >= 2 and ml_high:
        severity = "CRITICAL"
        confidence = min(0.98, confidence + 0.1)
    elif rule_triggered_count >= 2:
        severity = "HIGH"
    elif rule_triggered_count >= 1 and ml_high:
        severity = "CRITICAL"
        confidence = min(0.95, confidence + 0.05)
    elif rule_triggered_count >= 1:
        severity = "HIGH"
    elif ml_very_high:
        severity = "HIGH"
        confidence = min(0.85, confidence + 0.05)
    elif ml_above_threshold:
        # ML-only path: no rules fired but ML exceeds threshold -> at least MEDIUM
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # Temporal escalation: bump one level if not already CRITICAL
    if temporal_escalation and severity == "LOW":
        severity = "MEDIUM"
        confidence = min(0.75, confidence + 0.1)
    elif temporal_escalation and severity == "MEDIUM":
        severity = "HIGH"
        confidence = min(0.9, confidence + 0.05)
    elif temporal_escalation and severity == "HIGH":
        severity = "CRITICAL"
        confidence = min(0.95, confidence + 0.05)

    anomaly_summary = "; ".join(summary_parts) if summary_parts else "No indicators"
    return severity, float(np.clip(confidence, 0, 1)), anomaly_summary


def correlate_sessions(
    session_rule_summary: pd.DataFrame,
    session_ml_summary: pd.DataFrame,
    normal_ml_scores: np.ndarray,
    all_ml_scores: np.ndarray,
    session_starts: Optional[pd.DataFrame] = None,
    ml_threshold_percentile: float = ML_THRESHOLD_PERCENTILE,
) -> pd.DataFrame:
    """
    Join rule summary and ML summary per session; apply adaptive ML threshold,
    temporal correlation (sliding window), and recall-aware decision logic.
    Returns one row per session with final_severity, confidence_score,
    anomaly_summary, and detected_ml_only (True if severity >= MEDIUM with no rules).
    """
    ml_threshold = compute_adaptive_ml_threshold(all_ml_scores, ml_threshold_percentile)

    merged = session_rule_summary.merge(
        session_ml_summary,
        on="session_id",
        how="left",
    )
    merged["ml_score_max"] = merged["ml_score_max"].fillna(0)
    merged["ml_score_normalized"] = merged["ml_score_max"].apply(
        lambda s: _normalize_ml_score(s, normal_ml_scores)
    )
    merged["ml_above_threshold"] = merged["ml_score_max"] >= ml_threshold

    # Temporal correlation: sessions per station in sliding window
    if session_starts is not None and len(session_starts) > 0:
        temporal_count = temporal_correlation_per_station(
            session_starts,
            window_minutes=TEMPORAL_WINDOW_MINUTES,
            escalation_count=TEMPORAL_ESCALATION_COUNT,
        )
        merged["temporal_count_in_window"] = merged["session_id"].map(temporal_count).fillna(0)
        merged["temporal_escalation"] = merged["temporal_count_in_window"] >= TEMPORAL_ESCALATION_COUNT
    else:
        merged["temporal_escalation"] = False

    results = []
    for _, row in merged.iterrows():
        sev, conf, summary = decide_severity(
            int(row["rule_triggered_count"]),
            float(row["rule_max_severity"]),
            float(row["ml_score_normalized"]),
            bool(row["ml_above_threshold"]),
            str(row.get("anomaly_type", "normal")),
            bool(row["temporal_escalation"]),
        )
        # Detected only by ML: no rules fired but we assigned MEDIUM or above
        detected_ml_only = (int(row["rule_triggered_count"]) == 0) and (sev in ("MEDIUM", "HIGH", "CRITICAL"))
        results.append({
            "session_id": row["session_id"],
            "station_id": row["station_id"],
            "anomaly_label": row["anomaly_label"],
            "anomaly_type": row["anomaly_type"],
            "final_severity": sev,
            "confidence_score": conf,
            "anomaly_summary": summary,
            "detected_ml_only": detected_ml_only,
        })
    return pd.DataFrame(results)
