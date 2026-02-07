"""
Alert & Recommendation Engine.

Produces structured alerts per detected incident with severity, confidence,
risk score, and recommended action. Actions vary by severity.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

# Recommended actions by severity
ACTIONS = {
    "LOW": "Monitor",
    "MEDIUM": "Log and review",
    "HIGH": "Temporarily suspend session",
    "CRITICAL": "Flag station for inspection",
}


def build_alert(
    station_id: str,
    anomaly_type: str,
    severity: str,
    confidence: float,
    risk_score: float,
    anomaly_summary: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """Build one structured alert."""
    return {
        "station_id": station_id,
        "session_id": session_id,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "confidence": round(confidence, 2),
        "risk_score": round(risk_score, 1),
        "recommended_action": ACTIONS.get(severity, "Monitor"),
        "anomaly_summary": anomaly_summary,
    }


def alerts_from_correlation_and_risk(
    correlation_df,
    runtime_risk_series,
    combined_risk_series,
) -> List[Dict[str, Any]]:
    """
    Build list of alerts from correlation output and risk scores.
    correlation_df must have session_id, station_id, anomaly_type, final_severity, confidence_score, anomaly_summary.
    """
    alerts = []
    for idx, row in correlation_df.iterrows():
        sid = row["session_id"]
        runtime = runtime_risk_series.get(sid, 0)
        combined = combined_risk_series.get(sid, runtime)
        alert = build_alert(
            station_id=row["station_id"],
            anomaly_type=row["anomaly_type"],
            severity=row["final_severity"],
            confidence=row["confidence_score"],
            risk_score=combined,
            anomaly_summary=row.get("anomaly_summary", ""),
            session_id=sid,
        )
        alerts.append(alert)
    return alerts


def save_alerts(alerts: List[Dict[str, Any]], path: str | Path) -> None:
    """Write alerts to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)


def load_alerts(path: str | Path) -> List[Dict[str, Any]]:
    """Load alerts from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
