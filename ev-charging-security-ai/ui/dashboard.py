"""
SOC Analyst Console — Lightweight Streamlit dashboard for the hybrid
EV charging security monitoring system. Displays system overview, incident table,
time series visualization, and checklist risk. Research prototype; no authentication.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Project root (parent of ui/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


def load_alerts():
    path = RESULTS_DIR / "alerts.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_summary():
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_telemetry():
    path = DATA_DIR / "simulated_telemetry.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.set_page_config(page_title="EV Charging Security — SOC Console", layout="wide")

st.title("EV Charging Security — SOC Analyst Console")
st.caption("Hybrid Rule-Based and AI-Driven Security Monitoring (Recall-Aware)")

# Security disclaimer
st.info(
    "This system prioritizes recall over false positives, as required for "
    "critical infrastructure cybersecurity monitoring."
)

# --- A. System overview ---
st.header("A. System Overview")

alerts = load_alerts()
summary = load_summary()

if summary:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total charging sessions", summary.get("total_sessions", len(alerts)))
    with col2:
        st.metric("Anomalies detected", summary.get("anomaly_count", "—"))
    with col3:
        sev = summary.get("severity_counts", {})
        st.metric("Critical", sev.get("CRITICAL", 0))
    with col4:
        st.metric("High", sev.get("HIGH", 0))
    with col5:
        st.metric("Medium / Low", f"{sev.get('MEDIUM', 0)} / {sev.get('LOW', 0)}")
    st.metric("Average risk score", f"{summary.get('avg_risk_score', 0):.1f}")
else:
    df_a = pd.DataFrame(alerts)
    if len(df_a) > 0:
        st.metric("Total sessions (alerts)", len(df_a))
        st.metric("By severity", df_a["severity"].value_counts().to_string())
    else:
        st.write("No alerts loaded. Run the pipeline first: `python experiments/run_pipeline.py`")

# --- B. Incident table ---
st.header("B. Incident Table")

if alerts:
    df = pd.DataFrame(alerts)
    cols_display = ["station_id", "anomaly_type", "severity", "risk_score", "confidence", "recommended_action"]
    df_display = df[[c for c in cols_display if c in df.columns]]
    st.dataframe(df_display, use_container_width=True)
else:
    st.write("No alerts to display.")

# --- C. Time series visualization ---
st.header("C. Time Series by Station")

telemetry = load_telemetry()
if telemetry is not None:
    stations = sorted(telemetry["station_id"].unique().tolist())
    station_id = st.selectbox("Select station_id", stations)
    if station_id:
        subset = telemetry[telemetry["station_id"] == station_id].copy()
        subset = subset.sort_values("timestamp")
        # Sessions at this station that are anomalous (from alerts)
        high_severity = {"MEDIUM", "HIGH", "CRITICAL"}
        anomalous_sessions = set()
        for a in alerts:
            if a.get("station_id") == station_id and a.get("severity") in high_severity:
                anomalous_sessions.add(a.get("session_id", ""))
        subset["anomalous"] = subset["session_id"].isin(anomalous_sessions)
        subset["timestamp"] = pd.to_datetime(subset["timestamp"])

        chart_df = subset.set_index("timestamp")[["power_kw", "energy_kwh"]]
        st.line_chart(chart_df)
        if subset["anomalous"].any():
            anom_sessions = subset[subset["anomalous"]]["session_id"].unique().tolist()
            st.write("**Anomalous timestamps** (MEDIUM/HIGH/CRITICAL) at this station — session_ids:", anom_sessions)
            anom_ranges = subset[subset["anomalous"]].groupby("session_id").agg(
                start=("timestamp", "min"),
                end=("timestamp", "max"),
            ).reset_index()
            st.dataframe(anom_ranges, use_container_width=True, hide_index=True)
else:
    st.write("Telemetry not found. Run the pipeline to generate data/simulated_telemetry.csv.")

# --- D. Checklist risk summary ---
st.header("D. Checklist Risk Summary")

if summary and "checklist_risk" in summary:
    cr = summary["checklist_risk"]
    st.metric("Checklist risk score (0–100)", f"{cr:.1f}")
    if cr < 30:
        st.success("Low checklist risk (green: < 30).")
    elif cr <= 60:
        st.warning("Medium checklist risk (yellow: 30–60).")
    else:
        st.error("High checklist risk (red: > 60).")
else:
    st.write("Checklist risk not available. Run the pipeline to generate results/summary.json.")

# Footer
st.divider()
st.caption("Research prototype. Simulation only; no real infrastructure.")
