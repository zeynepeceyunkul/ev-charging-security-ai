"""
Full pipeline: Data Ingestion -> Rule Engine -> ML Engine -> Correlation ->
Checklist Risk -> Alert Engine. Writes results/alerts.json and results/evaluation.txt.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd
import numpy as np

from data_simulator import generate_all_telemetry, save_telemetry, load_telemetry
from rule_engine import (
    evaluate_rules_on_telemetry,
    get_session_rule_summary,
    get_stations_with_frequent_restarts,
)
from ml_engine import run_ml_engine, prepare_ml_features
from correlation_engine import correlate_sessions
from checklist_engine import (
    get_default_checklist,
    evaluate_checklist,
    runtime_risk_from_severity_and_confidence,
    combined_risk,
)
from alert_engine import alerts_from_correlation_and_risk, save_alerts


def ensure_telemetry(project_root: Path) -> pd.DataFrame:
    """Generate or load simulated telemetry."""
    data_path = project_root / "data" / "simulated_telemetry.csv"
    if not data_path.exists():
        df = generate_all_telemetry(
            n_normal=50,
            n_anomalous=20,
            seed=42,
        )
        save_telemetry(df, data_path)
    else:
        df = load_telemetry(data_path)
    return df


def run_pipeline(project_root: Path) -> dict:
    """Execute the full hybrid monitoring pipeline. Returns all intermediates and results."""
    # 1. Data ingestion
    telemetry = ensure_telemetry(project_root)
    normal_mask = (telemetry["anomaly_label"] == 0).values

    # 2. Rule-based detection
    telemetry_with_rules = evaluate_rules_on_telemetry(telemetry)
    session_rule_summary = get_session_rule_summary(telemetry_with_rules)
    # R08: frequent session restarts (evaluate at pipeline level)
    flooding_stations = get_stations_with_frequent_restarts(telemetry_with_rules)
    mask = session_rule_summary["station_id"].isin(flooding_stations)
    session_rule_summary.loc[mask, "rule_triggered_count"] += 1
    session_rule_summary.loc[mask, "rule_max_severity"] = np.maximum(
        session_rule_summary.loc[mask, "rule_max_severity"].values, 0.7
    )
    session_rule_summary.loc[mask, "rule_ids_triggered"] = session_rule_summary.loc[
        mask, "rule_ids_triggered"
    ].apply(lambda x: (x + ",R08_frequent_session_restarts") if x else "R08_frequent_session_restarts")

    # 3. ML anomaly detection (train on normal only)
    ml_scores, session_ml_summary, model, scaler = run_ml_engine(telemetry, normal_mask)
    telemetry_with_rules["ml_anomaly_score"] = ml_scores
    normal_scores = ml_scores[normal_mask]

    # 4. Correlation & decision (recall-aware: adaptive ML threshold, temporal window)
    session_starts = telemetry.groupby("session_id", as_index=False).agg(
        station_id=("station_id", "first"),
        start_ts=("timestamp", "min"),
    )
    all_ml_scores = session_ml_summary["ml_score_max"].values
    correlation_df = correlate_sessions(
        session_rule_summary,
        session_ml_summary,
        normal_scores,
        all_ml_scores,
        session_starts=session_starts,
    )

    # 5. Checklist risk
    checklist_df, checklist_risk_value = evaluate_checklist(
        get_default_checklist(),
        status_by_id=None,
        random_pass_rate=0.72,
        seed=42,
    )
    runtime_risks = {}
    combined_risks = {}
    for _, row in correlation_df.iterrows():
        sid = row["session_id"]
        rt = runtime_risk_from_severity_and_confidence(
            row["final_severity"],
            row["confidence_score"],
        )
        runtime_risks[sid] = rt
        combined_risks[sid] = combined_risk(rt, checklist_risk_value)
    correlation_df["runtime_risk"] = correlation_df["session_id"].map(runtime_risks)
    correlation_df["combined_risk"] = correlation_df["session_id"].map(combined_risks)

    # 6. Alerts
    alerts = alerts_from_correlation_and_risk(
        correlation_df,
        runtime_risks,
        combined_risks,
    )
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_alerts(alerts, results_dir / "alerts.json")

    # Evaluation metrics: recall is prioritized for cybersecurity (missed attacks > false positives)
    correlation_df["is_anomaly_true"] = correlation_df["anomaly_label"] == 1
    # Positive = MEDIUM or higher (recall-aware: we treat MEDIUM as actionable)
    correlation_df["is_alerted_positive"] = correlation_df["final_severity"].isin(["MEDIUM", "HIGH", "CRITICAL"])
    tp = ((correlation_df["is_anomaly_true"]) & (correlation_df["is_alerted_positive"])).sum()
    fp = ((~correlation_df["is_anomaly_true"]) & (correlation_df["is_alerted_positive"])).sum()
    fn = ((correlation_df["is_anomaly_true"]) & (~correlation_df["is_alerted_positive"])).sum()
    tn = ((~correlation_df["is_anomaly_true"]) & (~correlation_df["is_alerted_positive"])).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(correlation_df) if len(correlation_df) > 0 else 0

    # How many anomalies were detected ONLY by ML (no rules fired)
    anomalies_detected_ml_only = (
        (correlation_df["is_anomaly_true"]) & (correlation_df["detected_ml_only"])
    ).sum()
    total_anomalies = correlation_df["is_anomaly_true"].sum()

    eval_lines = [
        "=" * 60,
        "Hybrid Rule-Based and AI-Driven Security Monitoring",
        "Evaluation Report (Recall-Aware, Standalone Academic Project)",
        "=" * 60,
        "",
        "Pipeline: Data -> Rule Engine -> ML Engine -> Correlation -> Checklist -> Alerts",
        "Positive = MEDIUM / HIGH / CRITICAL severity (recall-focused).",
        "",
        f"Sessions: {len(correlation_df)} total",
        f"Normal: {(correlation_df['anomaly_label']==0).sum()}, Anomalous: {(correlation_df['anomaly_label']==1).sum()}",
        "",
        "Detection (MEDIUM+ severity as positive):",
        f"  True Positives:  {tp}",
        f"  False Positives: {fp}",
        f"  False Negatives: {fn}",
        f"  True Negatives:  {tn}",
        f"  Accuracy:  {accuracy:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall:    {recall:.4f}  (primary metric for SOC)",
        f"  F1-score:  {f1:.4f}",
        "",
        "Recall-focused evaluation:",
        f"  Anomalies detected ONLY by ML (no rules fired): {anomalies_detected_ml_only} / {total_anomalies}",
        "",
        f"Checklist risk (0-100): {checklist_risk_value:.1f}",
        f"Final risk formula: 0.6 * runtime_risk + 0.4 * checklist_risk",
        "",
        "Severity distribution:",
    ]
    for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        count = (correlation_df["final_severity"] == sev).sum()
        eval_lines.append(f"  {sev}: {count}")
    eval_lines.append("")
    eval_lines.append("=" * 60)

    with open(results_dir / "evaluation.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(eval_lines))

    # Summary for dashboard and downstream
    severity_counts = {sev: int((correlation_df["final_severity"] == sev).sum()) for sev in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
    avg_risk = float(correlation_df["combined_risk"].mean()) if "combined_risk" in correlation_df.columns else 0
    summary = {
        "total_sessions": len(correlation_df),
        "anomaly_count": int(correlation_df["is_anomaly_true"].sum()),
        "severity_counts": severity_counts,
        "checklist_risk": float(checklist_risk_value),
        "avg_risk_score": avg_risk,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "anomalies_detected_ml_only": int(anomalies_detected_ml_only),
    }
    import json
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "telemetry": telemetry_with_rules,
        "session_rule_summary": session_rule_summary,
        "session_ml_summary": session_ml_summary,
        "correlation_df": correlation_df,
        "checklist_df": checklist_df,
        "checklist_risk": checklist_risk_value,
        "alerts": alerts,
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "anomalies_detected_ml_only": anomalies_detected_ml_only,
    }


def main():
    print("Running Hybrid Security Monitoring Pipeline (Recall-Aware)")
    print("Project root:", PROJECT_ROOT)
    # Recall is prioritized: in critical infrastructure, missed attacks are
    # more costly than false positives; the correlation engine assigns at least
    # MEDIUM when ML score exceeds adaptive threshold even if no rules fire.
    results = run_pipeline(PROJECT_ROOT)
    print("\nPipeline complete.")
    print(f"Alerts written: {len(results['alerts'])}")
    print(f"Evaluation written: results/evaluation.txt")
    print("\n--- Recall-focused evaluation ---")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}  (primary for SOC)")
    print(f"F1-score:  {results['eval_f1']:.4f}")
    print(f"Anomalies detected ONLY by ML: {results['anomalies_detected_ml_only']} / {results['correlation_df']['anomaly_label'].sum()}")


if __name__ == "__main__":
    main()
