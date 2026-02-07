# Hybrid Rule-Based and AI-Driven Security Monitoring System for EV Charging Stations

**Standalone Academic Project** — This repository implements a hybrid cybersecurity monitoring system for simulated EV charging station environments. It combines rule-based intrusion detection, machine learning anomaly detection, risk correlation, a checklist-based risk engine, and structured alerting. The project is **individual and standalone**; it does not reference or reuse any team project, OCPP implementations, or prior systems.

---

## 1. Problem Definition

Electric vehicle charging infrastructure is a critical asset and potential attack surface. Threats include meter manipulation, power anomalies, session flooding, and operational faults that can indicate cyber intrusion or safety risks. This project addresses the need for **proactive detection** of cyber and operational anomalies by combining:

- **Deterministic rules** that encode domain and physical constraints (voltage, power, energy consistency, session behavior).
- **Machine learning** that learns normal behavior and flags deviations.
- **Correlation** of rule triggers and ML scores to assign severity and confidence.
- **Checklist-based risk** (physical and cybersecurity configuration) combined with runtime risk.
- **Structured alerts** with recommended mitigation actions.

The system is designed at an **architectural and analytical level**; all data and environments are **simulated**. No real hardware, charging protocols, or external APIs are used.

---

## 2. Threat Model

The following threat scenarios are considered in the design and in synthetic data generation:

| Threat / anomaly type        | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| Power spike                 | Abnormally high power draw (fault or attack).                               |
| Meter manipulation          | Meter readings inconsistent with physical power/energy (tampering/fraud).  |
| Session flooding            | Many short sessions from same station (DoS or probe).                      |
| Interrupted charging        | Excessive interruptions (instability or attack).                           |
| Unrealistic energy growth   | Energy increases faster than power × time allows.                           |
| Flatline meter             | Meter stops updating despite ongoing draw (tampering/fault).                 |

Additional rule-level checks cover: voltage out of range, power beyond physical limits, energy decreasing between timestamps, sudden meter jumps, session duration violations, repeated failures, and inconsistent power–energy relationship.

---

## 3. System Architecture

The system is implemented as a **modular pipeline** with six logical layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Data Ingestion Layer                                                │
│     Synthetic telemetry (time series, 1-min intervals)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. Rule-Based Detection Engine                                          │
│     Deterministic rules (voltage, power, energy, meter, session, etc.)   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. Machine Learning Anomaly Detection Engine                            │
│     Isolation Forest (trained on normal data only)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. Correlation & Decision Engine                                        │
│     Combines rules + ML score + anomaly type → severity, confidence      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. Risk Scoring Engine (Checklist-Based)                                │
│     Physical + cybersecurity checklist; Final = 0.6*runtime + 0.4*check  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. Alert & Recommendation Engine                                        │
│     Structured alerts with recommended_action by severity                 │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Data ingestion:** `src/data_simulator.py` — generates or loads `data/simulated_telemetry.csv`.
- **Rule engine:** `src/rule_engine.py` — 12+ rules with `rule_id`, trigger flag, severity weight.
- **ML engine:** `src/ml_engine.py` — Isolation Forest, normal-only training, anomaly score per timestamp.
- **Correlation:** `src/correlation_engine.py` — `final_severity`, `confidence_score`, `anomaly_summary`.
- **Checklist:** `src/checklist_engine.py` — 30+ items, checklist risk 0–100; combined risk formula.
- **Alerts:** `src/alert_engine.py` — JSON alerts with `recommended_action` by severity.
- **SOC dashboard:** `ui/dashboard.py` — Streamlit-based analyst console (see below).

---

## 4. Rule-Based vs ML Comparison

| Aspect              | Rule-based engine                    | ML engine (Isolation Forest)        |
|---------------------|--------------------------------------|-------------------------------------|
| Input               | Telemetry rows and session aggregates| Feature matrix (voltage, power, …)  |
| Logic               | Deterministic thresholds and checks  | Learned “normal” region              |
| Interpretability    | High (explicit rule_id and weight)   | Lower (score only)                   |
| Novel patterns      | Only those matching rules            | Can flag unseen deviation           |
| False positives     | Tunable via thresholds               | Depends on contamination/threshold  |

The **correlation engine** uses both: rule triggers and ML score are combined so that (e.g.) rule triggered + high ML score → CRITICAL; multiple rules → HIGH; ML-only → MEDIUM; no triggers → LOW. This hybrid design balances explainability (rules) with adaptability (ML).

---

## 4a. Recall vs Precision Trade-off (Security Perspective)

In critical infrastructure and SOC monitoring, **missed attacks are more critical than false positives**. A low-recall system may leave real intrusions undetected, whereas false positives can be triaged by analysts. This project therefore implements a **recall-aware** correlation engine:

- **ML-only path:** If the ML anomaly score exceeds an adaptive (percentile-based) threshold, the session is assigned **at least MEDIUM** severity even when no rules fire. This reduces rule dominance and ensures that ML-detected anomalies are not downgraded to LOW.
- **Positive class:** Evaluations treat MEDIUM, HIGH, and CRITICAL as “positive” (actionable), so that recall measures the fraction of true anomalies that receive any actionable severity.
- **Explicit design choice:** False positives are intentionally tolerated to reduce missed attacks. Analysts use the SOC dashboard to filter and investigate; the system prioritizes surfacing potential incidents.

---

## 4b. Temporal Correlation in IDS

The correlation engine implements **sliding-window temporal correlation** per station:

- For each session, the system counts how many other sessions at the **same station** started within the last **10 minutes**.
- If **≥ 3** such sessions occur in that window, **severity is escalated** (e.g. MEDIUM → HIGH, HIGH → CRITICAL).
- This helps detect patterns such as session flooding or repeated probing at a single station. State is maintained per `station_id`; no cross-station state is required.

---

## 5. Risk Scoring Methodology

- **Runtime risk (0–100):** Derived from `final_severity` and `confidence_score` (e.g. LOW→15, MEDIUM→40, HIGH→70, CRITICAL→95, scaled by confidence).
- **Checklist risk (0–100):** Weighted sum of failed items over total weight (30+ items: physical security + cybersecurity configuration). Each item has description, weight, and status (pass/fail).
- **Final risk:**  
  **Final Risk = 0.6 × runtime_risk + 0.4 × checklist_risk**  
  Used in alerts as `risk_score`.

---

## 6. Recommended Actions by Severity

| Severity  | Recommended action                 |
|-----------|------------------------------------|
| LOW       | Monitor                            |
| MEDIUM    | Log and review                     |
| HIGH      | Temporarily suspend session        |
| CRITICAL  | Flag station for inspection       |

---

## 7. Analyst-Oriented Visualization (SOC Dashboard)

A lightweight **Streamlit SOC Analyst Console** (`ui/dashboard.py`) provides:

- **System overview:** Total sessions, anomalies detected, counts by severity (Critical / High / Medium / Low), average risk score.
- **Incident table:** Loads `results/alerts.json`; sortable table with station_id, anomaly_type, severity, risk_score, confidence, recommended_action.
- **Time series by station:** Select a station; plot `power_kw` and `energy_kwh` over time; anomalous sessions (MEDIUM+ severity) are indicated and listed.
- **Checklist risk summary:** Checklist risk score with color coding: green (&lt; 30), yellow (30–60), red (&gt; 60).
- **Security disclaimer:** States that the system prioritizes recall over false positives for critical infrastructure monitoring.

Run from the project root: `streamlit run ui/dashboard.py`. No authentication; research prototype only.

---

## SOC Analyst Dashboard

### SOC Console Views

Screenshots of the Streamlit SOC console:

![SOC overview](diagrams/soc-overview.png)

*Figure 1: SOC overview panel summarizing detected anomalies and risk levels.*

![SOC incident table](diagrams/soc-incident-table.png)

*Figure 2: Incident table with station_id, anomaly type, severity, risk score, and recommended action.*

![SOC time-series](diagrams/soc-timeseries.png)

*Figure 3: Time-series anomaly visualization for a selected station (power_kw and energy_kwh).*

---

## 8. Project Structure

```
ev-charging-security-ai/
├── data/
│   └── simulated_telemetry.csv
├── src/
│   ├── data_simulator.py
│   ├── rule_engine.py
│   ├── ml_engine.py
│   ├── correlation_engine.py
│   ├── checklist_engine.py
│   ├── alert_engine.py
├── experiments/
│   └── run_pipeline.py
├── ui/
│   └── dashboard.py
├── results/
│   ├── alerts.json
│   ├── evaluation.txt
│   ├── summary.json
├── diagrams/
├── README.md
└── requirements.txt
```

---

## 9. Usage

From the project root:

```bash
pip install -r requirements.txt
python experiments/run_pipeline.py
streamlit run ui/dashboard.py
```

The pipeline generates (or loads) telemetry, runs the rule engine, trains the ML engine on normal data, runs the recall-aware correlation engine (adaptive ML threshold, temporal window), computes checklist and combined risk, and writes:

- `results/alerts.json` — structured alerts per session.
- `results/evaluation.txt` — detection metrics (precision, recall, F1) and recall-focused notes.
- `results/summary.json` — overview metrics for the dashboard.

The dashboard reads these results and `data/simulated_telemetry.csv` for the SOC console.

---

## System Evaluation Summary

### 1. Evaluation Setup

- **Synthetic EV charging sessions:** 50 normal and 20 anomalous sessions with time-series telemetry at 1-minute intervals; anomaly types include power spike, meter manipulation, session flooding, interrupted charging, unrealistic energy growth, and flatline meter.
- **Rule-based + ML hybrid IDS:** Deterministic rules (voltage, power, energy consistency, meter, session behavior) run first; Isolation Forest trained on normal data only produces per-timestamp anomaly scores; correlation engine combines both.
- **Temporal correlation enabled:** Sliding window of 10 minutes per station; ≥3 sessions from the same station in the window trigger severity escalation.
- **Recall-prioritized decision logic:** Adaptive ML threshold (85th percentile); ML-only path assigns at least MEDIUM severity when no rules fire, so that ML-detected anomalies are not missed.

### 2. Key Metrics (Representative)

| Metric    | Value  |
|----------|--------|
| Precision | ≈ 0.38 |
| Recall    | ≈ 0.85 |
| F1-score  | ≈ 0.52 |

Positive class: MEDIUM, HIGH, or CRITICAL severity (actionable alerts).

### 3. Security Interpretation

- **Why recall is prioritized over precision:** In critical infrastructure cybersecurity, undetected attacks pose a greater systemic risk than false alerts. A high-recall system surfaces more potential incidents for analyst triage; a high-precision, low-recall system may leave real intrusions unaddressed.
- **Why false positives are acceptable:** False positives consume analyst time but do not leave the infrastructure exposed. They can be filtered, reviewed, and dismissed; missed attacks cannot be recovered after the fact.
- **Why missed attacks are not acceptable:** A single undetected compromise can lead to safety failures, fraud, or cascading outages. The design therefore tolerates additional alerts to minimize the probability of missed detection.

*In critical infrastructure cybersecurity, undetected attacks pose a greater systemic risk than false alerts.*

### 4. ML-Only Detection Insight

A significant portion of anomalies in the evaluation were detected **solely by the ML engine** (no rules fired). This demonstrates resilience against rule evasion: attackers or faults that do not trigger any of the predefined rules can still be flagged by the learned normal-behavior model. The correlation engine explicitly surfaces this via the ML-only path and the reported count of *anomalies detected only by ML* in the evaluation output.

### 5. Limitations

- **Synthetic data:** All sessions and telemetry are simulated; no real charging stations or vehicles.
- **No live OCPP traffic:** The system does not ingest real Open Charge Point Protocol or other charging-protocol traffic.
- **Prototype-level deployment:** Intended for research and academic review; not production-hardened (e.g. no authentication, no formal deployment model).

---

## 10. Limitations and Future Work

- **Simulation only:** All data and stations are synthetic; no real charging protocols or hardware.
- **Threshold tuning:** The adaptive ML threshold (e.g. 85th percentile) and temporal window (10 minutes, ≥3 sessions) are configurable but not auto-tuned; operational deployment would benefit from validation on labeled data.
- **Checklist:** The security checklist is static and simulated (random pass/fail); in production it would be filled from real asset management or compliance data.
- **Dashboard:** No authentication or access control; suitable only for research and lab use. Future work could add role-based views, alert filtering, and integration with ticketing.

---

## 11. Ethical Disclaimer

**Simulation only.** This project uses **fully simulated data and simulated EV charging environments**. No real charging stations, hardware, or live systems are used or connected. The system is intended for **academic and research purposes** to demonstrate hybrid rule-based and AI-driven security monitoring design. It must not be used to make operational decisions on real infrastructure without appropriate validation, governance, and authorization.

**Recall-over-precision:** False positives are intentionally tolerated to reduce missed attacks; the design prioritizes recall for critical infrastructure cybersecurity monitoring.
