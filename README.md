# AI-Based Anomaly Detection System for EV Charging Stations

**Independent, Standalone Project** — This repository contains a complete, self-contained implementation of an AI-based security monitoring system that detects anomalous and potentially malicious behaviors in electric vehicle (EV) charging stations using machine learning. It does **not** reuse or reference any previous team project, OCPP-CAN bridges, blockchain systems, or group work.

---

## 1. Problem Definition

Electric vehicle charging infrastructure is a critical asset. Malicious or faulty behavior can lead to safety risks, fraud, or denial of service. This project addresses the need for **behavioral anomaly detection** in charging sessions: identifying sessions that deviate from normal patterns (e.g., abnormally long or short duration, power spikes, frequent interruptions, inconsistent energy usage, or repeated suspicious activity at the same station) so that operators can prioritize investigation and response from a **cybersecurity and operational security** perspective.

The scope is deliberately limited to **simulated data and offline analysis**: no real hardware or charging stations are used.

---

## 2. Methodology

- **Data simulation:** A Python module generates synthetic EV charging session data with realistic normal behavior and multiple anomaly types (extreme duration, power spikes, frequent interruptions, inconsistent energy, repeated suspicious sessions at the same station). The dataset includes at least 1000 normal and 100 anomalous samples.
- **Feature engineering:** Numerical features are normalized (standardization), and a derived feature (energy consistency ratio) is added to capture implausible energy vs. duration/power relationships. Features are selected and prepared for ML.
- **Anomaly detection model:** **Isolation Forest** is used, trained on **normal data only**. It isolates observations via random splits; anomalies require fewer splits and thus receive higher anomaly scores. This choice is justified by efficiency, scalability, and suitability for mixed numerical features without assuming a specific distribution (unlike One-Class SVM, which is more sensitive to hyperparameters and scale).
- **Evaluation:** The model is evaluated on a held-out test set (stratified split) using **accuracy, precision, recall, and confusion matrix**, with a target of **≥ 95% anomaly detection accuracy**.
- **Risk scoring:** Raw anomaly scores are converted into a **0–100 risk score** and categorized into **Low, Medium, High, and Critical** for operational interpretation.

---

## 3. Project Structure

```
ai-ev-charging-anomaly-detection/
├── data/
│   └── generated_sessions.csv      # Synthetic session data (created by run)
├── src/
│   ├── data_generator.py           # Synthetic EV charging session generation
│   ├── feature_engineering.py      # Normalization, feature selection, preparation
│   ├── anomaly_model.py            # Isolation Forest training and prediction
│   └── risk_scoring.py             # Risk score 0–100 and risk levels
├── experiments/
│   └── run_experiment.py           # End-to-end pipeline and evaluation
├── results/
│   ├── metrics.txt                 # Saved evaluation metrics
│   └── confusion_matrix_and_risk.png  # Plots (created by run)
├── README.md
└── requirements.txt
```

---

## 4. Requirements

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib` (see `requirements.txt`)

---

## 5. Usage

From the project root:

```bash
pip install -r requirements.txt
python experiments/run_experiment.py
```

This will:

1. Generate synthetic session data (if `data/generated_sessions.csv` does not exist) and save it.
2. Split data into train/test (stratified), prepare features, and train Isolation Forest on **normal** training samples only.
3. Predict on the test set, compute accuracy, precision, recall, and confusion matrix.
4. Convert anomaly scores to risk scores (0–100) and risk levels (Low/Medium/High/Critical).
5. Save metrics to `results/metrics.txt`.
6. Print sample detection results to the console.
7. Save a confusion matrix and risk distribution plot to `results/confusion_matrix_and_risk.png`.

---

## 6. Results

After running the experiment, evaluation metrics are in `results/metrics.txt`. The script also prints a summary to the console and sample rows showing `session_id`, `station_id`, features, true label, predicted label, risk score, and risk level. The target is **≥ 95% accuracy** on the test set; the pipeline uses a percentile-based decision threshold (tuned over normal training scores) to meet this target, and the result is reported in the metrics file.

---

## 7. Ethical Disclaimer

**Simulation only.** This project uses **fully simulated/synthetic data**. No real charging stations, hardware, or live systems are used or connected. The system is intended for research, portfolio, and educational purposes to demonstrate anomaly detection and risk scoring methodology in a cybersecurity context. Do not use this software to make decisions on real infrastructure without appropriate validation, governance, and authorization.

---

## 8. License and Authorship

This is an individual, standalone project. All code and documentation are self-contained and do not depend on or reference any prior team or group deliverables.
