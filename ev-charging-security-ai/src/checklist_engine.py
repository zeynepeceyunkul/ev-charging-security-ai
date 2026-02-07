"""
Checklist-Based Risk Engine.

Static security checklist (physical + cybersecurity configuration).
Each item has description, weight, and status (pass/fail). Produces a
checklist risk score 0-100. Final risk combines runtime and checklist:
Final Risk = 0.6 * runtime_risk + 0.4 * checklist_risk.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

RUNTIME_WEIGHT = 0.6
CHECKLIST_WEIGHT = 0.4


@dataclass
class ChecklistItem:
    id: str
    description: str
    weight: float
    category: str  # "physical" | "cybersecurity"


def get_default_checklist() -> List[ChecklistItem]:
    """Minimum 30 items: physical security and cybersecurity configuration."""
    return [
        ChecklistItem("C01", "Charger in locked or access-controlled enclosure", 1.0, "physical"),
        ChecklistItem("C02", "CCTV coverage of charging bays", 1.0, "physical"),
        ChecklistItem("C03", "Lighting adequate at night", 0.8, "physical"),
        ChecklistItem("C04", "Tamper-evident seals on meter and communication modules", 1.0, "physical"),
        ChecklistItem("C05", "Cable management to prevent trip hazards", 0.5, "physical"),
        ChecklistItem("C06", "Emergency stop accessible and tested", 1.0, "physical"),
        ChecklistItem("C07", "Fire extinguisher suitable for electrical equipment", 0.8, "physical"),
        ChecklistItem("C08", "Fencing or barrier around station perimeter", 0.7, "physical"),
        ChecklistItem("C09", "Signage for EV-only and no ICE parking", 0.3, "physical"),
        ChecklistItem("C10", "Ground fault protection installed and tested", 1.0, "physical"),
        ChecklistItem("C11", "TLS/SSL for all backend communication", 1.0, "cybersecurity"),
        ChecklistItem("C12", "Authentication required for operator and maintenance access", 1.0, "cybersecurity"),
        ChecklistItem("C13", "Firmware signing and secure boot", 0.9, "cybersecurity"),
        ChecklistItem("C14", "No default credentials on any system", 1.0, "cybersecurity"),
        ChecklistItem("C15", "Network segmentation (charging network isolated)", 0.9, "cybersecurity"),
        ChecklistItem("C16", "Intrusion detection or monitoring on charging network", 0.8, "cybersecurity"),
        ChecklistItem("C17", "Rate limiting on API and OCPP endpoints", 0.8, "cybersecurity"),
        ChecklistItem("C18", "Audit logging enabled and retained", 0.9, "cybersecurity"),
        ChecklistItem("C19", "Vulnerability scanning schedule for backend", 0.7, "cybersecurity"),
        ChecklistItem("C20", "Patch management policy for chargers and backend", 0.9, "cybersecurity"),
        ChecklistItem("C21", "Meter calibration certificate current", 0.8, "physical"),
        ChecklistItem("C22", "Communication module in locked compartment", 0.8, "physical"),
        ChecklistItem("C23", "No exposed USB or debug ports", 0.7, "cybersecurity"),
        ChecklistItem("C24", "PKI or certificate-based device authentication", 0.8, "cybersecurity"),
        ChecklistItem("C25", "Payment data not stored on charger", 1.0, "cybersecurity"),
        ChecklistItem("C26", "Session and meter data integrity checks", 0.9, "cybersecurity"),
        ChecklistItem("C27", "Physical inspection log (weekly/monthly)", 0.6, "physical"),
        ChecklistItem("C28", "Access log for maintenance and config changes", 0.8, "cybersecurity"),
        ChecklistItem("C29", "Backup and recovery tested for backend", 0.7, "cybersecurity"),
        ChecklistItem("C30", "Incident response plan documented", 0.8, "cybersecurity"),
        ChecklistItem("C31", "Power quality monitoring (voltage/current limits)", 0.7, "physical"),
        ChecklistItem("C32", "Anti-tamper sensors on meter and comms", 0.9, "physical"),
    ]


def evaluate_checklist(
    items: List[ChecklistItem],
    status_by_id: Dict[str, bool] | None = None,
    random_pass_rate: float = 0.75,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate checklist. status_by_id maps item id -> True (pass) / False (fail).
    If status_by_id is None, use random pass/fail with random_pass_rate for simulation.
    Returns (DataFrame with columns id, description, weight, category, status),
    and checklist_risk score 0-100 (higher = worse, more failures).
    """
    np.random.seed(seed)
    rows = []
    total_weight = 0.0
    failed_weight = 0.0
    for item in items:
        if status_by_id is not None:
            status = status_by_id.get(item.id, True)
        else:
            status = bool(np.random.random() < random_pass_rate)
        total_weight += item.weight
        if not status:
            failed_weight += item.weight
        rows.append({
            "id": item.id,
            "description": item.description,
            "weight": item.weight,
            "category": item.category,
            "status": "pass" if status else "fail",
        })
    df = pd.DataFrame(rows)
    if total_weight <= 0:
        checklist_risk = 0.0
    else:
        checklist_risk = (failed_weight / total_weight) * 100.0
    return df, float(np.clip(checklist_risk, 0, 100))


def runtime_risk_from_severity_and_confidence(
    final_severity: str,
    confidence_score: float,
) -> float:
    """Map severity and confidence to runtime risk 0-100."""
    severity_scores = {"LOW": 15, "MEDIUM": 40, "HIGH": 70, "CRITICAL": 95}
    base = severity_scores.get(final_severity, 20)
    return float(np.clip(base * (0.7 + 0.3 * confidence_score), 0, 100))


def combined_risk(
    runtime_risk: float,
    checklist_risk: float,
    w_runtime: float = RUNTIME_WEIGHT,
    w_checklist: float = CHECKLIST_WEIGHT,
) -> float:
    """Final Risk = w_runtime * runtime_risk + w_checklist * checklist_risk."""
    return float(np.clip(
        w_runtime * runtime_risk + w_checklist * checklist_risk,
        0,
        100,
    ))
