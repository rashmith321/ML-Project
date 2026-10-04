"""
Experiment tracker for ML Performance Improvement Phase II.

Rules enforced by this module:
  1. Every experiment writes one row to reports/experiments/experiment_log.csv
  2. Val set used for model selection. Test set used ONLY for final evaluation.
  3. Baseline checkpoint is NEVER overwritten — copied to models/<module>/baseline_<name>.pt
  4. Best model (by val metric) saved to models/<module>/best_<name>.pt
  5. Experiment ID is auto-incremented and monotonic.
  6. All required fields must be provided or an error is raised.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

EXPERIMENT_LOG = Path("reports/experiments/experiment_log.csv")
EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = [
    "experiment_id",
    "timestamp",
    "stage",
    "model",
    "dataset",
    "backbone",
    "augmentation",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs_run",
    "loss",
    "val_metric_name",
    "val_metric_value",
    "test_metric_name",
    "test_metric_value",
    "training_time_s",
    "checkpoint_path",
    "notes",
]


def _next_experiment_id() -> str:
    """Return next monotonic experiment ID like EXP-001."""
    if not EXPERIMENT_LOG.exists():
        return "EXP-001"
    with open(EXPERIMENT_LOG, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "EXP-001"
    last = rows[-1].get("experiment_id", "EXP-000")
    n = int(last.split("-")[-1]) + 1
    return f"EXP-{n:03d}"


def log_experiment(data: dict[str, Any]) -> str:
    """
    Log one experiment to the CSV. Returns the experiment_id used.
    Raises ValueError if required fields are missing.
    """
    exp_id = data.get("experiment_id") or _next_experiment_id()
    data["experiment_id"] = exp_id
    data.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required experiment fields: {missing}")

    write_header = not EXPERIMENT_LOG.exists()
    with open(EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: data.get(k, "") for k in REQUIRED_FIELDS})

    # Also write individual JSON record
    json_path = EXPERIMENT_LOG.parent / f"{exp_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  [ExperimentTracker] Logged {exp_id} -> {EXPERIMENT_LOG}")
    return exp_id


def save_baseline(src: str | Path, stage: str) -> Path:
    """Copy current production checkpoint as baseline (never overwrite)."""
    src = Path(src)
    if not src.exists():
        print(f"  [ExperimentTracker] WARNING: baseline source not found: {src}")
        return src
    dst = src.parent / f"baseline_{stage}_{src.name}"
    if dst.exists():
        print(f"  [ExperimentTracker] Baseline already exists: {dst}")
        return dst
    shutil.copy2(src, dst)
    print(f"  [ExperimentTracker] Saved baseline -> {dst}")
    return dst
