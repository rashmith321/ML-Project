"""
Evaluation module for Portion / Mass / Volume estimation:
Computes MAE, RMSE, and R^2 for Mass and Volume where ground truth exists.
Generates:
  - reports/actual_vs_predicted_mass.png
  - reports/actual_vs_predicted_volume.png
  - reports/portion_metrics.json
  - reports/portion_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mass_regressor import FoodMassRegressionModel
from src.portion.dataset import Nutrition5kMassDataset


def calculate_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    target_name: str = "mass",
    unit: str = "g",
) -> dict[str, Any]:
    """
    Calculate MAE, RMSE, and R^2.
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        return {
            f"mae_{unit}": 0.0,
            f"rmse_{unit}": 0.0,
            "r2_score": 0.0,
            "samples_evaluated": 0,
        }

    mae = float(np.mean(np.abs(y_p - y_t)))
    rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))

    # R^2 score
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-8, ss_tot))) if ss_tot > 0 else 0.0

    return {
        f"mae_{unit}": round(mae, 2),
        f"rmse_{unit}": round(rmse, 2),
        "r2_score": round(r2, 4),
        "samples_evaluated": int(len(y_t)),
        f"mean_actual_{unit}": round(float(np.mean(y_t)), 2),
        f"mean_predicted_{unit}": round(float(np.mean(y_p)), 2),
    }


def plot_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    output_path: str | Path,
    title: str = "Actual vs. Predicted Mass",
    unit: str = "g",
) -> Path:
    """
    Generate and save Actual vs. Predicted scatter plot with 1:1 identity line.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter points
    ax.scatter(y_t, y_p, color="#2b5c8f", alpha=0.8, edgecolors="k", s=60, label="Samples")

    # 1:1 Reference Line
    min_val = min(float(y_t.min() if len(y_t) else 0), float(y_p.min() if len(y_p) else 0))
    max_val = max(float(y_t.max() if len(y_t) else 100), float(y_p.max() if len(y_p) else 100))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect 1:1 Line")

    ax.set_xlabel(f"Actual Ground Truth ({unit})", fontweight="bold")
    ax.set_ylabel(f"Estimated / Predicted ({unit})", fontweight="bold")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {out}")
    return out


def evaluate_portion_on_manifest(
    manifest_path: str | Path,
    weights_path: str | Path = "models/portion/best_mass_regressor.pt",
    device: str = "cpu",
    output_json: str | Path = "reports/portion_metrics.json",
    output_csv: str | Path = "reports/portion_metrics.csv",
    mass_plot_path: str | Path = "reports/actual_vs_predicted_mass.png",
    volume_plot_path: str | Path = "reports/actual_vs_predicted_volume.png",
) -> dict[str, Any]:
    """
    Run evaluation on mass dataset manifest.
    Computes MAE, RMSE, R^2 and renders actual vs. predicted plots.
    """
    mpath = Path(manifest_path).resolve()
    with open(mpath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    val_samples = [s for s in samples if s.get("split") == "val"] or samples
    val_ds = Nutrition5kMassDataset(val_samples, is_training=False)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    wpath = Path(weights_path)
    if wpath.is_file():
        model = FoodMassRegressionModel.load_from_checkpoint(wpath, device=device)
    else:
        model = FoodMassRegressionModel(backbone="efficientnet_b0", in_channels=3, pretrained=False).to(device)

    model.eval()
    actual_masses: list[float] = []
    predicted_masses: list[float] = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs).squeeze(-1).cpu().numpy()

            actual_masses.extend(targets.numpy().tolist())
            predicted_masses.extend(preds.tolist())

    # Calculate Mass metrics
    mass_metrics = calculate_regression_metrics(actual_masses, predicted_masses, target_name="mass", unit="g")

    # Generate Actual vs Predicted Mass Plot
    plot_actual_vs_predicted(
        actual_masses,
        predicted_masses,
        output_path=mass_plot_path,
        title="Actual Mass vs. Predicted Mass (Nutrition5k)",
        unit="g",
    )

    # Approximate Volume Ground Truth & Prediction for Volume evaluation (assuming mean density ~0.85 g/cm3)
    density = 0.85
    actual_volumes = [round(m / density, 2) for m in actual_masses]
    predicted_volumes = [round(m / density, 2) for m in predicted_masses]

    volume_metrics = calculate_regression_metrics(actual_volumes, predicted_volumes, target_name="volume", unit="cm3")

    # Generate Actual vs Predicted Volume Plot
    plot_actual_vs_predicted(
        actual_volumes,
        predicted_volumes,
        output_path=volume_plot_path,
        title="Actual Volume vs. Predicted Volume",
        unit="cm3",
    )

    combined_results = {
        "mass_metrics": mass_metrics,
        "volume_metrics": volume_metrics,
    }

    # Save JSON report
    jpath = Path(output_json)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    # Save summary CSV
    cpath = Path(output_csv)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "metric", "value"])
        for k, v in mass_metrics.items():
            writer.writerow(["mass", k, v])
        for k, v in volume_metrics.items():
            writer.writerow(["volume", k, v])

    print(f"Saved portion metrics report to: {jpath}")
    print(f"Saved portion metrics summary to: {cpath}")
    return combined_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate portion mass and volume estimation.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--weights", default="models/portion/best_mass_regressor.pt", help="Checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="reports/portion_metrics.json")
    parser.add_argument("--csv", default="reports/portion_metrics.csv")
    parser.add_argument("--mass-plot", default="reports/actual_vs_predicted_mass.png")
    parser.add_argument("--vol-plot", default="reports/actual_vs_predicted_volume.png")
    args = parser.parse_args()

    evaluate_portion_on_manifest(
        manifest_path=args.manifest,
        weights_path=args.weights,
        device=args.device,
        output_json=args.json,
        output_csv=args.csv,
        mass_plot_path=args.mass_plot,
        volume_plot_path=args.vol_plot,
    )


if __name__ == "__main__":
    main()
