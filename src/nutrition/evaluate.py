"""
Comprehensive Evaluation Engine for Multi-Nutrient Estimation:
Calculates MAE, RMSE, R^2, and MAPE across Calories, Protein, Carbohydrates, and Fat.
Generates:
  - reports/actual_vs_predicted_calories.png
  - reports/actual_vs_predicted_protein.png
  - reports/actual_vs_predicted_carbohydrates.png
  - reports/actual_vs_predicted_fat.png
  - reports/nutrient_error_distributions.png
  - Error distributions, per-food error breakdowns, worst and best predictions.
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

from src.nutrition.dataset import Nutrition5kDataset
from src.nutrition.model import MultiNutrientModel

TARGET_NAMES = ["calories_kcal", "protein_g", "carbohydrates_g", "fat_g"]
TARGET_UNITS = ["kcal", "g", "g", "g"]
TARGET_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]


def calculate_metric_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    unit: str,
    eps: float = 1.0,
) -> dict[str, Any]:
    """
    Calculate MAE, RMSE, R^2, and MAPE for a single continuous nutrient target.
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    if len(y_t) == 0:
        return {
            f"mae_{unit}": 0.0,
            f"rmse_{unit}": 0.0,
            "r2_score": 0.0,
            "mape_percent": 0.0,
            "samples_evaluated": 0,
        }

    errors = np.abs(y_p - y_t)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))

    # R^2 Score
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-8, ss_tot))) if ss_tot > 0 else 0.0

    # MAPE with protective epsilon against zero-macro ground truths
    mape = float(np.mean(errors / np.maximum(y_t, eps)) * 100.0)

    # Error distribution stats
    error_percentiles = {
        "p25": round(float(np.percentile(errors, 25)), 2),
        "median": round(float(np.median(errors)), 2),
        "p75": round(float(np.percentile(errors, 75)), 2),
        "p90": round(float(np.percentile(errors, 90)), 2),
        "std": round(float(np.std(errors)), 2),
    }

    return {
        f"mae_{unit}": round(mae, 2),
        f"rmse_{unit}": round(rmse, 2),
        "r2_score": round(r2, 4),
        "mape_percent": round(mape, 2),
        "samples_evaluated": int(len(y_t)),
        f"mean_actual_{unit}": round(float(np.mean(y_t)), 2),
        f"mean_predicted_{unit}": round(float(np.mean(y_p)), 2),
        "error_distribution": error_percentiles,
    }


def plot_target_actual_vs_predicted(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    target_name: str,
    unit: str,
    color: str,
    output_path: str | Path,
) -> Path:
    """Generate and save scatter plot with 1:1 reference identity line."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_t, y_p, color=color, alpha=0.8, edgecolors="k", s=50, label="Dish Samples")

    min_val = min(float(y_t.min() if len(y_t) else 0), float(y_p.min() if len(y_p) else 0))
    max_val = max(float(y_t.max() if len(y_t) else 100), float(y_p.max() if len(y_p) else 100))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Perfect 1:1")

    ax.set_xlabel(f"Actual {target_name} ({unit})", fontweight="bold")
    ax.set_ylabel(f"Predicted {target_name} ({unit})", fontweight="bold")
    ax.set_title(f"Actual vs. Predicted {target_name}", pad=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_all_error_distributions(
    errors_dict: dict[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Generate 4-panel histogram of error distributions across all 4 targets."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, name in enumerate(TARGET_NAMES):
        ax = axes[i]
        errs = errors_dict.get(name, np.array([0.0]))
        unit = TARGET_UNITS[i]
        color = TARGET_COLORS[i]

        ax.hist(errs, bins=10, color=color, edgecolor="black", alpha=0.75)
        ax.set_title(f"{name} Absolute Errors ({unit})", fontweight="bold")
        ax.set_xlabel(f"Error ({unit})")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def evaluate_nutrition_on_manifest(
    manifest_path: str | Path,
    weights_path: str | Path = "models/nutrition/best_nutrient_model.pt",
    vocab_path: str | Path | None = None,
    device: str = "cpu",
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    """
    Run evaluation on nutrition dataset manifest using MultiNutrientModel.
    Generates all 4 scatter plots, error distribution plot, metrics JSON/CSV,
    and identifies worst/best predictions.
    """
    mpath = Path(manifest_path).resolve()
    with open(mpath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    vocab: list[str] = []
    if vocab_path and Path(vocab_path).is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    val_samples = [s for s in samples if s.get("split") == "val"] or samples
    val_ds = Nutrition5kDataset(val_samples, ingredient_vocab=vocab, is_training=False)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    wpath = Path(weights_path)
    if wpath.is_file():
        model = MultiNutrientModel.load_from_checkpoint(wpath, device=device)
    else:
        model = MultiNutrientModel(backbone="efficientnet_b0", pretrained=False).to(device)

    model.eval()
    all_targets: list[list[float]] = []
    all_preds: list[list[float]] = []

    with torch.no_grad():
        for batch in loader:
            crops = batch["visual_crop"].to(device)
            c_ids = batch["class_id"].to(device)
            mass = batch["mass"].to(device)
            seg = batch["seg_features"].to(device)
            depth = batch["depth_features"].to(device)
            ingr = batch["ingr_features"].to(device)
            targets = batch["targets"].to(device)

            preds = model(crops, class_ids=c_ids, masses=mass, seg_features=seg, depth_features=depth, ingr_features=ingr)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    y_t = np.array(all_targets)  # (N, 4)
    y_p = np.array(all_preds)    # (N, 4)

    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    metrics_per_target: dict[str, Any] = {}
    errors_per_target: dict[str, np.ndarray] = {}

    for idx, (name, unit, color) in enumerate(zip(TARGET_NAMES, TARGET_UNITS, TARGET_COLORS)):
        actual_col = y_t[:, idx]
        pred_col = y_p[:, idx]
        errors = np.abs(pred_col - actual_col)
        errors_per_target[name] = errors

        # Compute metric suite
        metrics_per_target[name] = calculate_metric_suite(actual_col, pred_col, target_name=name, unit=unit)

        # Plot individual actual vs. predicted scatter
        scatter_path = out_d / f"actual_vs_predicted_{name.replace('_kcal', '').replace('_g', '')}.png"
        plot_target_actual_vs_predicted(actual_col, pred_col, target_name=name, unit=unit, color=color, output_path=scatter_path)

    # Plot error distributions
    error_plot_path = out_d / "nutrient_error_distributions.png"
    plot_all_error_distributions(errors_per_target, error_plot_path)

    # Calculate overall calorie-normalized composite error per sample
    # Composite error = sum of normalized errors across 4 targets
    scales = np.array([500.0, 30.0, 50.0, 25.0])
    sample_norm_errors = np.sum(np.abs(y_p - y_t) / scales, axis=1)

    sorted_indices = np.argsort(sample_norm_errors)
    best_indices = sorted_indices[:5].tolist()
    worst_indices = sorted_indices[-5:][::-1].tolist()

    best_predictions = []
    for rank, idx in enumerate(best_indices, 1):
        s = val_samples[idx] if idx < len(val_samples) else {}
        best_predictions.append({
            "rank": rank,
            "food_name": s.get("food_name", f"sample_{idx}"),
            "composite_error": round(float(sample_norm_errors[idx]), 4),
            "actual": {TARGET_NAMES[j]: round(float(y_t[idx, j]), 1) for j in range(4)},
            "predicted": {TARGET_NAMES[j]: round(float(y_p[idx, j]), 1) for j in range(4)},
        })

    worst_predictions = []
    for rank, idx in enumerate(worst_indices, 1):
        s = val_samples[idx] if idx < len(val_samples) else {}
        worst_predictions.append({
            "rank": rank,
            "food_name": s.get("food_name", f"sample_{idx}"),
            "composite_error": round(float(sample_norm_errors[idx]), 4),
            "actual": {TARGET_NAMES[j]: round(float(y_t[idx, j]), 1) for j in range(4)},
            "predicted": {TARGET_NAMES[j]: round(float(y_p[idx, j]), 1) for j in range(4)},
        })

    # Per-food errors breakdown
    food_names = [s.get("food_name", "unknown") for s in val_samples]
    per_food_errors: dict[str, list[float]] = {}
    for name, err in zip(food_names, sample_norm_errors):
        per_food_errors.setdefault(name, []).append(float(err))
    mean_per_food_errors = {k: round(float(np.mean(v)), 4) for k, v in per_food_errors.items()}

    final_report = {
        "targets": metrics_per_target,
        "per_food_composite_errors": mean_per_food_errors,
        "best_predictions": best_predictions,
        "worst_predictions": worst_predictions,
        "plots_generated": [
            str(out_d / "actual_vs_predicted_calories.png"),
            str(out_d / "actual_vs_predicted_protein.png"),
            str(out_d / "actual_vs_predicted_carbohydrates.png"),
            str(out_d / "actual_vs_predicted_fat.png"),
            str(out_d / "nutrient_error_distributions.png"),
        ],
    }

    # Save JSON report
    jpath = out_d / "nutrition_metrics.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Save summary CSV
    cpath = out_d / "nutrition_metrics.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "metric", "value"])
        for tgt, mdict in metrics_per_target.items():
            for k, v in mdict.items():
                if k != "error_distribution":
                    writer.writerow([tgt, k, v])

    print(f"Saved multi-nutrient metrics to: {jpath}")
    print(f"Saved multi-nutrient summary to: {cpath}")
    return final_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-nutrient estimation model.")
    parser.add_argument("--manifest", required=True, help="Path to nutrition manifest JSON")
    parser.add_argument("--weights", default="models/nutrition/best_nutrient_model.pt", help="Checkpoint path")
    parser.add_argument("--vocab", default=None, help="Ingredient vocab JSON path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="reports", help="Reports output directory")
    args = parser.parse_args()

    evaluate_nutrition_on_manifest(
        manifest_path=args.manifest,
        weights_path=args.weights,
        vocab_path=args.vocab,
        device=args.device,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
