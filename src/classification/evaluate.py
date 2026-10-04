"""
Evaluation metrics and confusion matrix generator for food classification:
Accuracy, Top-1, Top-5, Precision, Recall, F1, and Confusion Matrix.
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

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.classification.dataset import FoodCropDataset
from src.classification.taxonomy import FoodTaxonomy
from src.models.classifier import FoodClassificationModel


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Compute comprehensive classification metrics and confusion matrix.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    if len(y_t) == 0:
        return {
            "accuracy": 0.0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "confusion_matrix": [],
        }

    acc = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Compute Top-k accuracy if probability matrix is provided
    top_k_acc = None
    if y_prob is not None and y_prob.ndim == 2:
        k = min(top_k, y_prob.shape[1])
        top_k_preds = np.argsort(-y_prob, axis=1)[:, :k]
        hits = sum(t in row for t, row in zip(y_t, top_k_preds))
        top_k_acc = float(hits / len(y_t))

    unique_labels = [int(x) for x in sorted(list(set(y_t) | set(y_p)))]
    raw_cm = confusion_matrix(y_t, y_p, labels=unique_labels).tolist()
    cm = [[int(val) for val in row] for row in raw_cm]

    return {
        "accuracy": round(acc, 4),
        "top1_accuracy": round(acc, 4),
        "top5_accuracy": round(top_k_acc if top_k_acc is not None else acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "samples_evaluated": len(y_t),
        "labels": unique_labels,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_csv(
    cm: list[list[int]],
    labels: list[int],
    class_names: Sequence[str] | None,
    output_path: str | Path,
) -> Path:
    """Save confusion matrix as a labeled CSV table."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = ["true_class"] + [
        (class_names[idx] if class_names and idx < len(class_names) else f"pred_{idx}")
        for idx in labels
    ]

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(cm):
            cls_name = class_names[labels[i]] if class_names and labels[i] < len(class_names) else f"class_{labels[i]}"
            writer.writerow([cls_name] + row)

    print(f"Saved confusion matrix to: {out}")
    return out


def evaluate_classifier_on_manifest(
    manifest_path: str | Path,
    weights_path: str | Path = "models/classification/best_classifier.pt",
    device: str = "cpu",
    output_json: str | Path = "reports/classification_metrics.json",
    output_csv: str | Path = "reports/classification_metrics.csv",
    cm_csv: str | Path = "reports/confusion_matrix.csv",
) -> dict[str, Any]:
    """
    Run evaluation on a dataset manifest using FoodClassificationModel.
    """
    mpath = Path(manifest_path).resolve()
    with open(mpath) as f:
        samples = json.load(f)

    val_samples = [s for s in samples if s.get("split") == "val"] or samples
    val_ds = FoodCropDataset(val_samples, is_training=False)
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    wpath = Path(weights_path)
    if wpath.is_file():
        model = FoodClassificationModel.load_from_checkpoint(wpath, device=device)
    else:
        # Pretrained fallback
        num_classes = max(s["class_id"] for s in samples) + 1 if samples else 101
        model = FoodClassificationModel(num_classes=num_classes, pretrained=True).to(device)

    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            probs = model.predict_probs(imgs).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    y_prob = np.vstack(all_probs) if all_probs else None
    taxonomy = FoodTaxonomy(primary_dataset="food101")

    metrics = compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=y_prob,
        class_names=taxonomy.classes,
    )

    # Save JSON report
    jpath = Path(output_json)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save summary CSV
    cpath = Path(output_csv)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ["accuracy", "top1_accuracy", "top5_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            writer.writerow([k, metrics.get(k, 0.0)])

    # Save confusion matrix CSV
    save_confusion_matrix_csv(
        cm=metrics["confusion_matrix"],
        labels=metrics["labels"],
        class_names=taxonomy.classes,
        output_path=cm_csv,
    )

    print(f"Saved classification metrics to:")
    print(f"  {jpath}")
    print(f"  {cpath}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate food classification model.")
    parser.add_argument("--manifest", required=True, help="Path to classification manifest JSON")
    parser.add_argument("--weights", default="models/classification/best_classifier.pt", help="Checkpoint path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="reports/classification_metrics.json")
    parser.add_argument("--csv", default="reports/classification_metrics.csv")
    parser.add_argument("--cm-csv", default="reports/confusion_matrix.csv")
    args = parser.parse_args()

    evaluate_classifier_on_manifest(
        manifest_path=args.manifest,
        weights_path=args.weights,
        device=args.device,
        output_json=args.json,
        output_csv=args.csv,
        cm_csv=args.cm_csv,
    )


if __name__ == "__main__":
    main()
