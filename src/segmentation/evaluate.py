"""
Evaluation utilities and metrics for food image segmentation:
IoU, Dice coefficient, Precision, and Recall.
Generates metrics in both JSON and CSV format.
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
from PIL import Image

from src.segmentation.segmenter import FoodSegmenter


def calculate_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute binary segmentation metrics between predicted mask and ground truth mask.
    Returns: {"iou": float, "dice": float, "precision": float, "recall": float}
    """
    pred_bin = (pred_mask > 0.5).astype(bool)
    gt_bin = (gt_mask > 0.5).astype(bool)

    intersection = float(np.logical_and(pred_bin, gt_bin).sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())

    iou = float((intersection + eps) / (union + eps)) if union > 0 else (1.0 if gt_sum == 0 else 0.0)
    dice = float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps)) if (pred_sum + gt_sum) > 0 else (1.0 if gt_sum == 0 else 0.0)
    precision = float((intersection + eps) / (pred_sum + eps)) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = float((intersection + eps) / (gt_sum + eps)) if gt_sum > 0 else 1.0

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def save_segmentation_metrics(
    metrics: dict[str, Any],
    json_path: str | Path = "reports/segmentation_metrics.json",
    csv_path: str | Path = "reports/segmentation_metrics.csv",
) -> None:
    """Save segmentation metrics to JSON and CSV files."""
    jpath = Path(json_path)
    cpath = Path(csv_path)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(cpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mean_iou", metrics.get("mean_iou", 0.0)])
        writer.writerow(["mean_dice", metrics.get("mean_dice", 0.0)])
        writer.writerow(["mean_precision", metrics.get("mean_precision", 0.0)])
        writer.writerow(["mean_recall", metrics.get("mean_recall", 0.0)])
        writer.writerow(["samples_evaluated", metrics.get("samples_evaluated", 0)])

    print(f"Saved segmentation evaluation metrics to:")
    print(f"  {jpath}")
    print(f"  {cpath}")


def evaluate_segmenter_on_samples(
    samples: Sequence[dict[str, Any]],
    segmenter: FoodSegmenter,
    output_json: str | Path = "reports/segmentation_metrics.json",
    output_csv: str | Path = "reports/segmentation_metrics.csv",
) -> dict[str, Any]:
    """
    Evaluate a FoodSegmenter instance across a sequence of image/mask sample pairs.
    """
    all_iou: list[float] = []
    all_dice: list[float] = []
    all_prec: list[float] = []
    all_rec: list[float] = []

    for item in samples:
        img_p = Path(item["image_path"])
        mask_p = Path(item["mask_path"])
        if not (img_p.is_file() and mask_p.is_file()):
            continue

        gt_mask = np.array(Image.open(mask_p).convert("L")) > 127
        h, w = gt_mask.shape

        # Use full-image bounding box if no detection box is present
        bbox = item.get("bbox", [0.0, 0.0, float(w), float(h)])
        pred_mask = segmenter.segment_from_box(img_p, bbox)

        # Ensure shapes match
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(Image.fromarray(pred_mask).resize((w, h), Image.NEAREST))

        m = calculate_segmentation_metrics(pred_mask, gt_mask)
        all_iou.append(m["iou"])
        all_dice.append(m["dice"])
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])

    mean_metrics = {
        "mean_iou": round(float(np.mean(all_iou)) if all_iou else 0.0, 4),
        "mean_dice": round(float(np.mean(all_dice)) if all_dice else 0.0, 4),
        "mean_precision": round(float(np.mean(all_prec)) if all_prec else 0.0, 4),
        "mean_recall": round(float(np.mean(all_rec)) if all_rec else 0.0, 4),
        "samples_evaluated": len(all_iou),
    }

    save_segmentation_metrics(mean_metrics, json_path=output_json, csv_path=output_csv)
    return mean_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate food image segmentation.")
    parser.add_argument("--manifest", required=True, help="Path to segmentation manifest JSON")
    parser.add_argument("--mode", default="sam", choices=["sam", "unet", "auto"])
    parser.add_argument("--unet-checkpoint", default="models/segmentation/best_unet.pt")
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="reports/segmentation_metrics.json")
    parser.add_argument("--csv", default="reports/segmentation_metrics.csv")
    args = parser.parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)

    segmenter = FoodSegmenter(
        mode=args.mode,
        sam_checkpoint=args.sam_checkpoint,
        unet_checkpoint=args.unet_checkpoint,
        device=args.device,
    )
    segmenter.load()

    metrics = evaluate_segmenter_on_samples(
        samples=samples,
        segmenter=segmenter,
        output_json=args.json,
        output_csv=args.csv,
    )
    print("Segmentation Evaluation Results:", metrics)


if __name__ == "__main__":
    main()
