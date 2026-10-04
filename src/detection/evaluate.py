"""
Evaluation utilities and metrics for food object detection:
Precision, Recall, mAP@50, and mAP@50:95.
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
from ultralytics import YOLO


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using standard 11-point interpolation or all-points envelope."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))
    return ap


def compute_detection_metrics(
    all_ground_truths: list[list[dict]],
    all_predictions: list[list[dict]],
    class_names: Sequence[str] | None = None,
    iou_thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> dict[str, Any]:
    """
    Compute Precision, Recall, mAP@50, and mAP@50:95 across a collection of images.
    Each ground truth entry: {"class_id": int, "bbox": [x1, y1, x2, y2]}
    Each prediction entry: {"class_id": int, "bbox": [x1, y1, x2, y2], "confidence": float}
    """
    all_class_ids = set()
    for gts in all_ground_truths:
        for g in gts:
            all_class_ids.add(g["class_id"])
    for preds in all_predictions:
        for p in preds:
            all_class_ids.add(p["class_id"])

    if not all_class_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "per_class": {},
        }

    per_class_results = {}
    ap50_list = []
    ap_all_list = []
    total_tp50 = 0
    total_fp50 = 0
    total_gt = 0

    for cls_id in sorted(all_class_ids):
        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"

        # Flatten predictions for this class across images
        cls_preds = []
        n_gt_cls = 0
        for img_idx, (gts, preds) in enumerate(zip(all_ground_truths, all_predictions)):
            gt_boxes = [g["bbox"] for g in gts if g["class_id"] == cls_id]
            n_gt_cls += len(gt_boxes)
            for p in preds:
                if p["class_id"] == cls_id:
                    cls_preds.append((img_idx, p["confidence"], p["bbox"]))

        total_gt += n_gt_cls
        if n_gt_cls == 0:
            continue

        # Sort predictions descending by confidence
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        ap_per_iou = []
        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched_gts: set[tuple[int, int]] = set()

            for p_idx, (img_idx, conf, p_box) in enumerate(cls_preds):
                gt_boxes = [g["bbox"] for g in all_ground_truths[img_idx] if g["class_id"] == cls_id]
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    iou = box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh and (img_idx, best_gt_idx) not in matched_gts:
                    tp[p_idx] = 1.0
                    matched_gts.add((img_idx, best_gt_idx))
                else:
                    fp[p_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / max(1, n_gt_cls)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)

            ap = compute_ap(recalls, precisions)
            ap_per_iou.append(ap)

            if np.isclose(iou_thresh, 0.50):
                cls_tp50 = int(np.sum(tp))
                cls_fp50 = int(np.sum(fp))
                cls_prec50 = float(cls_tp50 / max(1, cls_tp50 + cls_fp50))
                cls_rec50 = float(cls_tp50 / max(1, n_gt_cls))
                total_tp50 += cls_tp50
                total_fp50 += cls_fp50

        ap50 = ap_per_iou[0] if ap_per_iou else 0.0
        ap_mean = float(np.mean(ap_per_iou)) if ap_per_iou else 0.0
        ap50_list.append(ap50)
        ap_all_list.append(ap_mean)

        per_class_results[cls_name] = {
            "class_id": cls_id,
            "precision@50": round(cls_prec50, 4),
            "recall@50": round(cls_rec50, 4),
            "ap@50": round(ap50, 4),
            "ap@50:95": round(ap_mean, 4),
            "ground_truth_count": n_gt_cls,
        }

    overall_prec50 = float(total_tp50 / max(1, total_tp50 + total_fp50))
    overall_rec50 = float(total_tp50 / max(1, total_gt))
    overall_map50 = float(np.mean(ap50_list)) if ap50_list else 0.0
    overall_map50_95 = float(np.mean(ap_all_list)) if ap_all_list else 0.0

    return {
        "precision": round(overall_prec50, 4),
        "recall": round(overall_rec50, 4),
        "map50": round(overall_map50, 4),
        "map50_95": round(overall_map50_95, 4),
        "per_class": per_class_results,
    }


def save_metrics(
    metrics: dict[str, Any],
    json_path: str | Path = "reports/detection_metrics.json",
    csv_path: str | Path = "reports/detection_metrics.csv",
) -> None:
    """Save metrics to JSON and CSV files."""
    jpath = Path(json_path)
    cpath = Path(csv_path)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.parent.mkdir(parents=True, exist_ok=True)

    with open(jpath, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(cpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["precision@50", metrics.get("precision", 0.0)])
        writer.writerow(["recall@50", metrics.get("recall", 0.0)])
        writer.writerow(["map@50", metrics.get("map50", 0.0)])
        writer.writerow(["map@50:95", metrics.get("map50_95", 0.0)])

        per_class = metrics.get("per_class", {})
        if per_class:
            writer.writerow([])
            writer.writerow(["class_name", "precision@50", "recall@50", "ap@50", "ap@50:95", "support"])
            for name, data in per_class.items():
                writer.writerow([
                    name,
                    data.get("precision@50", 0.0),
                    data.get("recall@50", 0.0),
                    data.get("ap@50", 0.0),
                    data.get("ap@50:95", 0.0),
                    data.get("ground_truth_count", 0),
                ])

    print(f"Saved evaluation metrics to:")
    print(f"  {jpath}")
    print(f"  {cpath}")


def evaluate_detector(
    weights_path: str | Path = "models/yolov8n.pt",
    data_yaml: str | Path | None = None,
    split: str = "val",
    device: str = "cpu",
    output_json: str | Path = "reports/detection_metrics.json",
    output_csv: str | Path = "reports/detection_metrics.csv",
) -> dict[str, Any]:
    """
    Run evaluation using Ultralytics YOLO validator on a data.yaml dataset.
    """
    w_path = Path(weights_path)
    if not w_path.is_file() and (Path("models") / str(weights_path)).is_file():
        w_path = Path("models") / str(weights_path)

    model = YOLO(str(w_path))

    if data_yaml and Path(data_yaml).is_file():
        results = model.val(
            data=str(Path(data_yaml).resolve()).replace("\\", "/"),
            split=split,
            device=device,
            verbose=False,
        )

        metrics = {
            "precision": round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "map50": round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "map50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            "per_class": {},
        }
    else:
        # Default baseline status if no dataset is provided
        metrics = {
            "status": "dataset_not_specified",
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "per_class": {},
        }

    save_metrics(metrics, json_path=output_json, csv_path=output_csv)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate food object detector.")
    parser.add_argument("--weights", default="models/yolov8n.pt", help="Model weights path")
    parser.add_argument("--data", default=None, help="Path to data.yaml")
    parser.add_argument("--split", default="val", help="Split to evaluate on (val or test)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--json", default="reports/detection_metrics.json", help="Output JSON path")
    parser.add_argument("--csv", default="reports/detection_metrics.csv", help="Output CSV path")
    args = parser.parse_args()

    evaluate_detector(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        device=args.device,
        output_json=args.json,
        output_csv=args.csv,
    )


if __name__ == "__main__":
    main()
