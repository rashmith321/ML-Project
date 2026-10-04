"""
Per-stage evaluation metrics (IMPLEMENTATION_PLAN.md section H).
Each metric is reported against its own dataset's ground truth --
never blended across datasets with different label types.
"""
from __future__ import annotations

import numpy as np


def mean_absolute_error(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - targets)))


def mean_absolute_percentage_error(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((preds - targets) / (targets + eps))) * 100.0)


def top_k_accuracy(logits: np.ndarray, targets: np.ndarray, k: int = 1) -> float:
    top_k_preds = np.argsort(-logits, axis=1)[:, :k]
    hits = sum(t in row for t, row in zip(targets, top_k_preds))
    return hits / len(targets)


def mean_iou(pred_masks: np.ndarray, true_masks: np.ndarray) -> float:
    """pred_masks, true_masks: boolean arrays, shape (N, H, W)."""
    intersection = np.logical_and(pred_masks, true_masks).sum(axis=(1, 2))
    union = np.logical_or(pred_masks, true_masks).sum(axis=(1, 2))
    iou = np.where(union > 0, intersection / np.maximum(union, 1), 1.0)
    return float(np.mean(iou))
