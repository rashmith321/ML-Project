"""
Visualization utilities for food classification.
Renders confusion matrix heatmaps and annotated classified crops.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Union

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.classification.classifier import ClassificationResult


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path = "reports/confusion_matrix.png",
    title: str = "Food Classification Confusion Matrix",
) -> Path:
    """
    Generate and save a confusion matrix heatmap image.
    """
    matrix = np.array(cm)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.8)))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Label text inside cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = matrix[i, j]
            ax.text(j, i, str(val), va="center", ha="center", color="black" if val < matrix.max() / 2 else "white")

    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(title, pad=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved confusion matrix plot to: {out}")
    return out


def draw_classification_badge(
    image: Image.Image,
    result: ClassificationResult,
    position: tuple[int, int] = (10, 10),
) -> Image.Image:
    """
    Draw a stylish badge onto an image crop showing the top-1 prediction
    and top-k alternatives.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    lines = [f"{result.food_name.upper()} ({result.confidence * 100:.1f}%)"]
    for alt in result.top_k_predictions[1:3]:
        lines.append(f"  alt: {alt['food_name']} ({alt['confidence'] * 100:.1f}%)")

    badge_text = "\n".join(lines)
    x, y = position

    # Draw semi-transparent background
    draw.rectangle([x, y, x + 210, y + len(lines) * 16 + 6], fill=(20, 20, 20))
    draw.text((x + 6, y + 4), badge_text, fill=(240, 240, 240))

    return img
