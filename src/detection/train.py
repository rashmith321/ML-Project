"""
Training script for food object detection using YOLO transfer learning.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import yaml

from src.utils.config import load_config


def train_detector(
    data_yaml: str | Path,
    model_name_or_weights: str | Path = "models/yolov8n.pt",
    epochs: int = 10,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "cpu",
    project: str | Path = "outputs/detection/train",
    name: str = "exp",
    best_model_out: str | Path | None = "models/detection/best.pt",
    lr0: float = 0.01,
) -> dict[str, Any]:
    """
    Train a YOLO detector using transfer learning.
    """
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    # Check if weights exist locally in models/ or fallback to model_name
    weights_path = Path(model_name_or_weights)
    if not weights_path.is_file() and (Path("models") / str(model_name_or_weights)).is_file():
        weights_path = Path("models") / str(model_name_or_weights)

    model = YOLO(str(weights_path))

    project_dir = Path(project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting YOLO transfer learning training...")
    print(f"  Dataset: {yaml_path}")
    print(f"  Base weights: {weights_path}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}, Device: {device}")

    results = model.train(
        data=str(yaml_path).replace("\\", "/"),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(project_dir).replace("\\", "/"),
        name=name,
        lr0=lr0,
        exist_ok=True,
        verbose=True,
    )

    # Save best checkpoint to target models path
    train_run_dir = project_dir / name
    best_pt = train_run_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        # Fallback to last.pt if best.pt is not present
        best_pt = train_run_dir / "weights" / "last.pt"

    saved_path = None
    if best_pt.is_file() and best_model_out:
        out_path = Path(best_model_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, out_path)
        saved_path = str(out_path)
        print(f"Saved trained detector checkpoint to: {saved_path}")

    return {
        "status": "success",
        "checkpoint": saved_path,
        "run_dir": str(train_run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train food object detector.")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--weights", default="models/yolov8n.pt", help="Pretrained weights or model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--project", default="outputs/detection/train", help="Run output directory")
    parser.add_argument("--name", default="exp", help="Run name")
    parser.add_argument("--out", default="models/detection/best.pt", help="Output path for best checkpoint")
    args = parser.parse_args()

    train_detector(
        data_yaml=args.data,
        model_name_or_weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        best_model_out=args.out,
    )


if __name__ == "__main__":
    main()
