"""
Supervised training pipeline for food portion mass regression on Nutrition5k.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mass_regressor import FoodMassRegressionModel
from src.portion.dataset import Nutrition5kMassDataset


def train_mass_regressor(
    manifest_path: str | Path,
    backbone: str = "efficientnet_b0",
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    device: str = "cpu",
    save_dir: str | Path = "models/portion",
    checkpoint_name: str = "best_mass_regressor.pt",
) -> dict[str, Any]:
    """
    Train FoodMassRegressionModel to predict portion mass in grams.
    """
    mpath = Path(manifest_path).resolve()
    if not mpath.is_file():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    train_samples = [s for s in samples if s.get("split") == "train"] or samples
    val_samples = [s for s in samples if s.get("split") == "val"] or samples

    train_ds = Nutrition5kMassDataset(train_samples, is_training=True)
    val_ds = Nutrition5kMassDataset(val_samples, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FoodMassRegressionModel(
        backbone=backbone,
        in_channels=3,
        pretrained=True,
    ).to(device)

    criterion = nn.SmoothL1Loss()  # Huber loss is robust to mass outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    save_path = Path(save_dir) / checkpoint_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting Portion Mass Regression training...")
    print(f"  Backbone: {backbone}")
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples_count = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.float().unsqueeze(-1).to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_samples_count += imgs.size(0)

        avg_train_loss = train_loss / max(1, train_samples_count)

        # Validation step
        model.eval()
        val_abs_error = 0.0
        val_samples_count = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.float().unsqueeze(-1).to(device)
                preds = model(imgs)
                val_abs_error += torch.abs(preds - targets).sum().item()
                val_samples_count += imgs.size(0)

        val_mae = val_abs_error / max(1, val_samples_count)
        print(f"  Epoch {epoch:02d}/{epochs:02d} - Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.2f}g")

        if val_mae < best_mae:
            best_mae = val_mae
            model.save_checkpoint(save_path, extra_meta={"best_val_mae": best_mae, "epoch": epoch})
            print(f"  -> Saved best mass regressor checkpoint to {save_path}")

    return {
        "status": "success",
        "best_mae": round(best_mae, 2),
        "checkpoint": str(save_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train food portion mass regression model.")
    parser.add_argument("--manifest", required=True, help="Path to mass manifest JSON")
    parser.add_argument("--backbone", default="efficientnet_b0", help="Model backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out", default="models/portion/best_mass_regressor.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_mass_regressor(
        manifest_path=args.manifest,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        save_dir=Path(args.out).parent,
        checkpoint_name=Path(args.out).name,
    )


if __name__ == "__main__":
    main()
